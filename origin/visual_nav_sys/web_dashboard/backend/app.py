"""
Flask Backend for Visual Navigation System Web Dashboard
Streams video and sensor data to web interface
"""

from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import torch
import numpy as np
import time
import base64
import sys
import os
import psutil
from threading import Thread, Lock

# Add paths
sys.path.insert(0, "../../")
sys.path.insert(0, "../../../accelerated_features")

from modules.xfeat import XFeat
from ultralytics import YOLO
from subsystems.pixhawk import PixhawkIMU
from subsystems.xfeat import process_frame_for_xfeat, estimate_camera_guidance
from subsystems.detection import extract_object_info
from kalman_filter import AttitudeKalmanFilter, PositionKalmanFilter, VelocityKalmanFilter

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
app.config['SECRET_KEY'] = os.environ['VISUAL_NAV_SECRET_KEY']
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
class SystemState:
    def __init__(self):
        self.lock = Lock()
        self.camera = None
        self.xfeat = None
        self.yolo = None
        self.imu = None
        
        # System state
        self.reference_frame = None
        self.reference_tensor = None
        self.reference_gray = None
        self.reference_objects = []
        self.max_matches = 0
        
        # Current frame data
        self.current_data = {
            'fps': 0,
            'num_matches': 0,
            'confidence': 0,
            'guidance_source': 'none',
            'dx': 0,
            'dy': 0,
            'dz_scale': 1.0,
            'yolo_detections': [],
            'status': 'waiting'
        }
        
        # Telemetry running flag
        self.telemetry_running = False

state = SystemState()


def initialize_system():
    """Initialize all subsystems"""
    print("="*60)
    print("Initializing Visual Navigation System...")
    print("="*60)
    
    # Initialize XFeat
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    state.xfeat = XFeat()
    if device == "cuda":
        state.xfeat = state.xfeat.cuda()
    print("✓ XFeat initialized")
    
    # Initialize YOLO with PyTorch 2.6+ compatibility
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
        state.yolo = YOLO("../../yolo11n.pt")
        print("✓ YOLO initialized")
    except Exception as e:
        print(f"⚠ YOLO failed: {e}")
        state.yolo = None
    
    # Initialize camera
    state.camera = cv2.VideoCapture(4)
    state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if state.camera.isOpened():
        print("✓ Camera initialized")
    else:
        print("⚠ Camera failed to open")
    
    # Initialize Pixhawk IMU - TRY HARD TO CONNECT
    print("\nSearching for Pixhawk IMU...")
    possible_ports = [
        '/dev/ttyACM0', '/dev/ttyACM1', 
        '/dev/ttyUSB0', '/dev/ttyUSB1', 
        '/dev/ttyTHS1', '/dev/ttyAMA0',
        'udp:127.0.0.1:14550'
    ]
    
    for port in possible_ports:
        if '/dev/' in port and not os.path.exists(port):
            continue
        
        print(f"  Trying {port}...", end=" ")
        try:
            imu = PixhawkIMU(connection_string=port)
            if imu.connect():
                imu.start()
                state.imu = imu
                print("✓ CONNECTED!")
                break
            else:
                print("failed")
        except Exception as e:
            print(f"error: {e}")
            
    if state.imu is None:
        print("⚠ No Pixhawk found. IMU features disabled.")
        print("  Tip: sudo usermod -a -G dialout $USER")
    
    print("="*60)
    print("System initialized!")
    print("="*60)
    return True


def telemetry_thread():
    """
    PUB/SUB Pattern: Telemetry Publisher
    Publishes real-time IMU/position data to 'telemetry' channel at 150Hz
    """
    print(">>> [PUB] Telemetry publisher started @ 150Hz")
    state.telemetry_running = True
    
    # Initialize Kalman filters for stability
    attitude_kf = AttitudeKalmanFilter()
    velocity_kf = VelocityKalmanFilter()
    position_kf = PositionKalmanFilter()
    
    # Position accumulator with drift compensation
    px, py, pz = 0.0, 0.0, 0.0
    last_t = time.perf_counter()
    frame_count = 0
    last_debug = time.time()
    
    # Optimized decay constants
    VELOCITY_DECAY = 0.96
    POSITION_DECAY = 0.99
    
    # Simulate IMU data if no Pixhawk (for testing)
    sim_time = 0.0
    
    while state.telemetry_running:
        loop_start = time.perf_counter()
        dt = loop_start - last_t
        last_t = loop_start
        sim_time += dt
        
        # Try to get real IMU data
        imu = state.imu
        if imu is not None and imu.connected and (time.time() - imu.last_update) < 1.0:
            
            # Direct attribute access - fastest possible
            roll_raw = imu.roll
            pitch_raw = imu.pitch
            yaw_raw = imu.yaw
            vx_raw = imu.vel_x
            vy_raw = imu.vel_y
            vz_raw = imu.vel_z
            
            # Minimal Kalman filtering
            roll, pitch, yaw = attitude_kf.update(roll_raw, pitch_raw, yaw_raw)
            vx, vy, vz = velocity_kf.update(vx_raw, vy_raw, vz_raw)
            
            # Fast decay
            vx *= VELOCITY_DECAY
            vy *= VELOCITY_DECAY
            vz *= VELOCITY_DECAY
            
            # Fast integration with threshold
            if abs(vx) > 0.01:
                px += vx * dt * 0.05
            if abs(vy) > 0.01:
                py += vy * dt * 0.05
            if abs(vz) > 0.01:
                pz += vz * dt * 0.05
            
            # Fast decay and clamp
            px = max(-30, min(30, px * POSITION_DECAY))
            py = max(-30, min(30, py * POSITION_DECAY))
            pz = max(-30, min(30, pz * POSITION_DECAY))
            
            # Minimal position filtering
            px_f, py_f, pz_f = position_kf.update(px, py, pz)
            
        else:
            # SIMULATED DATA for testing (smooth movement)
            roll = 0.1 * np.sin(sim_time * 0.5)
            pitch = 0.08 * np.sin(sim_time * 0.7)
            yaw = sim_time * 0.2
            
            # Circular movement pattern
            px_f = 5.0 * np.sin(sim_time * 0.3)
            py_f = 5.0 * np.cos(sim_time * 0.3)
            pz_f = 2.0 * np.sin(sim_time * 0.2)
            
            if state.imu is None:
                if frame_count % 300 == 0:  # Every 2 seconds at 150Hz
                    print(f"[PUB] Using SIMULATED data (no Pixhawk)")
        
        # PUBLISH to 'telemetry' channel (PUB/SUB pattern)
        socketio.emit('telemetry', [roll, pitch, yaw, px_f, py_f, pz_f])
        frame_count += 1
        
        # Debug output every 2 seconds
        if time.time() - last_debug > 2.0:
            hz = frame_count / 2.0
            mode = "REAL" if (state.imu and state.imu.connected) else "SIM"
            print(f"[PUB/TELEM] {mode} | R:{roll:.2f} P:{pitch:.2f} Y:{yaw:.2f} | {hz:.0f}Hz")
            frame_count = 0
            last_debug = time.time()
        
        # Adaptive sleep - target 150Hz
        elapsed = time.perf_counter() - loop_start
        target_interval = 0.00667  # 150Hz
        sleep_time = max(0.0001, target_interval - elapsed)
        time.sleep(sleep_time)
    
    print(">>> [PUB] Telemetry stopped")


def process_frame():
    """Process single frame and return results"""
    if state.camera is None or not state.camera.isOpened():
        return None
    
    ret, frame = state.camera.read()
    if not ret:
        return None
    
    start_time = time.time()
    
    # Process only if reference exists
    if state.reference_frame is not None:
        # XFeat matching
        device = "cuda" if torch.cuda.is_available() else "cpu"
        current_tensor = process_frame_for_xfeat(frame, device)
        
        with torch.no_grad():
            mkpts_0, mkpts_1 = state.xfeat.match_xfeat(
                state.reference_tensor, current_tensor, top_k=4096
            )
        
        if isinstance(mkpts_0, torch.Tensor):
            mkpts_0 = mkpts_0.cpu().numpy()
        if isinstance(mkpts_1, torch.Tensor):
            mkpts_1 = mkpts_1.cpu().numpy()
        
        num_matches = len(mkpts_0)
        match_ratio = (num_matches / state.max_matches * 100) if state.max_matches > 0 else 0
        
        # XFeat guidance
        dx, dy, dz_scale, confidence, found = estimate_camera_guidance(mkpts_0, mkpts_1)
        guidance_source = "features" if found and confidence > 30 else "none"
        
        # Draw matches on frame (Green dots)
        for pt in mkpts_1:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        
        # YOLO detection (if available)
        detections = []
        if state.yolo is not None:
            try:
                yolo_results = state.yolo(frame, verbose=False)
                for result in yolo_results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = result.names[cls]
                        
                        detections.append({
                            'class': label,
                            'confidence': conf,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })
                        
                        # Draw YOLO box (Pink/Magenta)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            except Exception as e:
                pass  # Silently continue if YOLO fails
        
        # Update state
        with state.lock:
            state.current_data.update({
                'fps': 1.0 / max(0.001, time.time() - start_time),
                'num_matches': int(num_matches),
                'max_matches': int(state.max_matches),
                'match_ratio': float(match_ratio),
                'confidence': float(confidence),
                'dx': float(dx),
                'dy': float(dy),
                'dz_scale': float(dz_scale),
                'guidance_source': guidance_source,
                'yolo_detections': detections,
                'status': 'tracking'
            })
    
    # Encode frame for web AFTER drawing
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'frame': frame_b64,
        'data': state.current_data
    }


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """Get current system status"""
    with state.lock:
        return jsonify(state.current_data)


@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print('Client connected')
    emit('status', {'message': 'Connected to Visual Nav System'})


@socketio.on('capture_reference')
def handle_capture_reference():
    """Capture reference frame"""
    if state.camera is None:
        emit('error', {'message': 'Camera not initialized'})
        return
    
    ret, frame = state.camera.read()
    if not ret:
        emit('error', {'message': 'Failed to capture frame'})
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    state.reference_frame = frame.copy()
    state.reference_tensor = process_frame_for_xfeat(frame, device)
    state.reference_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    with torch.no_grad():
        ref_output = state.xfeat.detectAndCompute(state.reference_tensor, top_k=4096)[0]
        state.max_matches = len(ref_output["keypoints"])
    
    # Extract objects if YOLO is available
    if state.yolo is not None:
        try:
            ref_results = state.yolo(frame, verbose=False)
            state.reference_objects = extract_object_info(ref_results)
        except:
            state.reference_objects = []
    else:
        state.reference_objects = []
    
    # Encode reference frame
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    ref_b64 = base64.b64encode(buffer).decode('utf-8')
    
    emit('reference_captured', {
        'frame': ref_b64,
        'num_features': int(state.max_matches),
        'num_objects': len(state.reference_objects)
    })
    print(f"Reference captured: {state.max_matches} features")


@socketio.on('reset_reference')
def handle_reset():
    """Reset reference frame"""
    state.reference_frame = None
    state.reference_tensor = None
    state.reference_gray = None
    state.reference_objects = []
    state.max_matches = 0
    emit('reference_reset', {})
    print("Reference reset")


@socketio.on('reset_position')
def handle_reset_position():
    """Reset dead reckoning position"""
    if state.imu is not None:
        state.imu.reset_velocity()
    emit('position_reset', {})
    print("Position reset")


def system_health_thread():
    """
    PUB/SUB Pattern: System Health Publisher
    Publishes system metrics to 'health' channel at 2Hz
    """
    print(">>> [PUB] System health publisher started @ 2Hz")
    
    while state.running:
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            mem = psutil.virtual_memory()
            mem_percent = mem.percent
            mem_used_gb = mem.used / (1024**3)
            mem_total_gb = mem.total / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Temperature (if available)
            temp = None
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    temp = temps['coretemp'][0].current
                elif 'cpu_thermal' in temps:  # Raspberry Pi
                    temp = temps['cpu_thermal'][0].current
            except:
                pass
            
            # Network stats
            net_io = psutil.net_io_counters()
            bytes_sent = net_io.bytes_sent / (1024**2)  # MB
            bytes_recv = net_io.bytes_recv / (1024**2)  # MB
            
            # System uptime
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            uptime_hours = uptime_seconds / 3600
            
            # Build health packet
            health_data = {
                'cpu': {
                    'percent': round(cpu_percent, 1),
                    'cores': cpu_count,
                    'status': 'normal' if cpu_percent < 70 else 'warning' if cpu_percent < 90 else 'critical'
                },
                'memory': {
                    'percent': round(mem_percent, 1),
                    'used_gb': round(mem_used_gb, 1),
                    'total_gb': round(mem_total_gb, 1),
                    'status': 'normal' if mem_percent < 70 else 'warning' if mem_percent < 90 else 'critical'
                },
                'disk': {
                    'percent': round(disk_percent, 1),
                    'used_gb': round(disk_used_gb, 1),
                    'total_gb': round(disk_total_gb, 1),
                    'status': 'normal' if disk_percent < 70 else 'warning' if disk_percent < 90 else 'critical'
                },
                'temperature': {
                    'celsius': round(temp, 1) if temp else None,
                    'status': 'normal' if not temp or temp < 70 else 'warning' if temp < 80 else 'critical'
                },
                'network': {
                    'sent_mb': round(bytes_sent, 1),
                    'recv_mb': round(bytes_recv, 1)
                },
                'uptime_hours': round(uptime_hours, 1),
                'timestamp': time.time()
            }
            
            # PUBLISH to 'health' channel (PUB/SUB pattern)
            socketio.emit('health', health_data)
            
        except Exception as e:
            print(f"[HEALTH] Error: {e}")
        
        time.sleep(0.5)  # 2Hz update rate
    
    print(">>> [PUB] System health stopped")


def video_stream():
    """Generator for video streaming"""
    while True:
        result = process_frame()
        if result:
            # Emit frame data for vision stats
            socketio.emit('frame_data', result['data'])
            
            # Yield frame for HTTP streaming
            frame_bytes = base64.b64decode(result['frame'])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(video_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    initialize_system()
    
    # START ALL BACKGROUND THREADS
    telemetry_t = Thread(target=telemetry_thread, daemon=True)
    telemetry_t.start()
    
    health_t = Thread(target=system_health_thread, daemon=True)
    health_t.start()
    
    print("\n" + "="*60)
    print("🤖 BRACU DUBURI - Visual Navigation System")
    print("="*60)
    print("✓ Telemetry thread: REAL-TIME @ 150Hz")
    print("✓ System health monitor: ACTIVE @ 2Hz")
    print("✓ Video processing: ENABLED")
    print("\n🌐 Server: http://localhost:5000")
    print("📊 System Dashboard: READY")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

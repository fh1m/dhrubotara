"""
XFeat + YOLOv11 + Optical Flow + Pixhawk IMU Sensor Fusion
Complete hybrid navigation system with attitude and motion data from Pixhawk 2.4.8

Guidance sources (prioritized and fused):
1. Pixhawk IMU/Gyro (attitude, angular velocity, acceleration)
2. XFeat homography (visual position estimation)
3. Object-based tracking (semantic fallback)
4. Optical Flow (motion continuity)

Sensor Fusion: Complementary filter combining visual and inertial data
"""

import cv2
import torch
import numpy as np
import time
import sys
from pymavlink import mavutil
from threading import Thread, Lock
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, "accelerated_features")
from modules.xfeat import XFeat
from ultralytics import YOLO


class PixhawkIMU:
    """Thread-safe Pixhawk IMU data reader using pymavlink"""
    
    def __init__(self, connection_string='udp:127.0.0.1:14550'):
        """
        Initialize connection to Pixhawk
        connection_string options:
        - Serial: '/dev/ttyACM0' or 'COM3' (adjust based on your system)
        - UDP: 'udp:127.0.0.1:14550' (for SITL or MAVProxy forwarding)
        - TCP: 'tcp:127.0.0.1:5760'
        """
        self.connection_string = connection_string
        self.master = None
        self.lock = Lock()
        
        # IMU data (raw)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.rollspeed = 0.0
        self.pitchspeed = 0.0
        self.yawspeed = 0.0
        
        # Acceleration (m/s^2) - raw
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        
        # EKF2 data (more stable, filtered estimates)
        self.ekf_pos_x = 0.0
        self.ekf_pos_y = 0.0
        self.ekf_pos_z = 0.0
        self.ekf_vel_x = 0.0
        self.ekf_vel_y = 0.0
        self.ekf_vel_z = 0.0
        
        # EKF status
        self.ekf_flags = 0
        self.ekf_healthy = False
        
        # Linear velocity estimation (integrated from accel) - for comparison
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        
        # Status
        self.connected = False
        self.running = False
        self.last_update = time.time()
        self.last_ekf_update = time.time()
        
    def connect(self):
        """Connect to Pixhawk"""
        try:
            print(f"Connecting to Pixhawk at {self.connection_string}...")
            self.master = mavutil.mavlink_connection(self.connection_string, baud=57600)
            
            # Wait for heartbeat
            print("Waiting for heartbeat...")
            self.master.wait_heartbeat(timeout=5)
            print(f"✓ Heartbeat from system {self.master.target_system}")
            
            # Request data streams
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,  # 10 Hz
                1    # Start
            )
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to Pixhawk: {e}")
            print("  Make sure Pixhawk is connected and MAVLink is available")
            print("  For USB: Try /dev/ttyACM0 or COM port")
            print("  For simulation: Use udp:127.0.0.1:14550")
            return False
    
    def start(self):
        """Start IMU data reading thread"""
        if not self.connected:
            print("Cannot start - not connected to Pixhawk")
            return False
        
        self.running = True
        self.thread = Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print("✓ IMU reading thread started")
        return True
    
    def _update_loop(self):
        """Continuous update loop for IMU data"""
        last_time = time.time()
        
        while self.running:
            try:
                # Non-blocking receive
                msg = self.master.recv_match(blocking=False)
                
                if msg is not None:
                    msg_type = msg.get_type()
                    current_time = time.time()
                    dt = current_time - last_time
                    
                    with self.lock:
                        # ATTITUDE message (roll, pitch, yaw and rates)
                        if msg_type == 'ATTITUDE':
                            self.roll = msg.roll  # radians
                            self.pitch = msg.pitch  # radians
                            self.yaw = msg.yaw  # radians
                            self.rollspeed = msg.rollspeed  # rad/s
                            self.pitchspeed = msg.pitchspeed  # rad/s
                            self.yawspeed = msg.yawspeed  # rad/s
                            self.last_update = current_time
                        
                        # RAW_IMU message (acceleration)
                        elif msg_type == 'RAW_IMU':
                            # Convert from milli-Gs to m/s^2
                            self.accel_x = msg.xacc / 1000.0 * 9.81
                            self.accel_y = msg.yacc / 1000.0 * 9.81
                            self.accel_z = msg.zacc / 1000.0 * 9.81
                            
                            # Simple integration for velocity estimation (not perfect, drifts)
                            # In real application, fuse with visual odometry to correct drift
                            if dt < 0.5:  # Sanity check
                                self.vel_x += self.accel_x * dt
                                self.vel_y += self.accel_y * dt
                                self.vel_z += self.accel_z * dt
                        
                        # LOCAL_POSITION_NED message (EKF filtered position and velocity)
                        elif msg_type == 'LOCAL_POSITION_NED':
                            self.ekf_pos_x = msg.x  # meters (North)
                            self.ekf_pos_y = msg.y  # meters (East)
                            self.ekf_pos_z = msg.z  # meters (Down)
                            self.ekf_vel_x = msg.vx  # m/s (North)
                            self.ekf_vel_y = msg.vy  # m/s (East)
                            self.ekf_vel_z = msg.vz  # m/s (Down)
                            self.last_ekf_update = current_time
                        
                        # EKF_STATUS_REPORT message (EKF health)
                        elif msg_type == 'EKF_STATUS_REPORT':
                            self.ekf_flags = msg.flags
                            # Check if EKF is healthy (bit flags)
                            self.ekf_healthy = (msg.flags & 0x001) != 0  # EKF_ATTITUDE flag
                    
                    last_time = current_time
                
                time.sleep(0.001)  # Small sleep to prevent CPU spinning
                
            except Exception as e:
                print(f"Error reading IMU data: {e}")
                time.sleep(0.1)
    
    def get_attitude(self):
        """Get current attitude (thread-safe)"""
        with self.lock:
            return {
                'roll': self.roll,
                'pitch': self.pitch,
                'yaw': self.yaw,
                'rollspeed': self.rollspeed,
                'pitchspeed': self.pitchspeed,
                'yawspeed': self.yawspeed,
                'connected': self.connected,
                'age': time.time() - self.last_update
            }
    
    def get_acceleration(self):
        """Get current acceleration (thread-safe)"""
        with self.lock:
            return {
                'x': self.accel_x,
                'y': self.accel_y,
                'z': self.accel_z,
                'vel_x': self.vel_x,
                'vel_y': self.vel_y,
                'vel_z': self.vel_z
            }
    
    def get_ekf_data(self):
        """Get EKF filtered data (thread-safe)"""
        with self.lock:
            return {
                'pos_x': self.ekf_pos_x,
                'pos_y': self.ekf_pos_y,
                'pos_z': self.ekf_pos_z,
                'vel_x': self.ekf_vel_x,
                'vel_y': self.ekf_vel_y,
                'vel_z': self.ekf_vel_z,
                'healthy': self.ekf_healthy,
                'flags': self.ekf_flags,
                'age': time.time() - self.last_ekf_update
            }
    
    def reset_velocity(self):
        """Reset integrated velocity (useful when AUV stops)"""
        with self.lock:
            self.vel_x = 0.0
            self.vel_y = 0.0
            self.vel_z = 0.0
    
    def stop(self):
        """Stop IMU reading"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        if self.master:
            self.master.close()
        print("IMU reader stopped")


def estimate_camera_guidance(mkpts_0, mkpts_1, min_matches=8):
    """
    Estimate camera movement guidance based on feature matches
    Returns: (dx, dy, dz_scale, confidence, homography_found)
    """
    if len(mkpts_0) < min_matches:
        return 0, 0, 1.0, 0.0, False

    try:
        H, mask = cv2.findHomography(mkpts_1, mkpts_0, cv2.RANSAC, 5.0)

        if H is None:
            return 0, 0, 1.0, 0.0, False

        tx = H[0, 2]
        ty = H[1, 2]

        sx = np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
        sy = np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)
        scale = (sx + sy) / 2.0

        confidence = np.sum(mask) / len(mask) * 100 if mask is not None else 0

        return tx, ty, scale, confidence, True

    except Exception as e:
        return 0, 0, 1.0, 0.0, False


def estimate_guidance_from_objects(ref_objects, curr_objects):
    """
    Estimate camera guidance based on detected object positions
    Returns: (dx, dy, dz_scale, confidence, found)
    """
    if len(ref_objects) == 0 or len(curr_objects) == 0:
        return 0, 0, 1.0, 0.0, False

    matched_pairs = []
    for ref_obj in ref_objects:
        ref_class = ref_obj["class"]
        ref_center = ref_obj["center"]
        ref_area = ref_obj["area"]

        for curr_obj in curr_objects:
            if curr_obj["class"] == ref_class:
                matched_pairs.append(
                    {
                        "ref_center": ref_center,
                        "curr_center": curr_obj["center"],
                        "ref_area": ref_area,
                        "curr_area": curr_obj["area"],
                    }
                )
                break

    if len(matched_pairs) == 0:
        return 0, 0, 1.0, 0.0, False

    dx_total, dy_total, scale_total = 0, 0, 0
    for pair in matched_pairs:
        dx_total += pair["ref_center"][0] - pair["curr_center"][0]
        dy_total += pair["ref_center"][1] - pair["curr_center"][1]
        scale_total += np.sqrt(pair["ref_area"] / (pair["curr_area"] + 1e-6))

    n = len(matched_pairs)
    dx = dx_total / n
    dy = dy_total / n
    scale = scale_total / n

    confidence = min(100, n * 30)

    return dx, dy, scale, confidence, True


def estimate_guidance_from_optical_flow(ref_gray, curr_gray, prev_points=None):
    """
    Estimate camera guidance using Lucas-Kanade optical flow
    Returns: (dx, dy, dz_scale, confidence, found, tracked_points)
    """
    if ref_gray is None or curr_gray is None:
        return 0, 0, 1.0, 0.0, False, None

    if prev_points is None or len(prev_points) < 10:
        feature_params = dict(
            maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7
        )
        prev_points = cv2.goodFeaturesToTrack(ref_gray, mask=None, **feature_params)

        if prev_points is None:
            return 0, 0, 1.0, 0.0, False, None

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    try:
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            ref_gray, curr_gray, prev_points, None, **lk_params
        )

        if next_points is None:
            return 0, 0, 1.0, 0.0, False, None

        good_prev = prev_points[status == 1]
        good_next = next_points[status == 1]

        if len(good_prev) < 5:
            return 0, 0, 1.0, 0.0, False, None

        displacement = good_next - good_prev
        dx = np.median(displacement[:, 0])
        dy = np.median(displacement[:, 1])

        flow_magnitude = np.linalg.norm(displacement, axis=1)
        avg_flow = np.median(flow_magnitude)

        if avg_flow > 2.0:
            scale = 1.0 + (avg_flow - 2.0) / 50.0
        else:
            scale = 1.0

        flow_std = np.std(flow_magnitude)
        consistency = 1.0 / (1.0 + flow_std / 10.0)
        confidence = min(100, (len(good_prev) / 200.0) * 100 * consistency)

        return -dx, -dy, scale, confidence, True, good_next.reshape(-1, 1, 2)

    except Exception as e:
        return 0, 0, 1.0, 0.0, False, None


def fuse_visual_imu_guidance(visual_dx, visual_dy, visual_dz, visual_conf, 
                              imu_data, alpha=0.7):
    """
    Fuse visual guidance with EKF-filtered IMU data using complementary filter
    
    alpha: weight for visual data (0.7 = 70% visual, 30% IMU/EKF)
    Returns: (fused_dx, fused_dy, fused_dz, fused_confidence, fusion_info)
    """
    attitude = imu_data.get_attitude()
    ekf = imu_data.get_ekf_data()
    
    # Check if data is fresh and EKF is healthy
    imu_fresh = attitude['connected'] and attitude['age'] < 1.0
    ekf_fresh = ekf['age'] < 1.0 and ekf['healthy']
    
    fusion_info = {
        'imu_used': imu_fresh,
        'ekf_used': ekf_fresh,
        'correction_applied': False,
        'ekf_vel_mag': 0.0
    }
    
    if not imu_fresh:
        # No IMU data, use visual only
        return visual_dx, visual_dy, visual_dz, visual_conf, fusion_info
    
    # IMU corrections based on attitude
    # If AUV is tilted, visual estimates need compensation
    roll = attitude['roll']
    pitch = attitude['pitch']
    
    # Rotation compensation (simplified - full implementation needs rotation matrices)
    # This is a basic correction for small angles
    dx_corrected = visual_dx * np.cos(roll) - visual_dy * np.sin(roll)
    dy_corrected = visual_dx * np.sin(roll) + visual_dy * np.cos(roll)
    
    fusion_info['correction_applied'] = True
    
    # Use EKF velocity if available (more stable than integrated acceleration)
    if ekf_fresh:
        ekf_vel_magnitude = np.sqrt(ekf['vel_x']**2 + ekf['vel_y']**2)
        fusion_info['ekf_vel_mag'] = ekf_vel_magnitude
        
        # Blend visual with EKF velocity direction
        # Higher visual confidence = trust visual more
        visual_weight = alpha * (visual_conf / 100.0)
        ekf_weight = 1.0 - visual_weight
        
        # Use EKF velocity to scale and validate visual estimates
        # Convert EKF velocity (NED frame) to camera frame approximation
        # This is simplified - proper transformation needs camera calibration
        ekf_dx = -ekf['vel_y'] * 100  # Rough pixel scaling
        ekf_dy = ekf['vel_z'] * 100
        
        # Complementary filter
        fused_dx = visual_weight * dx_corrected + ekf_weight * ekf_dx
        fused_dy = visual_weight * dy_corrected + ekf_weight * ekf_dy
        fused_dz = visual_dz  # Keep visual depth estimate
        
        # Boost confidence if EKF and visual agree on motion direction
        dx_agree = (np.sign(dx_corrected) == np.sign(ekf_dx)) if abs(ekf_dx) > 5 else True
        dy_agree = (np.sign(dy_corrected) == np.sign(ekf_dy)) if abs(ekf_dy) > 5 else True
        
        if dx_agree and dy_agree and ekf_vel_magnitude > 0.1:
            fused_confidence = min(100, visual_conf * 1.3)
        else:
            fused_confidence = visual_conf
    else:
        # No EKF, use attitude-corrected visual only
        fused_dx = dx_corrected
        fused_dy = dy_corrected
        fused_dz = visual_dz
        fused_confidence = visual_conf
    
    return fused_dx, fused_dy, fused_dz, fused_confidence, fusion_info


def extract_object_info(results):
    """Extract object information from YOLO results"""
    objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            objects.append(
                {
                    "class": result.names[cls],
                    "center": (center_x, center_y),
                    "area": area,
                    "confidence": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                }
            )

    return objects


def draw_imu_raw_panel(imu_data, panel_h=200):
    """
    Create raw IMU data visualization panel (attitude and rates)
    """
    panel_w = 350
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = [45, 35, 30]
    
    attitude = imu_data.get_attitude()
    
    # Title
    cv2.putText(panel, "RAW IMU DATA", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 100), 2)
    
    y_pos = 55
    
    if attitude['connected']:
        # Connection status
        cv2.circle(panel, (panel_w - 30, 20), 8, (100, 255, 100), -1)
        cv2.circle(panel, (panel_w - 30, 20), 10, (150, 255, 150), 2)
        
        # Section: Attitude
        cv2.putText(panel, "ATTITUDE:", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 100), 2)
        y_pos += 25
        
        cv2.putText(panel, f"Roll:  {np.degrees(attitude['roll']):7.2f}°", 
                    (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1)
        y_pos += 22
        cv2.putText(panel, f"Pitch: {np.degrees(attitude['pitch']):7.2f}°", 
                    (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1)
        y_pos += 22
        cv2.putText(panel, f"Yaw:   {np.degrees(attitude['yaw']):7.2f}°", 
                    (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1)
        y_pos += 30
        
        # Section: Angular Rates
        cv2.putText(panel, "ANGULAR RATES:", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 100), 2)
        y_pos += 25
        
        cv2.putText(panel, f"Roll:  {np.degrees(attitude['rollspeed']):6.2f}°/s", 
                    (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 150), 1)
        y_pos += 22
        cv2.putText(panel, f"Pitch: {np.degrees(attitude['pitchspeed']):6.2f}°/s", 
                    (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 150), 1)
        y_pos += 22
        cv2.putText(panel, f"Yaw:   {np.degrees(attitude['yawspeed']):6.2f}°/s", 
                    (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 150), 1)
        
        # Data age warning
        if attitude['age'] > 0.5:
            cv2.putText(panel, "STALE", (panel_w - 60, panel_h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1)
    else:
        # Not connected
        cv2.circle(panel, (panel_w - 30, 20), 8, (100, 100, 100), -1)
        cv2.circle(panel, (panel_w - 30, 20), 10, (150, 150, 150), 2)
        cv2.putText(panel, "NOT CONNECTED", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 150), 1)
        cv2.putText(panel, "Check Pixhawk", (10, y_pos + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 120), 1)
    
    return panel


def draw_ekf_panel(imu_data, panel_h=200):
    """
    Create EKF filtered data visualization panel
    """
    panel_w = 350
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = [35, 45, 40]
    
    ekf = imu_data.get_ekf_data()
    
    # Title
    cv2.putText(panel, "EKF FILTERED", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 200), 2)
    
    y_pos = 55
    
    if ekf['age'] < 2.0:
        # EKF status
        status_color = (100, 255, 100) if ekf['healthy'] else (100, 100, 255)
        cv2.circle(panel, (panel_w - 30, 20), 8, status_color, -1)
        cv2.circle(panel, (panel_w - 30, 20), 10, (150, 255, 150) if ekf['healthy'] else (150, 150, 255), 2)
        
        # Section: Position (NED frame)
        cv2.putText(panel, "POSITION (NED):", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 100), 2)
        y_pos += 25
        
        cv2.putText(panel, f"North: {ekf['pos_x']:7.2f} m", 
                    (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 200), 1)
        y_pos += 22
        cv2.putText(panel, f"East:  {ekf['pos_y']:7.2f} m", 
                    (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 200), 1)
        y_pos += 22
        cv2.putText(panel, f"Down:  {ekf['pos_z']:7.2f} m", 
                    (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 200), 1)
        y_pos += 30
        
        # Section: Velocity (NED frame)
        cv2.putText(panel, "VELOCITY (NED):", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 100), 2)
        y_pos += 25
        
        cv2.putText(panel, f"North: {ekf['vel_x']:6.2f} m/s", 
                    (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)
        y_pos += 22
        cv2.putText(panel, f"East:  {ekf['vel_y']:6.2f} m/s", 
                    (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)
        y_pos += 22
        cv2.putText(panel, f"Down:  {ekf['vel_z']:6.2f} m/s", 
                    (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)
        
        # Velocity magnitude
        vel_mag = np.sqrt(ekf['vel_x']**2 + ekf['vel_y']**2 + ekf['vel_z']**2)
        cv2.putText(panel, f"Mag: {vel_mag:.2f} m/s", 
                    (panel_w - 120, panel_h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 150), 1)
        
        # Data age warning
        if ekf['age'] > 0.5:
            cv2.putText(panel, "STALE", (panel_w - 180, panel_h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1)
    else:
        # No EKF data
        cv2.circle(panel, (panel_w - 30, 20), 8, (100, 100, 100), -1)
        cv2.circle(panel, (panel_w - 30, 20), 10, (150, 150, 150), 2)
        cv2.putText(panel, "NO EKF DATA", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 150), 1)
        cv2.putText(panel, "Waiting...", (10, y_pos + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 120), 1)
    
    return panel


def draw_fusion_panel(visual_dx, visual_dy, visual_dz, visual_conf,
                       fused_dx, fused_dy, fused_dz, fused_conf, 
                       fusion_info, panel_h=200):
    """
    Create sensor fusion result panel showing before/after
    """
    panel_w = 350
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = [40, 40, 50]
    
    # Title
    cv2.putText(panel, "SENSOR FUSION", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 255), 2)
    
    y_pos = 55
    
    # Status indicators
    status_x = panel_w - 100
    if fusion_info['imu_used']:
        cv2.circle(panel, (status_x, 20), 6, (100, 255, 100), -1)
        cv2.putText(panel, "IMU", (status_x + 10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 255, 150), 1)
    
    if fusion_info['ekf_used']:
        cv2.circle(panel, (status_x + 45, 20), 6, (100, 255, 255), -1)
        cv2.putText(panel, "EKF", (status_x + 55, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 255, 255), 1)
    
    # Section: Visual Input
    cv2.putText(panel, "VISUAL INPUT:", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 100), 2)
    y_pos += 25
    
    cv2.putText(panel, f"DX: {visual_dx:7.1f} px", 
                (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_pos += 22
    cv2.putText(panel, f"DY: {visual_dy:7.1f} px", 
                (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_pos += 22
    cv2.putText(panel, f"DZ: {visual_dz:7.3f}x", 
                (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(panel, f"Conf: {visual_conf:.0f}%", 
                (panel_w - 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 200), 1)
    y_pos += 30
    
    # Section: Fused Output
    cv2.putText(panel, "FUSED OUTPUT:", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 2)
    y_pos += 25
    
    cv2.putText(panel, f"DX: {fused_dx:7.1f} px", 
                (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
    # Show delta
    delta_dx = fused_dx - visual_dx
    if abs(delta_dx) > 0.5:
        cv2.putText(panel, f"({delta_dx:+.1f})", 
                    (panel_w - 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 100), 1)
    y_pos += 22
    
    cv2.putText(panel, f"DY: {fused_dy:7.1f} px", 
                (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
    # Show delta
    delta_dy = fused_dy - visual_dy
    if abs(delta_dy) > 0.5:
        cv2.putText(panel, f"({delta_dy:+.1f})", 
                    (panel_w - 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 100), 1)
    y_pos += 22
    
    cv2.putText(panel, f"DZ: {fused_dz:7.3f}x", 
                (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
    cv2.putText(panel, f"Conf: {fused_conf:.0f}%", 
                (panel_w - 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 150), 1)
    
    # EKF velocity magnitude if available
    if fusion_info['ekf_used']:
        cv2.putText(panel, f"EKF Vel: {fusion_info['ekf_vel_mag']:.2f} m/s", 
                    (10, panel_h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 255), 1)
    
    # Correction status
    if fusion_info['correction_applied']:
        cv2.putText(panel, "CORRECTED", (panel_w - 100, panel_h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1)
    
    return panel


def draw_artificial_horizon(roll, pitch, yaw, size=(300, 300)):
    """
    Aviation-style attitude indicator showing roll, pitch, and yaw
    """
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    center = (size[0]//2, size[1]//2)
    
    # Calculate pitch offset (scale for visibility)
    pitch_offset = int(np.degrees(pitch) * 2.5)
    horizon_y = center[1] + pitch_offset
    
    # Create large canvas for rotation (prevent clipping)
    canvas_size = int(size[0] * 1.5)
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas_center = (canvas_size//2, canvas_size//2)
    
    # Sky (blue gradient) and ground (brown gradient)
    sky_color_top = (200, 120, 60)
    sky_color_bottom = (180, 100, 50)
    ground_color_top = (50, 80, 120)
    ground_color_bottom = (30, 60, 100)
    
    # Draw sky with gradient
    for y in range(0, horizon_y + canvas_center[1] - center[1]):
        factor = y / (horizon_y + canvas_center[1] - center[1]) if horizon_y > 0 else 0
        color = tuple(int(sky_color_top[i] + (sky_color_bottom[i] - sky_color_top[i]) * factor) 
                     for i in range(3))
        canvas[y, :] = color
    
    # Draw ground with gradient
    for y in range(horizon_y + canvas_center[1] - center[1], canvas_size):
        factor = (y - (horizon_y + canvas_center[1] - center[1])) / (canvas_size - (horizon_y + canvas_center[1] - center[1]))
        color = tuple(int(ground_color_top[i] + (ground_color_bottom[i] - ground_color_top[i]) * factor) 
                     for i in range(3))
        canvas[y, :] = color
    
    # Draw horizon line (white, thick)
    cv2.line(canvas, (0, horizon_y + canvas_center[1] - center[1]), 
             (canvas_size, horizon_y + canvas_center[1] - center[1]), 
             (255, 255, 255), 3)
    
    # Draw pitch ladder
    for pitch_deg in range(-30, 40, 10):
        if pitch_deg == 0:
            continue
        y_pos = horizon_y + canvas_center[1] - center[1] + int(pitch_deg * 2.5)
        if 0 < y_pos < canvas_size:
            line_length = 60 if pitch_deg % 20 == 0 else 40
            cv2.line(canvas, (canvas_center[0] - line_length, y_pos), 
                    (canvas_center[0] + line_length, y_pos), (255, 255, 255), 2)
            
            # Add degree markers
            text = f"{abs(pitch_deg)}°"
            cv2.putText(canvas, text, (canvas_center[0] - line_length - 35, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(canvas, text, (canvas_center[0] + line_length + 10, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Rotate entire canvas for roll
    M = cv2.getRotationMatrix2D(canvas_center, np.degrees(roll), 1.0)
    canvas = cv2.warpAffine(canvas, M, (canvas_size, canvas_size))
    
    # Crop back to original size
    offset = (canvas_size - size[0]) // 2
    img = canvas[offset:offset+size[1], offset:offset+size[0]]
    
    # Draw fixed aircraft symbol (orange, centered)
    wing_w = 70
    wing_y_offset = 5
    # Wings
    cv2.line(img, (center[0]-wing_w, center[1] + wing_y_offset), 
             (center[0]-20, center[1] + wing_y_offset), (0, 165, 255), 5)
    cv2.line(img, (center[0]+20, center[1] + wing_y_offset), 
             (center[0]+wing_w, center[1] + wing_y_offset), (0, 165, 255), 5)
    # Center dot
    cv2.circle(img, center, 10, (0, 165, 255), -1)
    cv2.circle(img, center, 12, (255, 255, 255), 2)
    # Vertical line (nose)
    cv2.line(img, (center[0], center[1]), 
             (center[0], center[1] - 15), (0, 165, 255), 4)
    
    # Draw roll arc at top
    arc_radius = 110
    for angle in range(-60, 70, 10):
        angle_rad = np.radians(angle)
        x = int(center[0] + arc_radius * np.sin(angle_rad))
        y = int(center[1] - arc_radius * np.cos(angle_rad))
        if angle % 30 == 0:
            cv2.circle(img, (x, y), 4, (255, 255, 255), -1)
            # Add angle text
            label = f"{abs(angle)}°"
            cv2.putText(img, label, (x - 12, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        else:
            cv2.circle(img, (x, y), 2, (255, 255, 255), -1)
    
    # Draw roll indicator (triangle at top pointing to current roll)
    roll_angle_rad = roll
    tri_x = int(center[0] + arc_radius * np.sin(roll_angle_rad))
    tri_y = int(center[1] - arc_radius * np.cos(roll_angle_rad))
    tri_size = 8
    triangle = np.array([
        [tri_x, tri_y - tri_size],
        [tri_x - tri_size//2, tri_y + tri_size//2],
        [tri_x + tri_size//2, tri_y + tri_size//2]
    ], np.int32)
    cv2.fillPoly(img, [triangle], (0, 255, 255))
    cv2.polylines(img, [triangle], True, (255, 255, 255), 2)
    
    # Draw circular mask to make it look like instrument
    mask = np.zeros_like(img)
    cv2.circle(mask, center, size[0]//2 - 10, (255, 255, 255), -1)
    img = cv2.bitwise_and(img, mask)
    
    # Draw instrument bezel
    cv2.circle(img, center, size[0]//2 - 10, (200, 200, 200), 3)
    cv2.circle(img, center, size[0]//2 - 5, (100, 100, 100), 1)
    
    # Add text displays
    # Roll
    cv2.rectangle(img, (5, 5), (100, 30), (40, 40, 40), -1)
    cv2.putText(img, f"Roll: {np.degrees(roll):5.1f}°", (10, 22), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Pitch
    cv2.rectangle(img, (5, 35), (100, 60), (40, 40, 40), -1)
    cv2.putText(img, f"Pitch: {np.degrees(pitch):5.1f}°", (10, 52), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Yaw (heading)
    cv2.rectangle(img, (size[0]-110, 5), (size[0]-5, 30), (40, 40, 40), -1)
    heading = (np.degrees(yaw) + 360) % 360
    cv2.putText(img, f"Hdg: {heading:5.1f}°", (size[0]-105, 22), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return img


class PerformanceMonitor:
    """Real-time performance graphs using matplotlib"""
    
    def __init__(self, history_length=100):
        self.fps_history = deque(maxlen=history_length)
        self.confidence_history = deque(maxlen=history_length)
        self.matches_history = deque(maxlen=history_length)
        self.ekf_vel_history = deque(maxlen=history_length)
        
        # Initialize with zeros
        for _ in range(10):
            self.fps_history.append(0)
            self.confidence_history.append(0)
            self.matches_history.append(0)
            self.ekf_vel_history.append(0)
    
    def update(self, fps, confidence, matches, ekf_vel_mag):
        """Update performance metrics"""
        self.fps_history.append(fps)
        self.confidence_history.append(confidence)
        self.matches_history.append(matches)
        self.ekf_vel_history.append(ekf_vel_mag)
    
    def draw_performance_panel(self, size=(400, 300)):
        """Draw performance graphs panel"""
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(size[0]/100, size[1]/100), dpi=100)
        fig.patch.set_facecolor('#1e1e2e')
        
        # Time axis
        x = list(range(len(self.fps_history)))
        
        # 1. FPS Graph
        ax1.plot(x, list(self.fps_history), 'g-', linewidth=2)
        ax1.set_ylabel('FPS', color='white', fontsize=8)
        ax1.set_ylim(0, 60)
        ax1.tick_params(colors='white', labelsize=6)
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.set_facecolor('#2a2a3e')
        ax1.axhline(y=30, color='yellow', linestyle='--', linewidth=1, alpha=0.5)
        for spine in ax1.spines.values():
            spine.set_color('gray')
        
        # 2. Confidence Graph
        ax2.plot(x, list(self.confidence_history), 'b-', linewidth=2)
        ax2.set_ylabel('Confidence %', color='white', fontsize=8)
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors='white', labelsize=6)
        ax2.grid(True, alpha=0.3, color='gray')
        ax2.set_facecolor('#2a2a3e')
        ax2.axhline(y=50, color='yellow', linestyle='--', linewidth=1, alpha=0.5)
        for spine in ax2.spines.values():
            spine.set_color('gray')
        
        # 3. Matches Graph
        ax3.plot(x, list(self.matches_history), 'c-', linewidth=2)
        ax3.set_ylabel('Matches', color='white', fontsize=8)
        ax3.set_xlabel('Time (frames)', color='white', fontsize=8)
        ax3.tick_params(colors='white', labelsize=6)
        ax3.grid(True, alpha=0.3, color='gray')
        ax3.set_facecolor('#2a2a3e')
        for spine in ax3.spines.values():
            spine.set_color('gray')
        
        # 4. EKF Velocity Graph
        ax4.plot(x, list(self.ekf_vel_history), 'm-', linewidth=2)
        ax4.set_ylabel('EKF Vel (m/s)', color='white', fontsize=8)
        ax4.set_xlabel('Time (frames)', color='white', fontsize=8)
        ax4.tick_params(colors='white', labelsize=6)
        ax4.grid(True, alpha=0.3, color='gray')
        ax4.set_facecolor('#2a2a3e')
        for spine in ax4.spines.values():
            spine.set_color('gray')
        
        plt.tight_layout()
        
        # Convert to OpenCV image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        
        # Resize to exact size
        img = cv2.resize(img, size)
        
        return img


def create_slick_3d_guidance(
    dx, dy, dz_scale, confidence, guidance_source, size=(600, 600), threshold=10
):
    """
    Create clean 3D guidance visualization focused on vectors
    """
    viz = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for i in range(size[1]):
        factor = i / size[1]
        viz[i, :] = [
            int(20 + factor * 30),
            int(25 + factor * 20),
            int(40 + factor * 25),
        ]

    center_x, center_y = size[0] // 2, size[1] // 2

    title = "3D GUIDANCE"
    cv2.putText(
        viz,
        title,
        (center_x - 80, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 220, 255),
        2,
    )

    axis_length = 120
    origin = (center_x, center_y)

    # Grid
    grid_color = (70, 75, 90)
    for i in range(-2, 3):
        offset = i * 30
        cv2.line(
            viz,
            (center_x + offset, center_y - 60),
            (center_x + offset, center_y + 60),
            grid_color,
            1,
        )
        cv2.line(
            viz,
            (center_x - 60, center_y + offset),
            (center_x + 60, center_y + offset),
            grid_color,
            1,
        )

    # X-axis (RIGHT) - Red
    x_end = (center_x + axis_length, center_y)
    cv2.line(viz, origin, x_end, (100, 100, 200), 5)
    cv2.arrowedLine(viz, origin, x_end, (80, 80, 255), 4, tipLength=0.15)
    cv2.circle(viz, x_end, 6, (80, 80, 255), -1)
    cv2.circle(viz, x_end, 8, (150, 150, 255), 2)
    cv2.putText(
        viz,
        "X",
        (x_end[0] + 15, x_end[1] + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (80, 80, 255),
        2,
    )

    # Y-axis (DOWN) - Green
    y_end = (center_x, center_y + axis_length)
    cv2.line(viz, origin, y_end, (100, 200, 100), 5)
    cv2.arrowedLine(viz, origin, y_end, (80, 255, 80), 4, tipLength=0.15)
    cv2.circle(viz, y_end, 6, (80, 255, 80), -1)
    cv2.circle(viz, y_end, 8, (150, 255, 150), 2)
    cv2.putText(
        viz,
        "Y",
        (y_end[0] + 15, y_end[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (80, 255, 80),
        2,
    )

    # Z-axis (DEPTH) - Orange
    z_end = (center_x - int(axis_length * 0.7), center_y - int(axis_length * 0.7))
    cv2.line(viz, origin, z_end, (200, 150, 100), 5)
    cv2.arrowedLine(viz, origin, z_end, (255, 180, 80), 4, tipLength=0.15)
    cv2.circle(viz, z_end, 6, (255, 180, 80), -1)
    cv2.circle(viz, z_end, 8, (255, 220, 150), 2)
    cv2.putText(
        viz,
        "Z",
        (z_end[0] - 25, z_end[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 180, 80),
        2,
    )

    # Movement Vectors
    arrow_scale = 0.35
    movement_color = (0, 255, 255)

    if abs(dx) > threshold:
        x_move = int(np.clip(dx * arrow_scale, -140, 140))
        move_end_x = (center_x + x_move, center_y)
        cv2.arrowedLine(viz, origin, move_end_x, (0, 180, 180), 8, tipLength=0.25)
        cv2.arrowedLine(viz, origin, move_end_x, movement_color, 5, tipLength=0.25)
        cv2.circle(viz, move_end_x, 12, (0, 200, 200), -1)
        cv2.circle(viz, move_end_x, 16, movement_color, 3)

    if abs(dy) > threshold:
        y_move = int(np.clip(dy * arrow_scale, -140, 140))
        move_end_y = (center_x, center_y + y_move)
        cv2.arrowedLine(viz, origin, move_end_y, (0, 180, 180), 8, tipLength=0.25)
        cv2.arrowedLine(viz, origin, move_end_y, movement_color, 5, tipLength=0.25)
        cv2.circle(viz, move_end_y, 12, (0, 200, 200), -1)
        cv2.circle(viz, move_end_y, 16, movement_color, 3)

    if abs(dz_scale - 1.0) > 0.1:
        z_move = int(np.clip((dz_scale - 1.0) * 140, -100, 100))
        move_end_z = (center_x - int(z_move * 0.7), center_y - int(z_move * 0.7))
        cv2.arrowedLine(viz, origin, move_end_z, (0, 180, 180), 8, tipLength=0.25)
        cv2.arrowedLine(viz, origin, move_end_z, movement_color, 5, tipLength=0.25)
        cv2.circle(viz, move_end_z, 12, (0, 200, 200), -1)
        cv2.circle(viz, move_end_z, 16, movement_color, 3)

    # Camera origin
    cv2.circle(viz, origin, 14, (100, 100, 100), -1)
    cv2.circle(viz, origin, 10, (255, 255, 255), -1)
    cv2.circle(viz, origin, 12, (200, 200, 200), 2)

    return viz


def draw_optical_flow_vectors(image, flow_points, color=(255, 255, 0)):
    """Draw optical flow tracked points on image"""
    if flow_points is not None and len(flow_points) > 0:
        for pt in flow_points:
            x, y = pt[0]
            cv2.circle(image, (int(x), int(y)), 3, color, -1)
            cv2.circle(image, (int(x), int(y)), 4, (255, 255, 255), 1)
    return image


def draw_unified_window(
    ref_frame,
    curr_frame,
    mkpts_0,
    mkpts_1,
    yolo_results,
    dx,
    dy,
    dz_scale,
    confidence,
    guidance_source,
    yolo_active,
    num_matches,
    max_matches,
    fps,
    stability,
    show_guidance,
    flow_points=None,
    imu_data=None,
    visual_dx=0,
    visual_dy=0,
    visual_dz=1.0,
    visual_conf=0,
    fusion_info=None,
    performance_monitor=None,
):
    """
    Create unified window with camera views, 3D guidance, comprehensive IMU/EKF/Fusion panels,
    artificial horizon, and performance graphs
    """
    h, w = curr_frame.shape[:2]

    scale = 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    ref_resized = cv2.resize(ref_frame, (new_w, new_h))
    curr_resized = cv2.resize(curr_frame, (new_w, new_h))

    # Draw YOLO detections
    if yolo_active and yolo_results is not None:
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = (
                    int(x1 * scale),
                    int(y1 * scale),
                    int(x2 * scale),
                    int(y2 * scale),
                )

                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{result.names[cls]} {conf:.2f}"

                cv2.rectangle(curr_resized, (x1, y1), (x2, y2), (255, 0, 255), 2)

                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    curr_resized,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    (255, 0, 255),
                    -1,
                )
                cv2.putText(
                    curr_resized,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

    # Draw feature matches (green)
    if len(mkpts_0) > 0 and len(mkpts_1) > 0:
        for pt1, pt2 in zip(mkpts_0[:50], mkpts_1[:50]):
            x1, y1 = int(pt1[0] * scale), int(pt1[1] * scale)
            x2, y2 = int(pt2[0] * scale), int(pt2[1] * scale)

            cv2.circle(ref_resized, (x1, y1), 3, (0, 255, 0), -1)
            cv2.circle(curr_resized, (x2, y2), 3, (0, 255, 0), -1)

    # Draw optical flow points (yellow)
    if flow_points is not None:
        curr_resized = draw_optical_flow_vectors(
            curr_resized, flow_points, color=(0, 255, 255)
        )

    # Add labels
    cv2.rectangle(ref_resized, (0, 0), (200, 40), (50, 50, 50), -1)
    cv2.putText(
        ref_resized,
        "REFERENCE",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (100, 200, 255),
        2,
    )

    cv2.rectangle(curr_resized, (0, 0), (200, 40), (50, 50, 50), -1)
    cv2.putText(
        curr_resized,
        "CURRENT",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # Create 3D guidance and artificial horizon side by side
    guidance_width = 400
    
    if show_guidance:
        guidance_viz = create_slick_3d_guidance(
            dx, dy, dz_scale, confidence, guidance_source, size=(guidance_width, new_h)
        )
    else:
        guidance_viz = np.zeros((new_h, guidance_width, 3), dtype=np.uint8)
        cv2.putText(
            guidance_viz,
            "Guidance Disabled",
            (100, new_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (100, 100, 100),
            2,
        )
    
    # Add artificial horizon if IMU data available
    horizon_width = 300
    if imu_data is not None:
        attitude = imu_data.get_attitude()
        if attitude['connected']:
            # Create horizon without title, will add title separately
            horizon_display = draw_artificial_horizon(
                attitude['roll'], 
                attitude['pitch'], 
                attitude['yaw'],
                size=(horizon_width, horizon_width)  # Square for circular display
            )
            # Resize to match height and add padding if needed
            if horizon_display.shape[0] != new_h:
                horizon_viz = np.zeros((new_h, horizon_width, 3), dtype=np.uint8)
                horizon_viz[:] = [30, 30, 30]
                # Center the horizon display vertically
                y_offset = (new_h - horizon_display.shape[0]) // 2
                if y_offset > 0:
                    horizon_viz[y_offset:y_offset+horizon_display.shape[0], :] = horizon_display
                else:
                    # Crop if too large
                    horizon_viz = horizon_display[:new_h, :]
            else:
                horizon_viz = horizon_display
            
            # Add title bar at top
            cv2.rectangle(horizon_viz, (0, 0), (horizon_width, 35), (50, 50, 50), -1)
            cv2.putText(horizon_viz, "ATTITUDE INDICATOR", (35, 23), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
        else:
            horizon_viz = np.zeros((new_h, horizon_width, 3), dtype=np.uint8)
            horizon_viz[:] = [30, 30, 30]
            cv2.putText(horizon_viz, "IMU", (100, new_h // 2 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            cv2.putText(horizon_viz, "Disconnected", (70, new_h // 2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    else:
        horizon_viz = np.zeros((new_h, horizon_width, 3), dtype=np.uint8)
        horizon_viz[:] = [30, 30, 30]
        cv2.putText(horizon_viz, "No IMU", (90, new_h // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

    top_row = np.hstack([ref_resized, curr_resized, guidance_viz, horizon_viz])

    # Info panel - EXPANDED with IMU
    info_h = 160
    info_panel = np.zeros((info_h, top_row.shape[1], 3), dtype=np.uint8)
    info_panel[:] = [30, 35, 45]

    match_ratio = (num_matches / max_matches * 100) if max_matches > 0 else 0

    # Left - Match info
    cv2.putText(
        info_panel,
        f"Matches: {num_matches}/{max_matches}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        info_panel,
        f"Ratio: {match_ratio:.1f}%",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 200, 255),
        1,
    )

    # Stability bar
    bar_x, bar_y, bar_w = 20, 75, 180
    cv2.rectangle(
        info_panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + 15), (60, 60, 80), -1
    )
    filled = int(bar_w * (stability / 100))
    if filled > 0:
        bar_color = (0, int(255 * stability / 100), int(255 * (1 - stability / 100)))
        cv2.rectangle(
            info_panel, (bar_x, bar_y), (bar_x + filled, bar_y + 15), bar_color, -1
        )
    cv2.rectangle(
        info_panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + 15), (120, 120, 150), 2
    )
    cv2.putText(
        info_panel,
        f"Stab: {stability:.0f}%",
        (bar_x + bar_w + 10, bar_y + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
    )

    flow_count = len(flow_points) if flow_points is not None else 0
    cv2.putText(
        info_panel,
        f"Flow pts: {flow_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 100),
        1,
    )

    # Middle - GUIDANCE COMMANDS
    cmd_x = 280
    cv2.putText(
        info_panel,
        "GUIDANCE:",
        (cmd_x, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 200, 100),
        2,
    )

    cmd_y = 50
    if show_guidance:
        threshold = 10
        if abs(dx) > threshold:
            cmd_text = "← LEFT" if dx < 0 else "RIGHT →"
            cv2.putText(
                info_panel,
                cmd_text,
                (cmd_x, cmd_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                info_panel,
                f"{abs(dx):.0f}px",
                (cmd_x + 90, cmd_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (180, 180, 200),
                1,
            )
            cmd_y += 25

        if abs(dy) > threshold:
            cmd_text = "↑ UP" if dy < 0 else "DOWN ↓"
            cv2.putText(
                info_panel,
                cmd_text,
                (cmd_x, cmd_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                info_panel,
                f"{abs(dy):.0f}px",
                (cmd_x + 90, cmd_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (180, 180, 200),
                1,
            )
            cmd_y += 25

        if dz_scale > 1.1:
            cv2.putText(
                info_panel,
                "↗ CLOSER",
                (cmd_x, cmd_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 255, 150),
                1,
            )
            cv2.putText(
                info_panel,
                f"{dz_scale:.2f}x",
                (cmd_x + 90, cmd_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (180, 200, 180),
                1,
            )
        elif dz_scale < 0.9:
            cv2.putText(
                info_panel,
                "↙ AWAY",
                (cmd_x, cmd_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 150, 255),
                1,
            )
            cv2.putText(
                info_panel,
                f"{dz_scale:.2f}x",
                (cmd_x + 90, cmd_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (180, 180, 200),
                1,
            )
        else:
            cv2.putText(
                info_panel,
                "✓ ALIGNED",
                (cmd_x, cmd_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 255, 150),
                2,
            )
    else:
        cv2.putText(
            info_panel,
            "Disabled",
            (cmd_x, cmd_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 100, 100),
            1,
        )

    # Source & Confidence with color coding
    source_colors = {
        "features": (150, 200, 255),
        "objects": (255, 200, 100),
        "optical_flow": (255, 255, 100),
        "imu_fused": (100, 255, 200),
    }
    source_color = source_colors.get(guidance_source, (200, 200, 200))
    cv2.putText(
        info_panel,
        f"Src: {guidance_source.upper()}",
        (cmd_x, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        source_color,
        1,
    )
    cv2.putText(
        info_panel,
        f"Conf: {confidence:.0f}%",
        (cmd_x, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 220, 255),
        1,
    )

    # YOLO status
    yolo_x = 600
    yolo_status = "YOLO: ON" if yolo_active else "YOLO: OFF"
    yolo_color = (100, 255, 100) if yolo_active else (120, 120, 120)
    cv2.circle(info_panel, (yolo_x, 35), 10, yolo_color, -1)
    cv2.circle(info_panel, (yolo_x, 35), 12, (200, 200, 200), 2)
    cv2.putText(
        info_panel,
        yolo_status,
        (yolo_x + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        yolo_color,
        1,
    )

    # FPS
    cv2.putText(
        info_panel,
        f"FPS: {fps:.1f}",
        (yolo_x, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
    )

    # Right - Controls
    ctrl_x = top_row.shape[1] - 350
    cv2.putText(
        info_panel,
        "CONTROLS:",
        (ctrl_x, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 200, 100),
        2,
    )
    cv2.putText(
        info_panel,
        "S: Capture | R: Reset",
        (ctrl_x, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        info_panel,
        "G: Guidance | Q: Quit",
        (ctrl_x, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
    )

    # Add performance graphs below info panel
    if performance_monitor is not None:
        perf_panel = performance_monitor.draw_performance_panel(size=(top_row.shape[1], 300))
        main_display = np.vstack([top_row, info_panel, perf_panel])
    else:
        main_display = np.vstack([top_row, info_panel])
    
    # Create comprehensive IMU/EKF/Fusion panels if IMU available
    if imu_data is not None:
        panel_height = 200
        
        # Create three separate panels
        imu_raw_panel = draw_imu_raw_panel(imu_data, panel_h=panel_height)
        ekf_panel = draw_ekf_panel(imu_data, panel_h=panel_height)
        
        if fusion_info is not None:
            fusion_panel = draw_fusion_panel(
                visual_dx, visual_dy, visual_dz, visual_conf,
                dx, dy, dz_scale, confidence,
                fusion_info, panel_h=panel_height
            )
        else:
            fusion_panel = np.zeros((panel_height, 350, 3), dtype=np.uint8)
            fusion_panel[:] = [40, 40, 50]
            cv2.putText(fusion_panel, "No Fusion Data", (80, panel_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        # Stack panels horizontally
        sensor_panels = np.hstack([imu_raw_panel, ekf_panel, fusion_panel])
        
        # Pad to match main display width if needed
        if sensor_panels.shape[1] < main_display.shape[1]:
            padding = np.zeros((panel_height, main_display.shape[1] - sensor_panels.shape[1], 3), dtype=np.uint8)
            padding[:] = [25, 25, 30]
            sensor_panels = np.hstack([sensor_panels, padding])
        elif sensor_panels.shape[1] > main_display.shape[1]:
            sensor_panels = sensor_panels[:, :main_display.shape[1]]
        
        # Stack vertically: main display on top, sensor panels on bottom
        final_viz = np.vstack([main_display, sensor_panels])
    else:
        final_viz = main_display

    return final_viz


def process_frame_for_xfeat(frame, device):
    """Convert frame to XFeat input format"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = (
        torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )
    if device == "cuda":
        frame_tensor = frame_tensor.cuda()
    return frame_tensor


def main():
    print("=" * 75)
    print("XFeat + YOLOv11 + Optical Flow + Pixhawk IMU Sensor Fusion")
    print("=" * 75)
    
    # Initialize Pixhawk IMU
    print("\n[1/4] Initializing Pixhawk IMU...")
    
    # Try different connection strings
    connection_options = [
        '/dev/ttyACM0',  # Common on Linux
        '/dev/ttyUSB0',  # Alternative Linux
        'COM3',          # Windows
        'udp:127.0.0.1:14550',  # SITL/MAVProxy
    ]
    
    print("Available connection options:")
    for i, opt in enumerate(connection_options, 1):
        print(f"  {i}. {opt}")
    print("  5. Custom connection string")
    print("  6. Skip IMU (visual only)")
    
    choice = input("\nSelect option (1-6) [default=6]: ").strip()
    
    imu_data = None
    if choice in ['1', '2', '3', '4']:
        conn_str = connection_options[int(choice) - 1]
        imu_data = PixhawkIMU(conn_str)
        if imu_data.connect():
            imu_data.start()
        else:
            print("Continuing without IMU...")
            imu_data = None
    elif choice == '5':
        conn_str = input("Enter connection string: ").strip()
        imu_data = PixhawkIMU(conn_str)
        if imu_data.connect():
            imu_data.start()
        else:
            print("Continuing without IMU...")
            imu_data = None
    else:
        print("Skipping IMU, using visual guidance only")

    print("\n[2/5] Initializing XFeat...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    xfeat = XFeat()
    if device == "cuda":
        xfeat = xfeat.cuda()

    print("\n[3/5] Loading YOLOv11 nano model...")
    yolo_model = YOLO("yolo11n.pt")
    print("YOLOv11 loaded successfully!")

    print("\n[4/5] Initializing performance monitor...")
    performance_monitor = PerformanceMonitor(history_length=100)
    print("✓ Performance monitor ready")

    print("\n[5/5] Opening webcam...")
    cap = cv2.VideoCapture(4)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 75)
    print("System Ready!")
    print("=" * 75)
    print("Features:")
    print("  • IMU/Gyro attitude and acceleration from Pixhawk")
    print("  • Sensor fusion: Visual + Inertial guidance")
    print("  • Quad guidance: IMU → Features → Objects → Optical Flow")
    print("  • Complementary filter for robust position estimation")
    print("  • Artificial Horizon attitude indicator")
    print("  • Real-time performance graphs (FPS, confidence, matches, velocity)")
    print("\nGuidance Priority:")
    print("  1. IMU-fused visual guidance (most robust)")
    print("  2. XFeat homography (accurate visual)")
    print("  3. Object-based tracking (semantic)")
    print("  4. Optical flow (motion continuity)")
    print("\nUI Components:")
    print("  • Left: Reference and current camera views")
    print("  • Center: 3D guidance vectors")
    print("  • Right: Artificial horizon (aviation-style)")
    print("  • Bottom: Performance graphs + IMU/EKF/Fusion panels")
    print("\nControls:")
    print("  S - Capture reference  |  R - Reset  |  G - Toggle guidance  |  Q - Quit")
    print("=" * 75 + "\n")

    # State variables
    reference_frame = None
    reference_tensor = None
    reference_gray = None
    reference_objects = []
    max_matches = 0
    fps_history = []
    yolo_active = False
    guidance_enabled = True

    flow_tracked_points = None

    activate_threshold = 45.0
    deactivate_threshold = 30.0

    match_history = []
    stability_window = 10
    stability_threshold = 40.0

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                reference_frame = frame.copy()
                reference_tensor = process_frame_for_xfeat(reference_frame, device)
                reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

                with torch.no_grad():
                    ref_output = xfeat.detectAndCompute(reference_tensor, top_k=4096)[0]
                    max_matches = len(ref_output["keypoints"])

                ref_results = yolo_model(reference_frame, verbose=False)
                reference_objects = extract_object_info(ref_results)

                flow_tracked_points = None
                
                if imu_data:
                    imu_data.reset_velocity()

                print(
                    f"✓ Reference captured! ({max_matches} features, {len(reference_objects)} objects)"
                )
                fps_history = []
                match_history = []
                yolo_active = False

            elif key == ord("r"):
                reference_frame = None
                reference_tensor = None
                reference_gray = None
                reference_objects = []
                max_matches = 0
                yolo_active = False
                match_history = []
                flow_tracked_points = None
                if imu_data:
                    imu_data.reset_velocity()
                print("Reference cleared")
                fps_history = []

            elif key == ord("g"):
                guidance_enabled = not guidance_enabled
                print(f"Guidance: {'ENABLED' if guidance_enabled else 'DISABLED'}")

            if reference_frame is None:
                cv2.putText(
                    frame,
                    "Press 'S' to capture reference frame",
                    (50, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("XFeat + YOLO + Flow + IMU", frame)
                continue

            # Process current frame
            current_tensor = process_frame_for_xfeat(frame, device)
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            with torch.no_grad():
                mkpts_0, mkpts_1 = xfeat.match_xfeat(
                    reference_tensor, current_tensor, top_k=4096
                )

            if isinstance(mkpts_0, torch.Tensor):
                mkpts_0 = mkpts_0.cpu().numpy()
            if isinstance(mkpts_1, torch.Tensor):
                mkpts_1 = mkpts_1.cpu().numpy()

            num_matches = len(mkpts_0)
            match_ratio = (num_matches / max_matches * 100) if max_matches > 0 else 0

            match_history.append(match_ratio)
            if len(match_history) > stability_window:
                match_history.pop(0)

            if len(match_history) >= 3:
                avg_ratio = np.mean(match_history)
                variance = np.var(match_history)
                stability = min(100, avg_ratio * (1 - min(1, variance / 1000)))
            else:
                stability = match_ratio

            # Hysteresis logic
            if not yolo_active:
                should_activate = (
                    match_ratio >= activate_threshold
                    and stability >= stability_threshold
                )
                if should_activate:
                    print(f"✓ YOLO ACTIVATED - Ratio: {match_ratio:.1f}%")
                    yolo_active = True
            else:
                should_deactivate = match_ratio < deactivate_threshold
                if should_deactivate:
                    print(f"✗ YOLO DEACTIVATED - Ratio: {match_ratio:.1f}%")
                    yolo_active = False

            curr_results = yolo_model(frame, verbose=False) if yolo_active else None

            # === HYBRID GUIDANCE CASCADE WITH IMU FUSION ===
            dx, dy, dz_scale, confidence, found = 0, 0, 1.0, 0.0, False
            guidance_source = "none"

            # Priority 1: XFeat homography
            dx, dy, dz_scale, confidence, found = estimate_camera_guidance(
                mkpts_0, mkpts_1
            )
            if found and confidence > 30:
                guidance_source = "features"
            else:
                # Priority 2: Object-based guidance
                if yolo_active and len(reference_objects) > 0:
                    current_objects = extract_object_info(curr_results)
                    dx_obj, dy_obj, dz_obj, conf_obj, found_obj = (
                        estimate_guidance_from_objects(
                            reference_objects, current_objects
                        )
                    )
                    if found_obj and conf_obj > confidence:
                        dx, dy, dz_scale, confidence, found = (
                            dx_obj,
                            dy_obj,
                            dz_obj,
                            conf_obj,
                            found_obj,
                        )
                        guidance_source = "objects"

                # Priority 3: Optical flow
                (
                    dx_flow,
                    dy_flow,
                    dz_flow,
                    conf_flow,
                    found_flow,
                    flow_tracked_points,
                ) = estimate_guidance_from_optical_flow(
                    reference_gray, current_gray, flow_tracked_points
                )

                if found_flow and (not found or conf_flow > confidence):
                    dx, dy, dz_scale, confidence, found = (
                        dx_flow,
                        dy_flow,
                        dz_flow,
                        conf_flow,
                        found_flow,
                    )
                    guidance_source = "optical_flow"

            # Store visual-only results for comparison
            visual_dx_only = dx
            visual_dy_only = dy
            visual_dz_only = dz_scale
            visual_conf_only = confidence
            fusion_info = None
            
            # Priority 0: Fuse with IMU if available
            if imu_data is not None and found:
                dx, dy, dz_scale, confidence, fusion_info = fuse_visual_imu_guidance(
                    dx, dy, dz_scale, confidence, imu_data
                )
                guidance_source = "imu_fused"

            # Calculate FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history) if fps_history else fps
            
            # Update performance monitor
            ekf_vel_mag = 0.0
            if imu_data is not None:
                ekf = imu_data.get_ekf_data()
                if ekf['age'] < 1.0:
                    ekf_vel_mag = np.sqrt(ekf['vel_x']**2 + ekf['vel_y']**2 + ekf['vel_z']**2)
            
            performance_monitor.update(avg_fps, confidence, num_matches, ekf_vel_mag)

            # Create unified visualization
            show_guidance = guidance_enabled and found

            unified_viz = draw_unified_window(
                reference_frame,
                frame,
                mkpts_0,
                mkpts_1,
                curr_results,
                dx,
                dy,
                dz_scale,
                confidence,
                guidance_source,
                yolo_active,
                num_matches,
                max_matches,
                avg_fps,
                stability,
                show_guidance,
                flow_tracked_points,
                imu_data,
                visual_dx_only if found else 0,
                visual_dy_only if found else 0,
                visual_dz_only if found else 1.0,
                visual_conf_only if found else 0,
                fusion_info,
                performance_monitor,
            )

            cv2.imshow("XFeat + YOLO + Flow + IMU", unified_viz)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        if imu_data:
            imu_data.stop()
        cap.release()
        cv2.destroyAllWindows()
        if fps_history:
            print(f"\nAverage FPS: {np.mean(fps_history):.1f}")
        print("Done!")


if __name__ == "__main__":
    main()

"""
Modern Dashboard - Complete Redesign
Inspired by reference UI with circular gauges, 3D visualization, and modular tabs
"""

import cv2
import numpy as np
from .widgets import CircularGauge, DataPanel, Heading3D, StatusIndicator


class ModernDashboard:
    """Modern dashboard with central overview and detailed subsystem tabs"""
    
    def __init__(self):
        self.active_tab = 0  # 0=Central, 1=Vision, 2=IMU/Fusion, 3=Flow, 4=Performance
        self.window_width = 1920
        self.window_height = 1080
        
        # Dark BLUE theme colors (not brown!)
        self.colors = {
            'bg_dark': (35, 25, 15),           # Dark blue-gray
            'bg_medium': (45, 35, 20),         # Medium blue-gray
            'bg_light': (60, 50, 30),          # Light blue-gray for borders
            'accent': (120, 90, 50),           # Blue accent
            'highlight': (150, 110, 60),       # Bright blue
            'text_primary': (240, 230, 210),   # Warm white
            'text_secondary': (190, 180, 160), # Light gray
            'success': (100, 200, 100),        # Green
            'warning': (100, 200, 255),        # Yellow
            'error': (100, 100, 255),          # Red
        }
        
        # Create widgets (smaller sizes)
        self.gauge_left = CircularGauge(size=180, max_value=2000, label="Speed X", units="px/s")
        self.gauge_right = CircularGauge(size=180, max_value=2000, label="Speed Y", units="px/s")
        self.heading_3d = Heading3D(size=400)
        
    def create_tab_bar(self, width, height=50):
        """Create modern tab bar"""
        bar = np.zeros((height, width, 3), dtype=np.uint8)
        bar[:] = self.colors['bg_medium']
        
        tabs = [
            ("CENTRAL", "*"),
            ("VISION", "#"),
            ("IMU/FUSION", "+"),
            ("FLOW", "~"),
            ("PERFORMANCE", "=")
        ]
        
        tab_width = width // len(tabs)
        
        for i, (tab_name, icon) in enumerate(tabs):
            x_start = i * tab_width
            x_end = (i + 1) * tab_width
            
            # Active tab
            if i == self.active_tab:
                cv2.rectangle(bar, (x_start, 0), (x_end, height-4), self.colors['accent'], -1)
                cv2.rectangle(bar, (x_start, height-4), (x_end, height), self.colors['highlight'], -1)
                text_color = self.colors['text_primary']
            else:
                text_color = self.colors['text_secondary']
            
            # Separator
            if i > 0:
                cv2.line(bar, (x_start, 10), (x_start, height-10), self.colors['bg_light'], 2)
            
            # Text
            text = f"{icon} {tab_name}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
            text_x = x_start + (tab_width - text_size[0]) // 2
            text_y = height // 2 + text_size[1] // 2
            
            cv2.putText(bar, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1, cv2.LINE_AA)
        
        # Controls hint
        hint = "[1-5] Tabs | [S] Capture | [R] Reset | [Q] Quit"
        cv2.putText(bar, hint, (width - 450, height - 15),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, self.colors['text_secondary'], 1, cv2.LINE_AA)
        
        return bar
    
    def draw_central_tab(self, ref_frame, curr_frame, mkpts_0, mkpts_1, yolo_results,
                        dx, dy, dz_scale, confidence, guidance_source, yolo_active,
                        num_matches, max_matches, fps, stability, show_guidance,
                        flow_points, imu_data):
        """Central dashboard - all key visualizations"""
        content_height = self.window_height - 60
        canvas = np.zeros((content_height, self.window_width, 3), dtype=np.uint8)
        canvas[:] = self.colors['bg_dark']
        
        # Title (smaller)
        cv2.putText(canvas, "CENTRAL COMMAND", (30, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, self.colors['highlight'], 1, cv2.LINE_AA)
        
        # Check if reference frame exists
        if ref_frame is None or curr_frame is None:
            msg1 = "NO REFERENCE FRAME"
            msg2 = "Press [S] to capture reference frame"
            cv2.putText(canvas, msg1, (self.window_width // 2 - 200, content_height // 2 - 30),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['warning'], 2, cv2.LINE_AA)
            cv2.putText(canvas, msg2, (self.window_width // 2 - 280, content_height // 2 + 20),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text_secondary'], 1, cv2.LINE_AA)
            return canvas
        
        # === LEFT COLUMN: Gauges + Motor Panel ===
        left_x = 30
        gauge_y = 60
        
        # Speed X gauge (smaller)
        vel_x = abs(dx) * 0.5
        gauge_img = self.gauge_left.draw(vel_x)
        canvas[gauge_y:gauge_y+180, left_x:left_x+180] = gauge_img
        
        # Speed Y gauge
        vel_y = abs(dy) * 0.5
        gauge_img = self.gauge_right.draw(vel_y)
        canvas[gauge_y:gauge_y+180, left_x+200:left_x+380] = gauge_img
        
        # Motor rotation panel (smaller and more compact)
        motor_panel_y = gauge_y + 200
        motor_panel = self._create_motor_panel(imu_data, dx, dy, dz_scale)
        canvas[motor_panel_y:motor_panel_y+220, left_x:left_x+380] = motor_panel
        
        # === CENTER: 3D Heading + Altimeter ===
        center_x = 440
        heading_y = 60
        
        # 3D coordinate system (smaller)
        if imu_data is not None:
            attitude = imu_data.get_attitude()
            if attitude['connected']:
                heading_img = self.heading_3d.draw(
                    attitude['roll'], attitude['pitch'], attitude['yaw'],
                    dx, dy, dz_scale - 1.0
                )
                cv2.rectangle(canvas, (center_x-2, heading_y-2),
                             (center_x+402, heading_y+402), self.colors['bg_light'], 2)
                canvas[heading_y:heading_y+400, center_x:center_x+400] = heading_img
        else:
            cv2.rectangle(canvas, (center_x, heading_y),
                         (center_x+400, heading_y+400), self.colors['bg_medium'], -1)
            cv2.putText(canvas, "IMU NOT CONNECTED", (center_x + 100, heading_y + 200),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text_secondary'], 1, cv2.LINE_AA)
        
        # Altimeter below 3D view
        alt_y = heading_y + 420
        if imu_data is not None:
            attitude = imu_data.get_attitude()
            if attitude['connected']:
                from .horizon_viz import draw_artificial_horizon
                altimeter = draw_artificial_horizon(
                    attitude['roll'], attitude['pitch'], attitude['yaw'], size=(380, 380)
                )
                cv2.rectangle(canvas, (center_x-2, alt_y-2),
                             (center_x+382, alt_y+382), self.colors['bg_light'], 2)
                canvas[alt_y:alt_y+380, center_x+10:center_x+390] = altimeter
        
        # === RIGHT COLUMN: Camera Views ===
        cam_x = 860
        cam_y = 60
        cam_w = 520
        cam_h = 390
        
        # Reference frame (top)
        ref_resized = cv2.resize(ref_frame, (cam_w, cam_h))
        
        # Draw feature matches on reference
        if len(mkpts_0) > 0:
            scale_x = cam_w / ref_frame.shape[1]
            scale_y = cam_h / ref_frame.shape[0]
            for pt in mkpts_0[:30]:
                x, y = int(pt[0] * scale_x), int(pt[1] * scale_y)
                cv2.circle(ref_resized, (x, y), 3, (100, 255, 100), -1)
        
        # Header
        cv2.rectangle(ref_resized, (0, 0), (cam_w, 30), (40, 60, 90), -1)
        cv2.putText(ref_resized, "REFERENCE", (10, 21),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, self.colors['text_primary'], 1, cv2.LINE_AA)
        
        cv2.rectangle(canvas, (cam_x-2, cam_y-2), (cam_x+cam_w+2, cam_y+cam_h+2),
                     self.colors['bg_light'], 2)
        canvas[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = ref_resized
        
        # Current frame (bottom)
        curr_y = cam_y + cam_h + 20
        curr_resized = cv2.resize(curr_frame, (cam_w, cam_h))
        
        # Draw feature matches
        if len(mkpts_1) > 0:
            scale_x = cam_w / curr_frame.shape[1]
            scale_y = cam_h / curr_frame.shape[0]
            for pt in mkpts_1[:30]:
                x, y = int(pt[0] * scale_x), int(pt[1] * scale_y)
                cv2.circle(curr_resized, (x, y), 3, (100, 255, 100), -1)
        
        # Draw YOLO detections - ALWAYS SHOW
        if yolo_results is not None:
            scale_x = cam_w / curr_frame.shape[1]
            scale_y = cam_h / curr_frame.shape[0]
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{result.names[cls]} {conf:.2f}"
                    
                    # Draw box
                    cv2.rectangle(curr_resized, (x1, y1), (x2, y2), (255, 100, 255), 2)
                    
                    # Draw label
                    cv2.rectangle(curr_resized, (x1, y1-20), (x1+len(label)*8, y1), (255, 100, 255), -1)
                    cv2.putText(curr_resized, label, (x1+2, y1-5),
                               cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Header
        header_color = (60, 90, 40) if yolo_active else (60, 40, 40)
        cv2.rectangle(curr_resized, (0, 0), (cam_w, 30), header_color, -1)
        status_text = "CURRENT (YOLO ON)" if yolo_active else "CURRENT (YOLO OFF)"
        cv2.putText(curr_resized, status_text, (10, 21),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, self.colors['text_primary'], 1, cv2.LINE_AA)
        
        cv2.rectangle(canvas, (cam_x-2, curr_y-2), (cam_x+cam_w+2, curr_y+cam_h+2),
                     self.colors['bg_light'], 2)
        canvas[curr_y:curr_y+cam_h, cam_x:cam_x+cam_w] = curr_resized
        
        # === BOTTOM: Compact Status Panels ===
        status_y = content_height - 140
        
        # System status (smaller)
        status_data = {
            "FPS": f"{fps:.1f}",
            "Matches": f"{num_matches}/{max_matches}",
            "Confidence": f"{confidence:.0f}%",
            "Guidance": guidance_source.upper(),
        }
        status_panel = DataPanel(width=380, height=120, title="SYSTEM STATUS").draw(status_data)
        canvas[status_y:status_y+120, left_x:left_x+380] = status_panel
        
        # Guidance info
        guidance_data = {
            "DX": f"{dx:+.1f} px",
            "DY": f"{dy:+.1f} px",
            "DZ": f"{dz_scale:.3f}x",
            "Status": "ALIGNED" if abs(dx) < 10 and abs(dy) < 10 else "ADJUSTING",
        }
        guidance_panel = DataPanel(width=380, height=120, title="GUIDANCE").draw(guidance_data)
        canvas[status_y:status_y+120, center_x:center_x+380] = guidance_panel
        
        # EKF stats
        if imu_data is not None:
            ekf = imu_data.get_ekf_data()
            vel_mag = np.sqrt(ekf['vel_x']**2 + ekf['vel_y']**2 + ekf['vel_z']**2)
            ekf_data = {
                "Vel X": f"{ekf['vel_x']:.2f} m/s",
                "Vel Y": f"{ekf['vel_y']:.2f} m/s",
                "Vel Mag": f"{vel_mag:.2f} m/s",
                "Health": "OK" if ekf['healthy'] else "DEGRADED",
            }
            ekf_panel = DataPanel(width=500, height=120, title="EKF DATA").draw(ekf_data)
            canvas[status_y:status_y+120, cam_x:cam_x+500] = ekf_panel
        
        return canvas
    
    def _create_motor_panel(self, imu_data, dx, dy, dz_scale):
        """Create compact motor rotation panel"""
        panel = np.zeros((220, 380, 3), dtype=np.uint8)
        panel[:] = self.colors['bg_medium']
        
        # Border
        cv2.rectangle(panel, (0, 0), (379, 219), self.colors['bg_light'], 2)
        
        # Title (smaller)
        cv2.rectangle(panel, (0, 0), (380, 28), self.colors['accent'], -1)
        cv2.putText(panel, "MOTOR ROTATION", (10, 19),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, self.colors['text_primary'], 1, cv2.LINE_AA)
        
        if imu_data is not None:
            attitude = imu_data.get_attitude()
            
            # Left column: Gyroscope (compact)
            y = 45
            cv2.putText(panel, "Gyroscope:", (12, y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, self.colors['text_secondary'], 1, cv2.LINE_AA)
            y += 23
            
            gyro_data = [
                ("X:", f"{np.degrees(attitude['rollspeed']):+6.2f}°/s"),
                ("Y:", f"{np.degrees(attitude['pitchspeed']):+6.2f}°/s"),
                ("Z:", f"{np.degrees(attitude['yawspeed']):+6.2f}°/s"),
            ]
            
            for label, value in gyro_data:
                cv2.putText(panel, label, (20, y),
                           cv2.FONT_HERSHEY_DUPLEX, 0.35, self.colors['text_secondary'], 1, cv2.LINE_AA)
                cv2.putText(panel, value, (50, y),
                           cv2.FONT_HERSHEY_DUPLEX, 0.4, (100, 200, 255), 1, cv2.LINE_AA)
                y += 22
            
            # Right column: Movement (compact)
            y = 45
            cv2.putText(panel, "Movement:", (200, y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, self.colors['text_secondary'], 1, cv2.LINE_AA)
            y += 23
            
            move_data = [
                ("X:", f"{dx:+6.1f} px"),
                ("Y:", f"{dy:+6.1f} px"),
                ("Z:", f"{(dz_scale-1.0)*100:+6.1f}%"),
            ]
            
            for label, value in move_data:
                cv2.putText(panel, label, (210, y),
                           cv2.FONT_HERSHEY_DUPLEX, 0.35, self.colors['text_secondary'], 1, cv2.LINE_AA)
                cv2.putText(panel, value, (240, y),
                           cv2.FONT_HERSHEY_DUPLEX, 0.4, (100, 255, 200), 1, cv2.LINE_AA)
                y += 22
            
            # Heading indicator (compact)
            heading = (np.degrees(attitude['yaw']) + 360) % 360
            heading_text = "FORWARD"
            if 45 < heading < 135:
                heading_text = "LEFT"
            elif 135 < heading < 225:
                heading_text = "BACKWARD"
            elif 225 < heading < 315:
                heading_text = "RIGHT"
            
            cv2.rectangle(panel, (110, 155), (270, 205), self.colors['accent'], -1)
            cv2.rectangle(panel, (110, 155), (270, 205), self.colors['highlight'], 2)
            
            text_size = cv2.getTextSize(heading_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)[0]
            text_x = 190 - text_size[0] // 2
            cv2.putText(panel, heading_text, (text_x, 185),
                       cv2.FONT_HERSHEY_DUPLEX, 0.65, self.colors['text_primary'], 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "IMU NOT CONNECTED", (110, 110),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, self.colors['text_secondary'], 1, cv2.LINE_AA)
        
        return panel
    
    def draw_vision_tab(self, ref_frame, curr_frame, mkpts_0, mkpts_1, yolo_results,
                       num_matches, max_matches, confidence):
        """Detailed vision subsystem tab"""
        content_height = self.window_height - 60
        canvas = np.zeros((content_height, self.window_width, 3), dtype=np.uint8)
        canvas[:] = self.colors['bg_dark']
        
        # Title
        cv2.putText(canvas, "VISION SUBSYSTEM - DETAILED VIEW", (40, 45),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['highlight'], 2, cv2.LINE_AA)
        
        # Check if frames exist
        if ref_frame is None or curr_frame is None:
            msg1 = "NO REFERENCE FRAME"
            msg2 = "Press [S] to capture reference frame"
            cv2.putText(canvas, msg1, (self.window_width // 2 - 200, content_height // 2 - 30),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['warning'], 2, cv2.LINE_AA)
            cv2.putText(canvas, msg2, (self.window_width // 2 - 280, content_height // 2 + 20),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text_secondary'], 1, cv2.LINE_AA)
            return canvas
        
        # Large camera views with all detections
        cam_w = 880
        cam_h = 660
        margin = 40
        
        # Reference frame (left)
        ref_resized = cv2.resize(ref_frame, (cam_w, cam_h))
        if len(mkpts_0) > 0:
            scale_x = cam_w / ref_frame.shape[1]
            scale_y = cam_h / ref_frame.shape[0]
            for pt in mkpts_0[:100]:
                x, y = int(pt[0] * scale_x), int(pt[1] * scale_y)
                cv2.circle(ref_resized, (x, y), 4, (100, 255, 100), -1)
                cv2.circle(ref_resized, (x, y), 5, (150, 255, 150), 1)
        
        # Header
        cv2.rectangle(ref_resized, (0, 0), (cam_w, 40), (40, 60, 90), -1)
        cv2.putText(ref_resized, f"REFERENCE - {len(mkpts_0)} Features", (15, 28),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text_primary'], 1, cv2.LINE_AA)
        
        canvas[80:80+cam_h, margin:margin+cam_w] = ref_resized
        cv2.rectangle(canvas, (margin-2, 78), (margin+cam_w+2, 82+cam_h),
                     self.colors['bg_light'], 3)
        
        # Current frame (right)
        curr_x = margin * 2 + cam_w
        curr_resized = cv2.resize(curr_frame, (cam_w, cam_h))
        
        # Draw feature matches
        if len(mkpts_1) > 0:
            scale_x = cam_w / curr_frame.shape[1]
            scale_y = cam_h / curr_frame.shape[0]
            for pt in mkpts_1[:100]:
                x, y = int(pt[0] * scale_x), int(pt[1] * scale_y)
                cv2.circle(curr_resized, (x, y), 4, (100, 255, 100), -1)
                cv2.circle(curr_resized, (x, y), 5, (150, 255, 150), 1)
        
        # Draw YOLO detections
        if yolo_results is not None:
            scale_x = cam_w / curr_frame.shape[1]
            scale_y = cam_h / curr_frame.shape[0]
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                    
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{result.names[cls]} {conf:.2f}"
                    
                    cv2.rectangle(curr_resized, (x1, y1), (x2, y2), (255, 100, 255), 3)
                    cv2.rectangle(curr_resized, (x1, y1-25), (x1+len(label)*10, y1), (255, 100, 255), -1)
                    cv2.putText(curr_resized, label, (x1+3, y1-8),
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Header
        cv2.rectangle(curr_resized, (0, 0), (cam_w, 40), (60, 40, 90), -1)
        cv2.putText(curr_resized, f"CURRENT - {len(mkpts_1)} Matches", (15, 28),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text_primary'], 1, cv2.LINE_AA)
        
        canvas[80:80+cam_h, curr_x:curr_x+cam_w] = curr_resized
        cv2.rectangle(canvas, (curr_x-2, 78), (curr_x+cam_w+2, 82+cam_h),
                     self.colors['bg_light'], 3)
        
        # Bottom info panels
        info_y = 80 + cam_h + 30
        
        # Match statistics
        match_ratio = (num_matches / max_matches * 100) if max_matches > 0 else 0
        match_data = {
            "Total Features": str(max_matches),
            "Current Matches": str(num_matches),
            "Match Ratio": f"{match_ratio:.1f}%",
            "Confidence": f"{confidence:.0f}%",
        }
        match_panel = DataPanel(width=600, height=200, title="FEATURE MATCHING").draw(match_data)
        canvas[info_y:info_y+200, margin:margin+600] = match_panel
        
        # YOLO statistics
        num_detections = 0
        if yolo_results is not None:
            for result in yolo_results:
                num_detections += len(result.boxes)
        
        yolo_data = {
            "Detections": str(num_detections),
            "Status": "ACTIVE" if yolo_results is not None else "INACTIVE",
            "Model": "YOLOv11n",
            "Tracking": "Enabled",
        }
        yolo_panel = DataPanel(width=600, height=200, title="OBJECT DETECTION").draw(yolo_data)
        canvas[info_y:info_y+200, margin+650:margin+1250] = yolo_panel
        
        return canvas
    
    def switch_tab(self, tab_index):
        """Switch active tab"""
        self.active_tab = tab_index
    
    def render(self, **kwargs):
        """Render current tab"""
        tab_bar = self.create_tab_bar(self.window_width)
        
        if self.active_tab == 0:
            # Central dashboard
            content = self.draw_central_tab(
                kwargs.get('ref_frame'),
                kwargs.get('curr_frame'),
                kwargs.get('mkpts_0', np.array([])),
                kwargs.get('mkpts_1', np.array([])),
                kwargs.get('yolo_results'),
                kwargs.get('dx', 0),
                kwargs.get('dy', 0),
                kwargs.get('dz_scale', 1.0),
                kwargs.get('confidence', 0),
                kwargs.get('guidance_source', 'none'),
                kwargs.get('yolo_active', False),
                kwargs.get('num_matches', 0),
                kwargs.get('max_matches', 100),
                kwargs.get('fps', 0.0),
                kwargs.get('stability', 0.0),
                kwargs.get('show_guidance', True),
                kwargs.get('flow_points'),
                kwargs.get('imu_data')
            )
        elif self.active_tab == 1:
            # Vision subsystem
            content = self.draw_vision_tab(
                kwargs.get('ref_frame'),
                kwargs.get('curr_frame'),
                kwargs.get('mkpts_0', np.array([])),
                kwargs.get('mkpts_1', np.array([])),
                kwargs.get('yolo_results'),
                kwargs.get('num_matches', 0),
                kwargs.get('max_matches', 100),
                kwargs.get('confidence', 0)
            )
        elif self.active_tab == 2:
            # IMU/Fusion subsystem
            from .imu_viz import draw_imu_raw_panel, draw_ekf_panel, draw_fusion_panel
            
            content_height = self.window_height - 60
            content = np.zeros((content_height, self.window_width, 3), dtype=np.uint8)
            content[:] = self.colors['bg_dark']
            
            cv2.putText(content, "IMU / FUSION SUBSYSTEM", (40, 45),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['highlight'], 2, cv2.LINE_AA)
            
            if kwargs.get('imu_data') is not None:
                panel_w = 550
                panel_h = 850
                margin = 40
                
                # IMU panel
                imu_panel = draw_imu_raw_panel(kwargs.get('imu_data'), panel_h=panel_h)
                imu_panel = cv2.resize(imu_panel, (panel_w, panel_h))
                content[80:80+panel_h, margin:margin+panel_w] = imu_panel
                
                # EKF panel
                ekf_panel = draw_ekf_panel(kwargs.get('imu_data'), panel_h=panel_h)
                ekf_panel = cv2.resize(ekf_panel, (panel_w, panel_h))
                content[80:80+panel_h, margin*2+panel_w:margin*2+panel_w*2] = ekf_panel
                
                # Fusion panel
                if kwargs.get('fusion_info') is not None:
                    fusion_panel = draw_fusion_panel(
                        kwargs.get('visual_dx', 0), kwargs.get('visual_dy', 0),
                        kwargs.get('visual_dz', 1.0), kwargs.get('visual_conf', 0),
                        kwargs.get('dx', 0), kwargs.get('dy', 0),
                        kwargs.get('dz_scale', 1.0), kwargs.get('confidence', 0),
                        kwargs.get('fusion_info'), panel_h=panel_h
                    )
                    fusion_panel = cv2.resize(fusion_panel, (panel_w, panel_h))
                    content[80:80+panel_h, margin*3+panel_w*2:margin*3+panel_w*3] = fusion_panel
            
            else:
                cv2.putText(content, "IMU NOT CONNECTED", (self.window_width // 2 - 150, content_height // 2),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text_secondary'], 1, cv2.LINE_AA)
        
        elif self.active_tab == 3:
            # Optical flow subsystem
            content_height = self.window_height - 60
            content = np.zeros((content_height, self.window_width, 3), dtype=np.uint8)
            content[:] = self.colors['bg_dark']
            
            cv2.putText(content, "OPTICAL FLOW SUBSYSTEM", (40, 45),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['highlight'], 2, cv2.LINE_AA)
            
            # Show current frame with flow vectors
            cam_w = 1200
            cam_h = 900
            margin = (self.window_width - cam_w) // 2
            
            curr_frame = kwargs.get('curr_frame')
            if curr_frame is not None:
                flow_img = cv2.resize(curr_frame, (cam_w, cam_h))
                
                # Draw flow points if available
                flow_points = kwargs.get('flow_points')
                if flow_points is not None and len(flow_points) > 0:
                    scale_x = cam_w / curr_frame.shape[1]
                    scale_y = cam_h / curr_frame.shape[0]
                    for pt in flow_points:
                        x, y = int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)
                        cv2.circle(flow_img, (x, y), 5, (100, 255, 255), -1)
                        cv2.circle(flow_img, (x, y), 7, (150, 255, 255), 1)
                
                # Header
                cv2.rectangle(flow_img, (0, 0), (cam_w, 50), (40, 60, 90), -1)
                num_points = len(flow_points) if flow_points is not None else 0
                cv2.putText(flow_img, f"OPTICAL FLOW - {num_points} Tracked Points", (20, 35),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text_primary'], 1, cv2.LINE_AA)
                
                content[80:80+cam_h, margin:margin+cam_w] = flow_img
                cv2.rectangle(content, (margin-2, 78), (margin+cam_w+2, 82+cam_h),
                             self.colors['bg_light'], 3)
        
        elif self.active_tab == 4:
            # Performance subsystem
            content_height = self.window_height - 60
            content = np.zeros((content_height, self.window_width, 3), dtype=np.uint8)
            content[:] = self.colors['bg_dark']
            
            cv2.putText(content, "PERFORMANCE METRICS", (40, 45),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['highlight'], 2, cv2.LINE_AA)
            
            performance_monitor = kwargs.get('performance_monitor')
            if performance_monitor is not None:
                graph_w = self.window_width - 160
                graph_h = content_height - 150
                
                try:
                    graphs = performance_monitor.draw_performance_panel(size=(graph_w, graph_h))
                    content[80:80+graph_h, 80:80+graph_w] = graphs
                    cv2.rectangle(content, (77, 77), (83+graph_w, 83+graph_h),
                                 self.colors['bg_light'], 3)
                except Exception as e:
                    cv2.putText(content, f"Error rendering graphs: {str(e)[:50]}", 
                               (100, content_height // 2),
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['error'], 1, cv2.LINE_AA)
        else:
            # Fallback
            content_height = self.window_height - 60
            content = np.zeros((content_height, self.window_width, 3), dtype=np.uint8)
            content[:] = self.colors['bg_dark']
        
        final = np.vstack([tab_bar, content])
        return final


# Export singleton
_modern_dashboard = ModernDashboard()

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
    """Wrapper for backwards compatibility - accepts positional arguments"""
    return _modern_dashboard.render(
        ref_frame=ref_frame,
        curr_frame=curr_frame,
        mkpts_0=mkpts_0,
        mkpts_1=mkpts_1,
        yolo_results=yolo_results,
        dx=dx,
        dy=dy,
        dz_scale=dz_scale,
        confidence=confidence,
        guidance_source=guidance_source,
        yolo_active=yolo_active,
        num_matches=num_matches,
        max_matches=max_matches,
        fps=fps,
        stability=stability,
        show_guidance=show_guidance,
        flow_points=flow_points,
        imu_data=imu_data,
        visual_dx=visual_dx,
        visual_dy=visual_dy,
        visual_dz=visual_dz,
        visual_conf=visual_conf,
        fusion_info=fusion_info,
        performance_monitor=performance_monitor,
    )

def switch_tab(tab_index):
    """Switch active tab"""
    _modern_dashboard.switch_tab(tab_index)


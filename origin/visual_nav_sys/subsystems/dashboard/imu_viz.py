"""
IMU and EKF Visualization Panels
Real-time display of Pixhawk IMU/EKF data and sensor fusion results
Ocean-inspired theme
"""

import cv2
import numpy as np

# Ocean-inspired color palette
COLORS = {
    'bg_dark': (45, 35, 20),
    'bg_medium': (55, 45, 30),
    'bg_light': (70, 60, 45),
    'accent_blue': (180, 120, 60),
    'accent_cyan': (200, 180, 80),
    'text_primary': (255, 255, 255),
    'text_secondary': (200, 200, 200),
    'text_dim': (150, 150, 150),
    'success': (100, 200, 100),
    'warning': (100, 200, 255),
    'error': (100, 100, 255),
}


def draw_imu_raw_panel(imu_data, panel_h=200):
    """
    Create raw IMU data visualization panel (attitude and rates)
    """
    panel_w = 350
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    
    # Gradient background
    for y in range(panel_h):
        factor = y / panel_h
        color = tuple(int(COLORS['bg_dark'][i] + 
                        (COLORS['bg_medium'][i] - COLORS['bg_dark'][i]) * factor * 0.5) 
                     for i in range(3))
        panel[y, :] = color
    
    # Add border
    cv2.rectangle(panel, (0, 0), (panel_w-1, panel_h-1), COLORS['bg_light'], 2)
    
    attitude = imu_data.get_attitude()
    
    # Title with better styling
    cv2.putText(panel, "RAW IMU DATA", (15, 30), 
                cv2.FONT_HERSHEY_DUPLEX, 0.65, COLORS['accent_cyan'], 1, cv2.LINE_AA)
    
    y_pos = 65
    
    if attitude['connected']:
        # Connection status indicator
        cv2.circle(panel, (panel_w - 25, 22), 7, COLORS['success'], -1)
        cv2.circle(panel, (panel_w - 25, 22), 9, (150, 255, 150), 2)
        cv2.putText(panel, "IMU", (panel_w - 55, 26), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.35, COLORS['success'], 1, cv2.LINE_AA)
        
        # Section: Attitude
        cv2.putText(panel, "ATTITUDE:", (15, y_pos), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS['text_dim'], 1, cv2.LINE_AA)
        y_pos += 28
        
        cv2.putText(panel, f"Roll:  {np.degrees(attitude['roll']):+7.2f}°", 
                    (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['accent_blue'], 1, cv2.LINE_AA)
        y_pos += 25
        cv2.putText(panel, f"Pitch: {np.degrees(attitude['pitch']):+7.2f}°", 
                    (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['accent_blue'], 1, cv2.LINE_AA)
        y_pos += 25
        cv2.putText(panel, f"Yaw:   {np.degrees(attitude['yaw']):+7.2f}°", 
                    (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['accent_blue'], 1, cv2.LINE_AA)
        y_pos += 35
        
        # Section: Angular Rates
        cv2.putText(panel, "ANGULAR RATES:", (15, y_pos), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS['text_dim'], 1, cv2.LINE_AA)
        y_pos += 28
        
        cv2.putText(panel, f"Roll:  {np.degrees(attitude['rollspeed']):+6.2f}°/s", 
                    (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['warning'], 1, cv2.LINE_AA)
        y_pos += 25
        cv2.putText(panel, f"Pitch: {np.degrees(attitude['pitchspeed']):+6.2f}°/s", 
                    (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['warning'], 1, cv2.LINE_AA)
        y_pos += 25
        cv2.putText(panel, f"Yaw:   {np.degrees(attitude['yawspeed']):+6.2f}°/s", 
                    (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['warning'], 1, cv2.LINE_AA)
        
        # Data age warning
        if attitude['age'] > 0.5:
            cv2.putText(panel, "⚠ STALE", (panel_w - 75, panel_h - 15), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, COLORS['error'], 1, cv2.LINE_AA)
    else:
        # Not connected
        cv2.circle(panel, (panel_w - 25, 22), 7, COLORS['text_dim'], -1)
        cv2.circle(panel, (panel_w - 25, 22), 9, (100, 100, 100), 2)
        cv2.putText(panel, "NOT CONNECTED", (15, y_pos + 10), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, COLORS['text_dim'], 1, cv2.LINE_AA)
        cv2.putText(panel, "Check Pixhawk connection", (15, y_pos + 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, COLORS['text_dim'], 1, cv2.LINE_AA)
    
    return panel


def draw_ekf_panel(imu_data, panel_h=200):
    """
    Create EKF filtered data visualization panel
    """
    panel_w = 350
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    
    # Gradient background
    for y in range(panel_h):
        factor = y / panel_h
        color = tuple(int(COLORS['bg_dark'][i] + 
                        (COLORS['bg_medium'][i] - COLORS['bg_dark'][i]) * factor * 0.5) 
                     for i in range(3))
        panel[y, :] = color
    
    # Add border
    cv2.rectangle(panel, (0, 0), (panel_w-1, panel_h-1), COLORS['bg_light'], 2)
    
    ekf = imu_data.get_ekf_data()
    
    # Title with better styling
    cv2.putText(panel, "EKF FILTERED", (15, 30), 
                cv2.FONT_HERSHEY_DUPLEX, 0.65, COLORS['success'], 1, cv2.LINE_AA)
    
    y_pos = 65
    
    # Very lenient age check - show if we ever received EKF data (even if old)
    has_ekf_data = ekf['age'] < 60.0  # Show even old data with warning
    
    if has_ekf_data:
        # EKF status indicator
        status_color = COLORS['success'] if ekf['healthy'] else COLORS['warning']
        cv2.circle(panel, (panel_w - 25, 22), 7, status_color, -1)
        cv2.circle(panel, (panel_w - 25, 22), 9, (150, 255, 150) if ekf['healthy'] else (150, 200, 255), 2)
        status_text = "EKF" if ekf['healthy'] else "⚠"
        cv2.putText(panel, status_text, (panel_w - 55, 26), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.35, status_color, 1, cv2.LINE_AA)
        
        # Section: Position (NED frame)
        cv2.putText(panel, "POSITION (NED):", (15, y_pos), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS['text_dim'], 1, cv2.LINE_AA)
        y_pos += 28
        
        cv2.putText(panel, f"North: {ekf['pos_x']:+7.2f} m", 
                    (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['accent_cyan'], 1, cv2.LINE_AA)
        y_pos += 25
        cv2.putText(panel, f"East:  {ekf['pos_y']:+7.2f} m", 
                    (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['accent_cyan'], 1, cv2.LINE_AA)
        y_pos += 25
        cv2.putText(panel, f"Down:  {ekf['pos_z']:+7.2f} m", 
                    (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['accent_cyan'], 1, cv2.LINE_AA)
        y_pos += 35
        
        # Section: Velocity (NED frame)
        cv2.putText(panel, "VELOCITY (NED):", (15, y_pos), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS['text_dim'], 1, cv2.LINE_AA)
        y_pos += 28
        
        cv2.putText(panel, f"North: {ekf['vel_x']:+6.2f} m/s", 
                    (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['success'], 1, cv2.LINE_AA)
        y_pos += 25
        cv2.putText(panel, f"East:  {ekf['vel_y']:+6.2f} m/s", 
                    (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['success'], 1, cv2.LINE_AA)
        y_pos += 25
        cv2.putText(panel, f"Down:  {ekf['vel_z']:+6.2f} m/s", 
                    (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['success'], 1, cv2.LINE_AA)
        
        # Velocity magnitude
        vel_mag = np.sqrt(ekf['vel_x']**2 + ekf['vel_y']**2 + ekf['vel_z']**2)
        cv2.putText(panel, f"|V|: {vel_mag:.2f} m/s", 
                    (15, panel_h - 15), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, COLORS['warning'], 1, cv2.LINE_AA)
        
        # Data age indicator
        age_text = f"Age: {ekf['age']:.1f}s"
        age_color = COLORS['success'] if ekf['age'] < 0.5 else (COLORS['warning'] if ekf['age'] < 2.0 else COLORS['error'])
        cv2.putText(panel, age_text, (panel_w - 95, panel_h - 15), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.35, age_color, 1, cv2.LINE_AA)
    else:
        # No EKF data - provide helpful information
        cv2.circle(panel, (panel_w - 25, 22), 7, COLORS['text_dim'], -1)
        cv2.circle(panel, (panel_w - 25, 22), 9, (100, 100, 100), 2)
        
        cv2.putText(panel, "NO EKF DATA", (15, y_pos + 10), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, COLORS['text_dim'], 1, cv2.LINE_AA)
        y_pos += 45
        
        cv2.putText(panel, "EKF requires:", (15, y_pos), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, COLORS['text_dim'], 1, cv2.LINE_AA)
        y_pos += 25
        cv2.putText(panel, "- IMU connected", (20, y_pos), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.35, COLORS['text_dim'], 1, cv2.LINE_AA)
        y_pos += 22
        cv2.putText(panel, "- GPS or vision", (20, y_pos), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.35, COLORS['text_dim'], 1, cv2.LINE_AA)
        y_pos += 22
        cv2.putText(panel, "- Sensor fusion active", (20, y_pos), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.35, COLORS['text_dim'], 1, cv2.LINE_AA)
        
        # Show age to help debug
        cv2.putText(panel, f"Last update: {ekf['age']:.1f}s ago", (15, panel_h - 15), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.35, COLORS['text_dim'], 1, cv2.LINE_AA)
    
    return panel


def draw_fusion_panel(visual_dx, visual_dy, visual_dz, visual_conf,
                       fused_dx, fused_dy, fused_dz, fused_conf, 
                       fusion_info, panel_h=200):
    """
    Create sensor fusion result panel showing before/after
    """
    panel_w = 350
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    
    # Gradient background
    for y in range(panel_h):
        factor = y / panel_h
        color = tuple(int(COLORS['bg_dark'][i] + 
                        (COLORS['bg_medium'][i] - COLORS['bg_dark'][i]) * factor * 0.5) 
                     for i in range(3))
        panel[y, :] = color
    
    # Add border
    cv2.rectangle(panel, (0, 0), (panel_w-1, panel_h-1), COLORS['bg_light'], 2)
    
    # Title with better styling
    cv2.putText(panel, "SENSOR FUSION", (15, 30), 
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 180, 255), 1, cv2.LINE_AA)
    
    y_pos = 65
    
    # Status indicators with better styling
    status_x = panel_w - 90
    if fusion_info['imu_used']:
        cv2.circle(panel, (status_x, 22), 5, COLORS['success'], -1)
        cv2.circle(panel, (status_x, 22), 7, (150, 255, 150), 1)
        cv2.putText(panel, "IMU", (status_x + 10, 26), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.35, COLORS['success'], 1, cv2.LINE_AA)
    
    if fusion_info['ekf_used']:
        cv2.circle(panel, (status_x + 45, 22), 5, COLORS['accent_cyan'], -1)
        cv2.circle(panel, (status_x + 45, 22), 7, (150, 255, 255), 1)
        cv2.putText(panel, "EKF", (status_x + 55, 26), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.35, COLORS['accent_cyan'], 1, cv2.LINE_AA)
    
    # Section: Visual Input
    cv2.putText(panel, "VISUAL INPUT:", (15, y_pos), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS['text_dim'], 1, cv2.LINE_AA)
    y_pos += 28
    
    cv2.putText(panel, f"DX: {visual_dx:+7.1f} px", 
                (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['text_secondary'], 1, cv2.LINE_AA)
    y_pos += 25
    cv2.putText(panel, f"DY: {visual_dy:+7.1f} px", 
                (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['text_secondary'], 1, cv2.LINE_AA)
    y_pos += 25
    cv2.putText(panel, f"DZ: {visual_dz:7.3f}x", 
                (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['text_secondary'], 1, cv2.LINE_AA)
    cv2.putText(panel, f"{visual_conf:.0f}%", 
                (panel_w - 55, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.4, COLORS['text_dim'], 1, cv2.LINE_AA)
    y_pos += 35
    
    # Section: Fused Output
    cv2.putText(panel, "FUSED OUTPUT:", (15, y_pos), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS['text_dim'], 1, cv2.LINE_AA)
    y_pos += 28
    
    cv2.putText(panel, f"DX: {fused_dx:+7.1f} px", 
                (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['success'], 1, cv2.LINE_AA)
    # Show delta
    delta_dx = fused_dx - visual_dx
    if abs(delta_dx) > 0.5:
        cv2.putText(panel, f"({delta_dx:+.1f})", 
                    (panel_w - 75, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.35, COLORS['warning'], 1, cv2.LINE_AA)
    y_pos += 25
    
    cv2.putText(panel, f"DY: {fused_dy:+7.1f} px", 
                (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['success'], 1, cv2.LINE_AA)
    # Show delta
    delta_dy = fused_dy - visual_dy
    if abs(delta_dy) > 0.5:
        cv2.putText(panel, f"({delta_dy:+.1f})", 
                    (panel_w - 75, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.35, COLORS['warning'], 1, cv2.LINE_AA)
    y_pos += 25
    
    cv2.putText(panel, f"DZ: {fused_dz:7.3f}x", 
                (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS['success'], 1, cv2.LINE_AA)
    cv2.putText(panel, f"{fused_conf:.0f}%", 
                (panel_w - 55, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.4, COLORS['success'], 1, cv2.LINE_AA)
    
    # Bottom info
    y_bottom = panel_h - 15
    
    # EKF velocity magnitude if available
    if fusion_info['ekf_used']:
        cv2.putText(panel, f"EKF: {fusion_info['ekf_vel_mag']:.2f} m/s", 
                    (15, y_bottom), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, COLORS['accent_cyan'], 1, cv2.LINE_AA)
    
    # Correction status
    if fusion_info['correction_applied']:
        cv2.putText(panel, "✓ FUSED", (panel_w - 75, y_bottom), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, COLORS['success'], 1, cv2.LINE_AA)
    
    return panel

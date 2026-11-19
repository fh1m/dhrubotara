"""
Artificial Horizon Visualization
Aviation-style attitude indicator
"""

import cv2
import numpy as np


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
    cv2.circle(img, center, 10, (255, 255, 255), -1)
    cv2.circle(img, center, 12, (200, 200, 200), 2)
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
    
    # Add text displays (positioned to avoid overlap)
    box_bg = (25, 20, 15)
    box_border = (80, 65, 40)
    
    # Roll (bottom left)
    cv2.rectangle(img, (8, size[1]-35), (95, size[1]-8), box_bg, -1)
    cv2.rectangle(img, (8, size[1]-35), (95, size[1]-8), box_border, 1)
    cv2.putText(img, f"R:{np.degrees(roll):+5.1f}", (13, size[1]-16), 
               cv2.FONT_HERSHEY_DUPLEX, 0.35, (200, 230, 255), 1, cv2.LINE_AA)
    
    # Pitch (bottom center)
    cv2.rectangle(img, (size[0]//2-45, size[1]-35), (size[0]//2+45, size[1]-8), box_bg, -1)
    cv2.rectangle(img, (size[0]//2-45, size[1]-35), (size[0]//2+45, size[1]-8), box_border, 1)
    cv2.putText(img, f"P:{np.degrees(pitch):+5.1f}", (size[0]//2-40, size[1]-16), 
               cv2.FONT_HERSHEY_DUPLEX, 0.35, (200, 230, 255), 1, cv2.LINE_AA)
    
    # Yaw/Heading (bottom right)
    cv2.rectangle(img, (size[0]-102, size[1]-35), (size[0]-8, size[1]-8), box_bg, -1)
    cv2.rectangle(img, (size[0]-102, size[1]-35), (size[0]-8, size[1]-8), box_border, 1)
    heading = (np.degrees(yaw) + 360) % 360
    cv2.putText(img, f"H:{heading:5.1f}", (size[0]-97, size[1]-16), 
               cv2.FONT_HERSHEY_DUPLEX, 0.35, (200, 230, 255), 1, cv2.LINE_AA)
    
    return img

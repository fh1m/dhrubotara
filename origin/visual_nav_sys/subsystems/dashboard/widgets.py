"""
Custom GUI Widgets for Visual Navigation System
Inspired by reference UI design with circular gauges and modern dark theme
"""

import cv2
import numpy as np


class CircularGauge:
    """Circular gauge widget like the reference image"""
    
    def __init__(self, size=200, max_value=2000, label="Speed", units="m/s"):
        self.size = size
        self.max_value = max_value
        self.label = label
        self.units = units
        self.center = (size // 2, size // 2)
        self.radius = int(size * 0.4)
        
    def draw(self, value):
        """Draw the circular gauge with blue theme"""
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        img[:] = (35, 25, 15)  # Dark blue background
        
        # Draw outer rings (blue theme)
        cv2.circle(img, self.center, self.radius + 10, (120, 90, 50), 3)
        cv2.circle(img, self.center, self.radius + 5, (80, 60, 35), 1)
        
        # Draw gauge arc (270 degrees, -135 to +135)
        start_angle = -135
        end_angle = 135
        total_angle = end_angle - start_angle
        
        # Background arc
        cv2.ellipse(img, self.center, (self.radius, self.radius), 
                   0, start_angle, end_angle, (30, 40, 50), 8)
        
        # Value arc
        value_angle = start_angle + (value / self.max_value) * total_angle
        value_angle = min(value_angle, end_angle)
        
        # Color gradient based on value
        if value < self.max_value * 0.5:
            color = (100, 200, 100)  # Green
        elif value < self.max_value * 0.75:
            color = (100, 200, 255)  # Yellow
        else:
            color = (100, 100, 255)  # Red
        
        cv2.ellipse(img, self.center, (self.radius, self.radius), 
                   0, start_angle, value_angle, color, 8)
        
        # Draw tick marks
        for i in range(0, 11):
            angle = start_angle + (i / 10) * total_angle
            angle_rad = np.radians(angle)
            
            if i % 5 == 0:
                r1 = self.radius - 15
                r2 = self.radius
                thickness = 2
            else:
                r1 = self.radius - 10
                r2 = self.radius
                thickness = 1
            
            x1 = int(self.center[0] + r1 * np.cos(angle_rad))
            y1 = int(self.center[1] + r1 * np.sin(angle_rad))
            x2 = int(self.center[0] + r2 * np.cos(angle_rad))
            y2 = int(self.center[1] + r2 * np.sin(angle_rad))
            
            cv2.line(img, (x1, y1), (x2, y2), (150, 180, 200), thickness)
        
        # Draw value text (large in center)
        value_text = f"{int(value)}"
        text_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
        text_x = self.center[0] - text_size[0] // 2
        text_y = self.center[1] + text_size[1] // 2
        cv2.putText(img, value_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, (200, 230, 255), 3, cv2.LINE_AA)
        
        # Draw units
        unit_size = cv2.getTextSize(self.units, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
        unit_x = self.center[0] - unit_size[0] // 2
        cv2.putText(img, self.units, (unit_x, self.center[1] + 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (150, 170, 190), 1, cv2.LINE_AA)
        
        # Draw label at top
        label_size = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
        label_x = self.center[0] - label_size[0] // 2
        cv2.putText(img, self.label, (label_x, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 200, 220), 1, cv2.LINE_AA)
        
        return img


class DataPanel:
    """Data panel widget with multiple fields"""
    
    def __init__(self, width=300, height=200, title="Data Panel"):
        self.width = width
        self.height = height
        self.title = title
        
    def draw(self, data_dict):
        """Draw data panel with dictionary of label: value pairs"""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (45, 35, 20)  # Dark blue background
        
        # Border
        cv2.rectangle(img, (0, 0), (self.width-1, self.height-1), (80, 65, 40), 2)
        
        # Title bar (blue theme)
        cv2.rectangle(img, (0, 0), (self.width, 30), (60, 50, 30), -1)
        cv2.putText(img, self.title, (10, 20),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (240, 230, 210), 1, cv2.LINE_AA)
        
        # Data fields (more compact)
        y = 50
        for label, value in data_dict.items():
            # Label
            cv2.putText(img, f"{label}:", (15, y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, (190, 180, 160), 1, cv2.LINE_AA)
            
            # Value (aligned to right)
            value_text = str(value)
            text_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)[0]
            value_x = self.width - text_size[0] - 15
            cv2.putText(img, value_text, (value_x, y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.45, (150, 200, 255), 1, cv2.LINE_AA)
            
            y += 22
        
        return img


class Heading3D:
    """3D coordinate system visualization like reference image"""
    
    def __init__(self, size=400):
        self.size = size
        self.center = (size // 2, size // 2)
        
    def draw(self, roll=0, pitch=0, yaw=0, dx=0, dy=0, dz=0):
        """Draw 3D coordinate system with enhanced depth and perspective"""
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        img[:] = (35, 25, 15)  # Dark blue background
        
        # Draw 3D floor grid plane for depth perception
        grid_spacing = 25
        floor_z = -80  # Floor at fixed Z depth
        
        # Draw grid lines on floor (X-Z plane)
        for i in range(-8, 9):
            # Grid lines along X direction
            pts = []
            for j in range(-8, 9):
                x3d = i * grid_spacing
                y3d = 0  # Floor level
                z3d = j * grid_spacing + floor_z
                
                # Perspective projection
                fov = 300
                distance = 350
                if z3d + distance > 0:
                    scale = fov / (z3d + distance)
                    x2d = int(self.center[0] + x3d * scale)
                    y2d = int(self.center[1] - y3d * scale + 80)  # Offset down
                    
                    if 0 <= x2d < self.size and 0 <= y2d < self.size:
                        pts.append((x2d, y2d))
            
            # Draw lines with fading
            if len(pts) > 1:
                for k in range(len(pts) - 1):
                    # Calculate brightness based on distance
                    avg_y = (pts[k][1] + pts[k+1][1]) / 2
                    brightness = int(60 * (1 - (avg_y - self.center[1]) / self.size))
                    brightness = max(20, min(80, brightness))
                    cv2.line(img, pts[k], pts[k+1], (brightness, brightness-10, brightness-20), 1)
        
        # Grid lines along Z direction
        for j in range(-8, 9):
            pts = []
            for i in range(-8, 9):
                x3d = i * grid_spacing
                y3d = 0
                z3d = j * grid_spacing + floor_z
                
                fov = 300
                distance = 350
                if z3d + distance > 0:
                    scale = fov / (z3d + distance)
                    x2d = int(self.center[0] + x3d * scale)
                    y2d = int(self.center[1] - y3d * scale + 80)
                    
                    if 0 <= x2d < self.size and 0 <= y2d < self.size:
                        pts.append((x2d, y2d))
            
            if len(pts) > 1:
                for k in range(len(pts) - 1):
                    avg_y = (pts[k][1] + pts[k+1][1]) / 2
                    brightness = int(60 * (1 - (avg_y - self.center[1]) / self.size))
                    brightness = max(20, min(80, brightness))
                    cv2.line(img, pts[k], pts[k+1], (brightness, brightness-10, brightness-20), 1)
        
        # Calculate axis endpoints with rotation (longer for visibility)
        axis_length = 130
        
        # Rotation matrices
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        
        # Create axes origins and endpoints
        # X axis (RED) - points right
        x_end = np.array([axis_length, 0, 0])
        # Y axis (GREEN) - points up  
        y_end = np.array([0, -axis_length, 0])  # Negative because screen Y is inverted
        # Z axis (BLUE) - points forward/toward viewer
        z_end = np.array([0, 0, axis_length])
        
        # Apply rotations (yaw -> pitch -> roll)
        def rotate_point(point):
            # Yaw rotation (around Z)
            p = np.array([
                point[0] * cos_y - point[1] * sin_y,
                point[0] * sin_y + point[1] * cos_y,
                point[2]
            ])
            # Pitch rotation (around Y)
            p = np.array([
                p[0] * cos_p + p[2] * sin_p,
                p[1],
                -p[0] * sin_p + p[2] * cos_p
            ])
            # Roll rotation (around X)
            p = np.array([
                p[0],
                p[1] * cos_r - p[2] * sin_r,
                p[1] * sin_r + p[2] * cos_r
            ])
            return p
        
        x_end = rotate_point(x_end)
        y_end = rotate_point(y_end)
        z_end = rotate_point(z_end)
        
        # Enhanced perspective projection for strong 3D effect
        def project_2d(point, add_perspective=True):
            if add_perspective:
                # Strong perspective projection
                fov = 400  # Tighter FOV for more dramatic perspective
                distance = 300  # Closer camera for more depth
                z_val = point[2] + distance
                if z_val > 10:  # Prevent division by zero
                    scale = fov / z_val
                else:
                    scale = fov / 10
                
                return (
                    int(self.center[0] + point[0] * scale * 1.8),
                    int(self.center[1] + point[1] * scale * 1.8)  # Don't invert Y here
                )
            else:
                return (
                    int(self.center[0] + point[0]),
                    int(self.center[1] + point[1])
                )
        
        # Helper to get depth-based thickness
        def get_thickness(z_depth):
            # Closer = thicker
            base_thickness = 2 + int((z_depth + distance) / 80)
            return max(2, min(8, base_thickness))
        
        origin_2d = self.center
        
        # Project axis endpoints
        x_2d = project_2d(x_end)
        y_2d = project_2d(y_end)
        z_2d = project_2d(z_end)
        
        # Also project negative axes for full 3D effect
        x_neg = project_2d(rotate_point(np.array([-axis_length * 0.3, 0, 0])))
        y_neg = project_2d(rotate_point(np.array([0, axis_length * 0.3, 0])))
        z_neg = project_2d(rotate_point(np.array([0, 0, -axis_length * 0.3])))
        
        # Sort axes by depth (Z coordinate) for proper rendering order
        axes = [
            ('Z', z_end, z_2d, z_neg, (255, 140, 100), (255, 180, 140)),
            ('X', x_end, x_2d, x_neg, (140, 140, 255), (180, 180, 255)),
            ('Y', y_end, y_2d, y_neg, (140, 255, 140), (180, 255, 180)),
        ]
        # Sort by z-coordinate (further back drawn first)
        axes.sort(key=lambda a: a[1][2])
        
        # Draw axes in order (back to front) with depth cues
        for label, vec_3d, point_2d, point_neg, color_dark, color_light in axes:
            thickness = get_thickness(vec_3d[2])
            
            # Draw negative part (behind origin) - darker
            neg_color = tuple(int(c * 0.4) for c in color_dark)
            cv2.line(img, origin_2d, point_neg, neg_color, max(1, thickness-1))
            
            # Draw shadow for positive part (in front of origin)
            shadow_offset = 3
            cv2.line(img, 
                    (origin_2d[0] + shadow_offset, origin_2d[1] + shadow_offset),
                    (point_2d[0] + shadow_offset, point_2d[1] + shadow_offset),
                    (20, 15, 10), thickness)
            
            # Draw axis line with gradient (darker at base, brighter at tip)
            cv2.line(img, origin_2d, point_2d, color_dark, thickness + 2)
            cv2.line(img, origin_2d, point_2d, color_light, thickness)
            
            # Draw arrow head
            cv2.arrowedLine(img, origin_2d, point_2d, 
                           tuple(int(c * 1.2) for c in color_light), 
                           thickness - 1, tipLength=0.12)
            
            # End cap with glow effect
            cv2.circle(img, point_2d, thickness + 3, tuple(int(c * 0.6) for c in color_light), -1)
            cv2.circle(img, point_2d, thickness + 1, color_light, -1)
            cv2.circle(img, point_2d, thickness + 4, (240, 230, 210), 1)
            
            # Label with 3D effect
            label_offset = 18
            if vec_3d[2] > 0:  # Coming toward viewer
                label_pos = (point_2d[0] + label_offset, point_2d[1] - label_offset // 2)
            else:  # Going away
                label_pos = (point_2d[0] + label_offset, point_2d[1] + label_offset // 2)
            
            # Label background with depth
            cv2.circle(img, label_pos, 15, (25, 18, 12), -1)
            cv2.circle(img, label_pos, 13, (45, 35, 20), -1)
            cv2.putText(img, label, (label_pos[0] - 7, label_pos[1] + 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, color_light, 2, cv2.LINE_AA)
        
        # Draw movement vector if significant (enhanced 3D look)
        if abs(dx) > 5 or abs(dy) > 5 or abs(dz) > 5:
            move_vec = rotate_point(np.array([dx, -dy, dz]) * 1.2)
            move_2d = project_2d(move_vec)
            move_thickness = get_thickness(move_vec[2])
            
            # Cyan movement vector with glow
            cv2.line(img, origin_2d, move_2d, (180, 160, 70), move_thickness + 3)
            cv2.arrowedLine(img, origin_2d, move_2d, (220, 200, 100), move_thickness + 1, tipLength=0.18)
            cv2.circle(img, move_2d, move_thickness + 2, (240, 220, 120), -1)
            cv2.circle(img, move_2d, move_thickness + 4, (200, 180, 80), 2)
        
        # Origin point (3D sphere effect with shading)
        # Base shadow
        cv2.circle(img, (origin_2d[0] + 2, origin_2d[1] + 2), 14, (15, 12, 8), -1)
        # Main sphere with gradient effect
        for r in range(12, 0, -1):
            intensity = int(180 + (240 - 180) * (12 - r) / 12)
            cv2.circle(img, origin_2d, r, (intensity, intensity-10, intensity-20), -1)
        # Highlight
        cv2.circle(img, (origin_2d[0] - 3, origin_2d[1] - 3), 4, (255, 250, 245), -1)
        # Outline
        cv2.circle(img, origin_2d, 13, (180, 170, 150), 2)
        
        # Add title at top
        title = "3D COORDINATE SYSTEM"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
        title_x = (self.size - title_size[0]) // 2
        cv2.putText(img, title, (title_x, 25),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (150, 180, 220), 1, cv2.LINE_AA)
        
        # Add orientation info at bottom
        info_text = f"Roll:{np.degrees(roll):+5.1f} Pitch:{np.degrees(pitch):+5.1f} Yaw:{np.degrees(yaw):+5.1f}"
        info_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_DUPLEX, 0.35, 1)[0]
        info_x = (self.size - info_size[0]) // 2
        cv2.putText(img, info_text, (info_x, self.size - 10),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, (120, 140, 160), 1, cv2.LINE_AA)
        
        return img


class StatusIndicator:
    """Status indicator widget with icon and text"""
    
    def __init__(self, width=150, height=40):
        self.width = width
        self.height = height
        
    def draw(self, label, value, status="normal"):
        """Draw status indicator
        status: "normal", "warning", "error", "success"
        """
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (25, 30, 40)
        
        # Border
        if status == "success":
            border_color = (100, 200, 100)
        elif status == "warning":
            border_color = (100, 200, 255)
        elif status == "error":
            border_color = (100, 100, 255)
        else:
            border_color = (80, 100, 120)
        
        cv2.rectangle(img, (0, 0), (self.width-1, self.height-1), border_color, 2)
        
        # Status indicator circle
        cv2.circle(img, (15, self.height // 2), 6, border_color, -1)
        
        # Label
        cv2.putText(img, label, (28, self.height // 2 - 5),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (150, 170, 190), 1, cv2.LINE_AA)
        
        # Value
        value_text = str(value)
        text_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
        value_x = self.width - text_size[0] - 10
        cv2.putText(img, value_text, (value_x, self.height // 2 + 8),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 220, 240), 1, cv2.LINE_AA)
        
        return img


"""
3D Guidance Visualization
Vector-based 3D guidance display with coordinate axes
Ocean-inspired theme
"""

import cv2
import numpy as np

# Ocean-inspired color palette
COLORS = {
    'bg_dark': (45, 35, 20),
    'bg_medium': (55, 45, 30),
    'accent_blue': (180, 120, 60),
    'accent_cyan': (200, 180, 80),
    'text_primary': (255, 255, 255),
}


def create_slick_3d_guidance(
    dx, dy, dz_scale, confidence, guidance_source, size=(600, 600), threshold=10
):
    """
    Create clean 3D guidance visualization focused on vectors
    """
    viz = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Ocean gradient background
    for i in range(size[1]):
        factor = i / size[1]
        color = tuple(int(COLORS['bg_dark'][j] + 
                        (COLORS['bg_medium'][j] - COLORS['bg_dark'][j]) * factor * 0.7) 
                     for j in range(3))
        viz[i, :] = color
    
    # Add subtle texture
    texture = np.random.randint(-4, 4, (size[1], size[0], 3), dtype=np.int16)
    viz = np.clip(viz.astype(np.int16) + texture, 0, 255).astype(np.uint8)

    center_x, center_y = size[0] // 2, size[1] // 2

    # Title with better styling
    title = "3D GUIDANCE"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0]
    title_x = (size[0] - title_size[0]) // 2
    cv2.putText(
        viz,
        title,
        (title_x, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        COLORS['accent_cyan'],
        2,
        cv2.LINE_AA
    )

    axis_length = 120
    origin = (center_x, center_y)

    # Grid with ocean colors
    grid_color = (80, 70, 50)
    grid_spacing = 40
    for i in range(-3, 4):
        offset = i * grid_spacing
        alpha = 0.7 if i == 0 else 0.4
        thickness = 2 if i == 0 else 1
        cv2.line(
            viz,
            (center_x + offset, center_y - grid_spacing * 3),
            (center_x + offset, center_y + grid_spacing * 3),
            grid_color,
            thickness,
        )
        cv2.line(
            viz,
            (center_x - grid_spacing * 3, center_y + offset),
            (center_x + grid_spacing * 3, center_y + offset),
            grid_color,
            thickness,
        )

    # X-axis (RIGHT) - Ocean Red/Coral
    x_color = (100, 120, 255)  # Coral red
    x_end = (center_x + axis_length, center_y)
    cv2.line(viz, origin, x_end, (80, 100, 200), 5)
    cv2.arrowedLine(viz, origin, x_end, x_color, 4, tipLength=0.15)
    cv2.circle(viz, x_end, 6, x_color, -1)
    cv2.circle(viz, x_end, 9, (150, 170, 255), 2)
    cv2.putText(
        viz,
        "X",
        (x_end[0] + 15, x_end[1] + 8),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        x_color,
        2,
        cv2.LINE_AA
    )

    # Y-axis (DOWN) - Ocean Green/Aqua
    y_color = (120, 255, 120)  # Aqua green
    y_end = (center_x, center_y + axis_length)
    cv2.line(viz, origin, y_end, (100, 200, 100), 5)
    cv2.arrowedLine(viz, origin, y_end, y_color, 4, tipLength=0.15)
    cv2.circle(viz, y_end, 6, y_color, -1)
    cv2.circle(viz, y_end, 9, (170, 255, 170), 2)
    cv2.putText(
        viz,
        "Y",
        (y_end[0] + 15, y_end[1] + 5),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        y_color,
        2,
        cv2.LINE_AA
    )

    # Z-axis (DEPTH) - Ocean Amber
    z_color = (120, 200, 255)  # Amber/gold
    z_end = (center_x - int(axis_length * 0.7), center_y - int(axis_length * 0.7))
    cv2.line(viz, origin, z_end, (100, 160, 200), 5)
    cv2.arrowedLine(viz, origin, z_end, z_color, 4, tipLength=0.15)
    cv2.circle(viz, z_end, 6, z_color, -1)
    cv2.circle(viz, z_end, 9, (170, 230, 255), 2)
    cv2.putText(
        viz,
        "Z",
        (z_end[0] - 25, z_end[1] - 10),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        z_color,
        2,
        cv2.LINE_AA
    )

    # Movement Vectors (Ocean cyan theme)
    arrow_scale = 0.35
    movement_color = (200, 200, 100)  # Ocean cyan

    if abs(dx) > threshold:
        x_move = int(np.clip(dx * arrow_scale, -140, 140))
        move_end_x = (center_x + x_move, center_y)
        cv2.arrowedLine(viz, origin, move_end_x, (160, 160, 80), 8, tipLength=0.25)
        cv2.arrowedLine(viz, origin, move_end_x, movement_color, 5, tipLength=0.25)
        cv2.circle(viz, move_end_x, 12, (180, 180, 90), -1)
        cv2.circle(viz, move_end_x, 16, movement_color, 3)

    if abs(dy) > threshold:
        y_move = int(np.clip(dy * arrow_scale, -140, 140))
        move_end_y = (center_x, center_y + y_move)
        cv2.arrowedLine(viz, origin, move_end_y, (160, 160, 80), 8, tipLength=0.25)
        cv2.arrowedLine(viz, origin, move_end_y, movement_color, 5, tipLength=0.25)
        cv2.circle(viz, move_end_y, 12, (180, 180, 90), -1)
        cv2.circle(viz, move_end_y, 16, movement_color, 3)

    if abs(dz_scale - 1.0) > 0.1:
        z_move = int(np.clip((dz_scale - 1.0) * 140, -100, 100))
        move_end_z = (center_x - int(z_move * 0.7), center_y - int(z_move * 0.7))
        cv2.arrowedLine(viz, origin, move_end_z, (160, 160, 80), 8, tipLength=0.25)
        cv2.arrowedLine(viz, origin, move_end_z, movement_color, 5, tipLength=0.25)
        cv2.circle(viz, move_end_z, 12, (180, 180, 90), -1)
        cv2.circle(viz, move_end_z, 16, movement_color, 3)

    # Camera origin (ocean-styled)
    cv2.circle(viz, origin, 15, (60, 50, 35), -1)
    cv2.circle(viz, origin, 11, COLORS['text_primary'], -1)
    cv2.circle(viz, origin, 14, COLORS['accent_cyan'], 3)
    
    # Add info panel
    info_y = size[1] - 60
    cv2.rectangle(viz, (10, info_y), (size[0] - 10, size[1] - 10), (55, 45, 30), -1)
    cv2.rectangle(viz, (10, info_y), (size[0] - 10, size[1] - 10), (80, 70, 50), 2)
    
    source_text = f"Source: {guidance_source.upper()}"
    conf_text = f"Confidence: {confidence:.0f}%"
    
    cv2.putText(viz, source_text, (20, info_y + 20), 
               cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS['text_primary'], 1, cv2.LINE_AA)
    cv2.putText(viz, conf_text, (20, info_y + 40), 
               cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS['accent_cyan'], 1, cv2.LINE_AA)

    return viz


def draw_optical_flow_vectors(image, flow_points, color=(255, 255, 0)):
    """Draw optical flow tracked points on image"""
    if flow_points is not None and len(flow_points) > 0:
        for pt in flow_points:
            x, y = pt[0]
            cv2.circle(image, (int(x), int(y)), 3, color, -1)
            cv2.circle(image, (int(x), int(y)), 4, (255, 255, 255), 1)
    return image

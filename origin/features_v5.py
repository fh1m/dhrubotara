"""
XFeat Real-time Webcam Feature Matching with YOLOv11 + 3D Camera Guidance
Enhanced with separate 3D coordinate visualization window showing movement commands

When tracking degrades, a dedicated window shows 3D axes and movement vectors
to guide the camera back to the reference pose.
"""

import cv2
import torch
import numpy as np
import time
import sys
sys.path.insert(0, 'accelerated_features')
from modules.xfeat import XFeat
from ultralytics import YOLO


def estimate_camera_guidance(mkpts_0, mkpts_1, min_matches=8):
    """
    Estimate camera movement guidance based on feature matches
    Returns: (dx, dy, dz_scale, confidence, homography_found)
    """
    if len(mkpts_0) < min_matches:
        return 0, 0, 1.0, 0.0, False
    
    try:
        # Compute homography
        H, mask = cv2.findHomography(mkpts_1, mkpts_0, cv2.RANSAC, 5.0)
        
        if H is None:
            return 0, 0, 1.0, 0.0, False
        
        # Extract translation and scale from homography
        tx = H[0, 2]
        ty = H[1, 2]
        
        # Scale estimation from the homography
        sx = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
        sy = np.sqrt(H[0, 1]**2 + H[1, 1]**2)
        scale = (sx + sy) / 2.0
        
        # Inlier ratio as confidence
        confidence = np.sum(mask) / len(mask) * 100 if mask is not None else 0
        
        return tx, ty, scale, confidence, True
        
    except Exception as e:
        return 0, 0, 1.0, 0.0, False


def create_3d_guidance_window(dx, dy, dz_scale, confidence, size=(500, 500), threshold=10):
    """
    Create a dedicated 3D coordinate visualization window with movement commands
    Shows X, Y, Z axes and movement vectors in isometric 3D view
    """
    # Create dark blue-gray canvas for better contrast
    viz = np.ones((size[1], size[0], 3), dtype=np.uint8) * np.array([40, 40, 40], dtype=np.uint8)
    
    center_x, center_y = size[0] // 2, size[1] // 2 - 50
    
    # Title
    cv2.putText(viz, "CAMERA GUIDANCE SYSTEM", (center_x - 170, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(viz, (50, 50), (size[0] - 50, 50), (100, 100, 100), 2)
    
    # === Draw 3D Coordinate System (Isometric) ===
    axis_length = 100
    axis_thickness = 3
    
    # Origin point
    origin = (center_x, center_y)
    
    # X-axis (RIGHT) - Red
    x_end = (center_x + axis_length, center_y)
    cv2.arrowedLine(viz, origin, x_end, (50, 50, 255), axis_thickness, tipLength=0.15)
    cv2.circle(viz, x_end, 4, (50, 50, 255), -1)
    cv2.putText(viz, "X (Right)", (x_end[0] + 15, x_end[1] + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
    
    # Y-axis (DOWN) - Green
    y_end = (center_x, center_y + axis_length)
    cv2.arrowedLine(viz, origin, y_end, (50, 255, 50), axis_thickness, tipLength=0.15)
    cv2.circle(viz, y_end, 4, (50, 255, 50), -1)
    cv2.putText(viz, "Y (Down)", (y_end[0] + 15, y_end[1]), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
    
    # Z-axis (FORWARD/DEPTH) - Blue, diagonal up-left for isometric view
    z_end = (center_x - int(axis_length * 0.7), center_y - int(axis_length * 0.7))
    cv2.arrowedLine(viz, origin, z_end, (255, 150, 50), axis_thickness, tipLength=0.15)
    cv2.circle(viz, z_end, 4, (255, 150, 50), -1)
    cv2.putText(viz, "Z (Depth)", (z_end[0] - 90, z_end[1] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 50), 2)
    
    # Draw grid lines for reference
    grid_color = (70, 70, 70)
    for i in range(-1, 2):
        offset = i * 40
        # X grid
        cv2.line(viz, (center_x + offset, center_y - 50), 
                (center_x + offset, center_y + 50), grid_color, 1)
        # Y grid
        cv2.line(viz, (center_x - 50, center_y + offset), 
                (center_x + 50, center_y + offset), grid_color, 1)
    
    # === Draw Movement Vectors (Cyan/Yellow) ===
    arrow_scale = 0.4
    movement_color = (0, 255, 255)  # Cyan
    movement_thickness = 4
    
    # X movement (horizontal)
    if abs(dx) > threshold:
        x_move = int(np.clip(dx * arrow_scale, -120, 120))
        move_end_x = (center_x + x_move, center_y)
        cv2.arrowedLine(viz, origin, move_end_x, movement_color, movement_thickness, tipLength=0.25)
        
        # Pulsating effect circle at arrow end
        cv2.circle(viz, move_end_x, 8, movement_color, -1)
        cv2.circle(viz, move_end_x, 12, movement_color, 2)
    
    # Y movement (vertical)
    if abs(dy) > threshold:
        y_move = int(np.clip(dy * arrow_scale, -120, 120))
        move_end_y = (center_x, center_y + y_move)
        cv2.arrowedLine(viz, origin, move_end_y, movement_color, movement_thickness, tipLength=0.25)
        
        cv2.circle(viz, move_end_y, 8, movement_color, -1)
        cv2.circle(viz, move_end_y, 12, movement_color, 2)
    
    # Z movement (depth)
    if abs(dz_scale - 1.0) > 0.1:
        z_move = int(np.clip((dz_scale - 1.0) * 120, -90, 90))
        move_end_z = (center_x - int(z_move * 0.7), center_y - int(z_move * 0.7))
        cv2.arrowedLine(viz, origin, move_end_z, movement_color, movement_thickness, tipLength=0.25)
        
        cv2.circle(viz, move_end_z, 8, movement_color, -1)
        cv2.circle(viz, move_end_z, 12, movement_color, 2)
    
    # Origin marker (camera position)
    cv2.circle(viz, origin, 8, (255, 255, 255), -1)
    cv2.circle(viz, origin, 10, (200, 200, 200), 2)
    cv2.putText(viz, "CAM", (origin[0] - 20, origin[1] - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # === Movement Commands Panel ===
    cmd_y_start = size[1] - 220
    
    # Panel background
    cv2.rectangle(viz, (10, cmd_y_start - 10), (size[0] - 10, size[1] - 10), 
                 (60, 60, 60), -1)
    cv2.rectangle(viz, (10, cmd_y_start - 10), (size[0] - 10, size[1] - 10), 
                 (100, 100, 100), 2)
    
    cv2.putText(viz, "MOVEMENT COMMANDS:", (20, cmd_y_start + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cmd_y = cmd_y_start + 50
    commands_shown = False
    
    # Horizontal movement command
    if abs(dx) > threshold:
        if dx < 0:
            cmd_text = "◄◄◄ MOVE CAMERA LEFT"
            cmd_icon = "←"
        else:
            cmd_text = "MOVE CAMERA RIGHT ►►►"
            cmd_icon = "→"
        
        cv2.putText(viz, cmd_text, (30, cmd_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(viz, f"X-axis: {abs(dx):.1f} px", (350, cmd_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cmd_y += 40
        commands_shown = True
    
    # Vertical movement command
    if abs(dy) > threshold:
        if dy < 0:
            cmd_text = "▲▲▲ MOVE CAMERA UP"
            cmd_icon = "↑"
        else:
            cmd_text = "MOVE CAMERA DOWN ▼▼▼"
            cmd_icon = "↓"
        
        cv2.putText(viz, cmd_text, (30, cmd_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(viz, f"Y-axis: {abs(dy):.1f} px", (350, cmd_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cmd_y += 40
        commands_shown = True
    
    # Depth movement command
    if dz_scale > 1.1:
        cv2.putText(viz, "↗↗↗ MOVE CLOSER (Forward)", (30, cmd_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.putText(viz, f"Z-scale: {dz_scale:.2f}x", (350, cmd_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        commands_shown = True
    elif dz_scale < 0.9:
        cv2.putText(viz, "↙↙↙ MOVE AWAY (Backward)", (30, cmd_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        cv2.putText(viz, f"Z-scale: {dz_scale:.2f}x", (350, cmd_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        commands_shown = True
    
    # If aligned, show success message
    if not commands_shown:
        cv2.putText(viz, "✓ CAMERA ALIGNED!", (100, cmd_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2)
        cv2.putText(viz, "No movement needed", (130, cmd_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # === Status Bar at Top ===
    # Status indicator
    if abs(dx) < threshold and abs(dy) < threshold and 0.9 <= dz_scale <= 1.1:
        status_text = "STATUS: ALIGNED ✓"
        status_color = (100, 255, 100)
        status_bg = (20, 80, 20)
    else:
        status_text = "STATUS: ADJUSTING..."
        status_color = (0, 255, 255)
        status_bg = (40, 60, 60)
    
    # Status background
    cv2.rectangle(viz, (10, 60), (size[0] - 10, 100), status_bg, -1)
    cv2.rectangle(viz, (10, 60), (size[0] - 10, 100), status_color, 2)
    cv2.putText(viz, status_text, (20, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Confidence meter
    conf_bar_width = 150
    conf_bar_x = size[0] - conf_bar_width - 20
    cv2.putText(viz, "Confidence:", (conf_bar_x, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Confidence bar
    bar_y = 90
    cv2.rectangle(viz, (conf_bar_x, bar_y), (conf_bar_x + conf_bar_width, bar_y + 8), 
                 (80, 80, 80), -1)
    
    conf_filled = int(conf_bar_width * (confidence / 100.0))
    conf_color = (0, int(255 * confidence / 100), int(255 * (1 - confidence / 100)))
    cv2.rectangle(viz, (conf_bar_x, bar_y), (conf_bar_x + conf_filled, bar_y + 8), 
                 conf_color, -1)
    cv2.putText(viz, f"{confidence:.0f}%", (conf_bar_x + conf_bar_width + 5, 98),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
    
    return viz


def draw_matches(img1, img2, kpts1, kpts2, color=(0, 255, 0), radius=4, thickness=2):
    """Draw matches between two images side by side"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create side-by-side image
    max_h = max(h1, h2)
    canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2
    
    # Draw matches
    if len(kpts1) > 0 and len(kpts2) > 0:
        for pt1, pt2 in zip(kpts1, kpts2):
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]) + w1, int(pt2[1])
            
            # Draw line connecting matches
            cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            
            # Draw keypoints
            cv2.circle(canvas, (x1, y1), radius, color, -1)
            cv2.circle(canvas, (x2, y2), radius, color, -1)
            cv2.circle(canvas, (x1, y1), radius + 1, (255, 255, 255), 1)
            cv2.circle(canvas, (x2, y2), radius + 1, (255, 255, 255), 1)
    
    return canvas


def draw_yolo_detections(image, results, offset_x=0):
    """Draw YOLO detection boxes and labels"""
    vis_image = image.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1) + offset_x, int(y1), int(x2) + offset_x, int(y2)
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{result.names[cls]} {conf:.2f}"
            
            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (255, 0, 255), -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return vis_image


def draw_info_overlay(image, num_matches, max_matches, fps, yolo_active, stability, 
                     activate_thresh, deactivate_thresh, guidance_mode, mode="waiting"):
    """Draw information overlay with stability meter and guidance status"""
    overlay = image.copy()
    height, width = image.shape[:2]
    
    # Semi-transparent dark background for text
    if mode == "waiting":
        cv2.rectangle(overlay, (10, 10), (450, 100), (0, 0, 0), -1)
    else:
        cv2.rectangle(overlay, (10, 10), (450, 240), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    # Draw text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if mode == "waiting":
        cv2.putText(image, 'Press S to capture reference', (20, 40), 
                    font, 0.7, (0, 255, 255), 2)
        cv2.putText(image, 'Press Q to quit', (20, 70), 
                    font, 0.6, (255, 255, 255), 1)
    else:
        # Calculate match ratio
        match_ratio = (num_matches / max_matches * 100) if max_matches > 0 else 0
        
        # Color based on match ratio with hysteresis zones
        if match_ratio >= activate_thresh:
            ratio_color = (0, 255, 0)  # Green
        elif match_ratio >= deactivate_thresh:
            ratio_color = (0, 255, 255)  # Yellow
        else:
            ratio_color = (0, 0, 255)  # Red
        
        cv2.putText(image, f'Matches: {num_matches}/{max_matches}', (20, 40), 
                    font, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f'Match Ratio: {match_ratio:.1f}%', (20, 70), 
                    font, 0.7, ratio_color, 2)
        
        # Stability meter (visual progress bar)
        bar_width = 200
        bar_height = 20
        bar_x, bar_y = 20, 90
        
        # Draw bar background
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), 2)
        
        # Draw filled portion based on stability
        filled_width = int(bar_width * (stability / 100.0))
        bar_color = (0, int(255 * stability / 100), int(255 * (1 - stability / 100)))
        if filled_width > 0:
            cv2.rectangle(image, (bar_x + 2, bar_y + 2), 
                         (bar_x + filled_width - 2, bar_y + bar_height - 2), 
                         bar_color, -1)
        
        cv2.putText(image, f'Stability: {stability:.0f}%', (bar_x + bar_width + 10, bar_y + 14), 
                    font, 0.5, (255, 255, 255), 1)
        
        cv2.putText(image, f'FPS: {fps:.1f}', (20, 130), 
                    font, 0.6, (255, 255, 0), 2)
        
        # YOLO status with visual indicator
        yolo_status = "ACTIVE" if yolo_active else "INACTIVE"
        yolo_color = (0, 255, 0) if yolo_active else (128, 128, 128)
        
        # Draw YOLO indicator circle
        indicator_x, indicator_y = 130, 155
        cv2.circle(image, (indicator_x, indicator_y), 8, yolo_color, -1)
        cv2.circle(image, (indicator_x, indicator_y), 9, (255, 255, 255), 1)
        
        cv2.putText(image, f'YOLO: {yolo_status}', (20, 162), 
                    font, 0.6, yolo_color, 2)
        
        # Guidance mode indicator
        guidance_text = "GUIDANCE: ON" if guidance_mode else "GUIDANCE: OFF"
        guidance_color = (0, 255, 255) if guidance_mode else (100, 100, 100)
        cv2.putText(image, guidance_text, (20, 188), 
                    font, 0.5, guidance_color, 2)
        
        # Threshold info
        cv2.putText(image, f'Thresholds: On>{activate_thresh:.0f}% | Off<{deactivate_thresh:.0f}%', 
                    (20, 213), font, 0.4, (200, 200, 200), 1)
        
        cv2.putText(image, 'S: New ref | R: Reset | G: Toggle Guidance | Q: Quit', 
                    (20, 235), font, 0.35, (255, 255, 255), 1)
    
    return image


def process_frame_for_xfeat(frame, device):
    """Convert frame to XFeat input format"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    if device == 'cuda':
        frame_tensor = frame_tensor.cuda()
    return frame_tensor


def main():
    print("Initializing XFeat...")
    
    # Initialize XFeat
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    xfeat = XFeat()
    if device == 'cuda':
        xfeat = xfeat.cuda()
    
    # Initialize YOLOv11 nano model
    print("Loading YOLOv11 nano model...")
    yolo_model = YOLO('yolo11n.pt')
    print("YOLOv11 loaded successfully!")
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*70)
    print("XFeat + YOLOv11 with 3D Camera Guidance Visualization")
    print("="*70)
    print("Controls:")
    print("  S - Capture reference frame")
    print("  R - Reset (clear reference)")
    print("  G - Toggle camera guidance window")
    print("  Q - Quit")
    print("\n3D Camera Guidance Window:")
    print("  • Separate window with 3D coordinate system (X, Y, Z axes)")
    print("  • Visual movement vectors showing where to move camera")
    print("  • Clear directional commands (LEFT/RIGHT, UP/DOWN, CLOSER/AWAY)")
    print("  • Real-time confidence meter")
    print("  • Activates when tracking drops below threshold")
    print("="*70 + "\n")
    
    # State variables
    reference_frame = None
    reference_tensor = None
    max_matches = 0
    fps_history = []
    yolo_active = False
    guidance_enabled = True  # Start with guidance ON
    
    # Hysteresis thresholds
    activate_threshold = 45.0
    deactivate_threshold = 30.0
    
    # Stability tracking
    match_history = []
    stability_window = 10
    stability_threshold = 40.0
    
    try:
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                reference_frame = frame.copy()
                reference_tensor = process_frame_for_xfeat(reference_frame, device)
                
                # Get initial feature count
                with torch.no_grad():
                    ref_output = xfeat.detectAndCompute(reference_tensor, top_k=4096)[0]
                    max_matches = len(ref_output['keypoints'])
                
                print(f"✓ Reference captured! ({max_matches} features detected)")
                fps_history = []
                match_history = []
                yolo_active = False
            elif key == ord('r'):
                reference_frame = None
                reference_tensor = None
                max_matches = 0
                yolo_active = False
                match_history = []
                print("Reference cleared")
                fps_history = []
            elif key == ord('g'):
                guidance_enabled = not guidance_enabled
                print(f"Camera guidance: {'ENABLED' if guidance_enabled else 'DISABLED'}")
            
            # If no reference, just show current frame with instructions
            if reference_frame is None:
                display_frame = draw_info_overlay(frame, 0, 0, 0, False, 0,
                                                 activate_threshold, deactivate_threshold,
                                                 guidance_enabled, mode="waiting")
                cv2.imshow('XFeat + YOLOv11', display_frame)
                
                # Show "waiting" guidance window
                if guidance_enabled:
                    waiting_viz = np.ones((500, 500, 3), dtype=np.uint8) * 40
                    cv2.putText(waiting_viz, "WAITING FOR", (140, 230),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
                    cv2.putText(waiting_viz, "REFERENCE FRAME", (100, 280),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
                    cv2.putText(waiting_viz, "Press 'S' to capture", (120, 330),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
                    cv2.imshow('Camera Guidance - 3D View', waiting_viz)
                
                continue
            
            # Process current frame
            current_tensor = process_frame_for_xfeat(frame, device)
            
            # Match features between reference and current frame
            with torch.no_grad():
                mkpts_0, mkpts_1 = xfeat.match_xfeat(reference_tensor, current_tensor, top_k=4096)
            
            # Convert to numpy (if needed)
            if isinstance(mkpts_0, torch.Tensor):
                mkpts_0 = mkpts_0.cpu().numpy()
            if isinstance(mkpts_1, torch.Tensor):
                mkpts_1 = mkpts_1.cpu().numpy()
            
            num_matches = len(mkpts_0)
            
            # Calculate match ratio
            match_ratio = (num_matches / max_matches * 100) if max_matches > 0 else 0
            
            # Add to match history for stability calculation
            match_history.append(match_ratio)
            if len(match_history) > stability_window:
                match_history.pop(0)
            
            # Calculate stability
            if len(match_history) >= 3:
                avg_ratio = np.mean(match_history)
                variance = np.var(match_history)
                stability = min(100, avg_ratio * (1 - min(1, variance / 1000)))
            else:
                stability = match_ratio
            
            # Hysteresis logic
            if not yolo_active:
                should_activate = (match_ratio >= activate_threshold and 
                                 stability >= stability_threshold)
                if should_activate:
                    print(f"✓ YOLO ACTIVATED - Ratio: {match_ratio:.1f}% | Stability: {stability:.1f}%")
                    yolo_active = True
            else:
                should_deactivate = match_ratio < deactivate_threshold
                if should_deactivate:
                    print(f"✗ YOLO DEACTIVATED - Ratio dropped to {match_ratio:.1f}%")
                    yolo_active = False
            
            # Create base visualization with matches
            match_vis = draw_matches(reference_frame, frame, mkpts_0, mkpts_1)
            
            # Run YOLO if active
            if yolo_active:
                results = yolo_model(frame, verbose=False)
                match_vis = draw_yolo_detections(match_vis, results, offset_x=reference_frame.shape[1])
            
            # === 3D Camera Guidance Window (Separate) ===
            if guidance_enabled:
                # Estimate camera movement needed
                dx, dy, dz_scale, confidence, found = estimate_camera_guidance(mkpts_0, mkpts_1)
                
                if found and match_ratio < activate_threshold:
                    # Show guidance when tracking is degraded
                    guidance_viz = create_3d_guidance_window(dx, dy, dz_scale, confidence)
                    cv2.imshow('Camera Guidance - 3D View', guidance_viz)
                elif match_ratio >= activate_threshold:
                    # Show "tracking OK" when stable
                    ok_viz = np.ones((500, 500, 3), dtype=np.uint8) * 40
                    cv2.putText(ok_viz, "✓ TRACKING OK", (120, 230),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 3)
                    cv2.putText(ok_viz, "No guidance needed", (110, 290),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    cv2.putText(ok_viz, f"Match Ratio: {match_ratio:.1f}%", (140, 340),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                    cv2.imshow('Camera Guidance - 3D View', ok_viz)
                else:
                    # Not enough matches for homography
                    search_viz = np.ones((500, 500, 3), dtype=np.uint8) * 40
                    cv2.putText(search_viz, "SEARCHING...", (130, 230),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.putText(search_viz, "Not enough matches", (100, 280),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    cv2.putText(search_viz, f"Matches: {num_matches}", (180, 320),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                    cv2.imshow('Camera Guidance - 3D View', search_viz)
            
            # Calculate FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history) if fps_history else fps
            
            # Add info overlay
            match_vis = draw_info_overlay(match_vis, num_matches, max_matches, 
                                         avg_fps, yolo_active, stability,
                                         activate_threshold, deactivate_threshold,
                                         guidance_enabled, mode="matching")
            
            # Display main window
            cv2.imshow('XFeat + YOLOv11', match_vis)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if fps_history:
            print(f"\nAverage FPS: {np.mean(fps_history):.1f}")
        print("Done!")


if __name__ == '__main__':
    main()

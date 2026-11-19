"""
XFeat + YOLOv11 with Optical Flow Guidance - Complete Hybrid System
Single unified window with 3D guidance using multiple estimation methods

Guidance sources (in priority order):
1. XFeat homography (most accurate, needs good features)
2. Object-based tracking (fallback when features degrade)
3. Optical Flow (smooth motion estimation, works even with few features)

Uses Lucas-Kanade sparse optical flow for robust motion tracking.
"""

import cv2
import torch
import numpy as np
import time
import sys

sys.path.insert(0, "accelerated_features")
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

    # If no previous points, detect features to track
    if prev_points is None or len(prev_points) < 10:
        # Use Shi-Tomasi corner detection
        feature_params = dict(
            maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7
        )
        prev_points = cv2.goodFeaturesToTrack(ref_gray, mask=None, **feature_params)

        if prev_points is None:
            return 0, 0, 1.0, 0.0, False, None

    # Lucas-Kanade parameters
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    try:
        # Calculate optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            ref_gray, curr_gray, prev_points, None, **lk_params
        )

        if next_points is None:
            return 0, 0, 1.0, 0.0, False, None

        # Select good points
        good_prev = prev_points[status == 1]
        good_next = next_points[status == 1]

        if len(good_prev) < 5:
            return 0, 0, 1.0, 0.0, False, None

        # Calculate displacement
        displacement = good_next - good_prev
        dx = np.median(displacement[:, 0])
        dy = np.median(displacement[:, 1])

        # Estimate scale from flow magnitude distribution
        flow_magnitude = np.linalg.norm(displacement, axis=1)
        avg_flow = np.median(flow_magnitude)

        # Larger flow = moving closer (simple heuristic)
        # This is approximate - real scale needs stereo or known object sizes
        if avg_flow > 2.0:
            scale = 1.0 + (avg_flow - 2.0) / 50.0  # Rough approximation
        else:
            scale = 1.0

        # Confidence based on number of tracked points and consistency
        flow_std = np.std(flow_magnitude)
        consistency = 1.0 / (1.0 + flow_std / 10.0)
        confidence = min(100, (len(good_prev) / 200.0) * 100 * consistency)

        return -dx, -dy, scale, confidence, True, good_next.reshape(-1, 1, 2)

    except Exception as e:
        return 0, 0, 1.0, 0.0, False, None


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


def create_slick_3d_guidance(
    dx, dy, dz_scale, confidence, guidance_source, size=(600, 600), threshold=10
):
    """
    Create clean 3D guidance visualization focused on vectors
    """
    # Gradient background
    viz = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for i in range(size[1]):
        factor = i / size[1]
        viz[i, :] = [
            int(20 + factor * 30),
            int(25 + factor * 20),
            int(40 + factor * 25),
        ]

    center_x, center_y = size[0] // 2, size[1] // 2

    # Title
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

    # 3D Coordinate System
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
):
    """
    Create unified window with camera views and 3D guidance
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

    # Create 3D guidance
    if show_guidance:
        guidance_viz = create_slick_3d_guidance(
            dx, dy, dz_scale, confidence, guidance_source, size=(600, new_h)
        )
    else:
        guidance_viz = np.zeros((new_h, 600, 3), dtype=np.uint8)
        cv2.putText(
            guidance_viz,
            "Guidance Disabled",
            (150, new_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 100, 100),
            2,
        )

    top_row = np.hstack([ref_resized, curr_resized, guidance_viz])

    # Info panel - EXPANDED
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

    # Flow points count
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

    final_viz = np.vstack([top_row, info_panel])

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
    print("Initializing XFeat...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    xfeat = XFeat()
    if device == "cuda":
        xfeat = xfeat.cuda()

    print("Loading YOLOv11 nano model...")
    yolo_model = YOLO("yolo11n.pt")
    print("YOLOv11 loaded successfully!")

    print("Opening webcam...")
    cap = cv2.VideoCapture(4)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 75)
    print("XFeat + YOLOv11 + Optical Flow - Complete Hybrid Guidance")
    print("=" * 75)
    print("Features:")
    print("  • Triple guidance: Features → Objects → Optical Flow")
    print("  • Lucas-Kanade sparse optical flow for smooth tracking")
    print("  • Automatic fallback cascade for maximum robustness")
    print("  • Visual feedback: Green=features, Yellow=flow points")
    print("\nGuidance Priority:")
    print("  1. XFeat homography (most accurate)")
    print("  2. Object-based tracking (semantic fallback)")
    print("  3. Optical flow (motion continuity)")
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

    # Optical flow tracking
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

                # Extract reference objects
                ref_results = yolo_model(reference_frame, verbose=False)
                reference_objects = extract_object_info(ref_results)

                # Reset optical flow
                flow_tracked_points = None

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
                cv2.imshow("XFeat + YOLOv11 + Optical Flow", frame)
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

            # Get YOLO results
            curr_results = yolo_model(frame, verbose=False) if yolo_active else None

            # === HYBRID GUIDANCE CASCADE ===
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
                        print(
                            f"ℹ Using object-based guidance (confidence: {confidence:.0f}%)"
                        )

                # Priority 3: Optical flow (always try as last resort)
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
                    if guidance_source == "optical_flow" and conf_flow > 20:
                        print(
                            f"ℹ Using optical flow guidance (confidence: {confidence:.0f}%, {len(flow_tracked_points)} pts)"
                        )

            # Calculate FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history) if fps_history else fps

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
            )

            cv2.imshow("XFeat + YOLOv11 + Optical Flow", unified_viz)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if fps_history:
            print(f"\nAverage FPS: {np.mean(fps_history):.1f}")
        print("Done!")


if __name__ == "__main__":
    main()

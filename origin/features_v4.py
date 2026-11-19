"""
XFeat Real-time Webcam Feature Matching with YOLOv11 Object Detection
Enhanced with Hysteresis and Stability Tracking to prevent flickering

Press 's' to capture reference frame, then see live matches.
YOLO detection uses smart activation with temporal smoothing.
"""

import cv2
import torch
import numpy as np
import time
import sys

sys.path.insert(0, "accelerated_features")
from modules.xfeat import XFeat
from ultralytics import YOLO


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
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (255, 0, 255),
                -1,
            )

            # Draw label text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    return vis_image


def draw_info_overlay(
    image,
    num_matches,
    max_matches,
    fps,
    yolo_active,
    stability,
    activate_thresh,
    deactivate_thresh,
    mode="waiting",
):
    """Draw information overlay with stability meter"""
    overlay = image.copy()
    height, width = image.shape[:2]

    # Semi-transparent dark background for text
    if mode == "waiting":
        cv2.rectangle(overlay, (10, 10), (450, 100), (0, 0, 0), -1)
    else:
        cv2.rectangle(overlay, (10, 10), (450, 220), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Draw text information
    font = cv2.FONT_HERSHEY_SIMPLEX

    if mode == "waiting":
        cv2.putText(
            image, "Press S to capture reference", (20, 40), font, 0.7, (0, 255, 255), 2
        )
        cv2.putText(image, "Press Q to quit", (20, 70), font, 0.6, (255, 255, 255), 1)
    else:
        # Calculate match ratio
        match_ratio = (num_matches / max_matches * 100) if max_matches > 0 else 0

        # Color based on match ratio with hysteresis zones
        if match_ratio >= activate_thresh:
            ratio_color = (0, 255, 0)  # Green - will activate/stay active
        elif match_ratio >= deactivate_thresh:
            ratio_color = (0, 255, 255)  # Yellow - hysteresis zone
        else:
            ratio_color = (0, 0, 255)  # Red - will deactivate/stay inactive

        cv2.putText(
            image,
            f"Matches: {num_matches}/{max_matches}",
            (20, 40),
            font,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            image,
            f"Match Ratio: {match_ratio:.1f}%",
            (20, 70),
            font,
            0.7,
            ratio_color,
            2,
        )

        # Stability meter (visual progress bar)
        bar_width = 200
        bar_height = 20
        bar_x, bar_y = 20, 90

        # Draw bar background
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (100, 100, 100),
            2,
        )

        # Draw filled portion based on stability
        filled_width = int(bar_width * (stability / 100.0))
        bar_color = (0, int(255 * stability / 100), int(255 * (1 - stability / 100)))
        if filled_width > 0:
            cv2.rectangle(
                image,
                (bar_x + 2, bar_y + 2),
                (bar_x + filled_width - 2, bar_y + bar_height - 2),
                bar_color,
                -1,
            )

        cv2.putText(
            image,
            f"Stability: {stability:.0f}%",
            (bar_x + bar_width + 10, bar_y + 14),
            font,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.putText(image, f"FPS: {fps:.1f}", (20, 130), font, 0.6, (255, 255, 0), 2)

        # YOLO status with visual indicator
        yolo_status = "ACTIVE" if yolo_active else "INACTIVE"
        yolo_color = (0, 255, 0) if yolo_active else (128, 128, 128)

        # Draw YOLO indicator circle
        indicator_x, indicator_y = 130, 155
        cv2.circle(image, (indicator_x, indicator_y), 8, yolo_color, -1)
        cv2.circle(image, (indicator_x, indicator_y), 9, (255, 255, 255), 1)

        cv2.putText(image, f"YOLO: {yolo_status}", (20, 162), font, 0.6, yolo_color, 2)

        # Threshold info (show the hysteresis band)
        cv2.putText(
            image,
            f"Activation Zone: >{activate_thresh:.0f}%  |  "
            + f"Deactivation: <{deactivate_thresh:.0f}%",
            (20, 188),
            font,
            0.45,
            (200, 200, 200),
            1,
        )

        cv2.putText(
            image,
            "S: New ref | R: Reset | Q: Quit",
            (20, 213),
            font,
            0.4,
            (255, 255, 255),
            1,
        )

    return image


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

    # Initialize XFeat
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    xfeat = XFeat()
    if device == "cuda":
        xfeat = xfeat.cuda()

    # Initialize YOLOv11 nano model
    print("Loading YOLOv11 nano model...")
    yolo_model = YOLO("yolo11n.pt")
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

    print("\n" + "=" * 60)
    print("XFeat + YOLOv11 with Smart Activation")
    print("=" * 60)
    print("Controls:")
    print("  S - Capture reference frame")
    print("  R - Reset (clear reference)")
    print("  Q - Quit")
    print("\nSmart YOLO Detection:")
    print("  • Activates: Match ratio ≥45% AND stable tracking")
    print("  • Deactivates: Match ratio <30%")
    print("  • Hysteresis prevents flickering")
    print("  • Stability meter shows tracking confidence")
    print("=" * 60 + "\n")

    # State variables
    reference_frame = None
    reference_tensor = None
    max_matches = 0
    fps_history = []
    yolo_active = False

    # Hysteresis thresholds (different for activation vs deactivation)
    activate_threshold = 45.0  # Need 45% to turn ON
    deactivate_threshold = 30.0  # Need to drop below 30% to turn OFF

    # Stability tracking (temporal smoothing)
    match_history = []
    stability_window = 10  # Number of frames to consider
    stability_threshold = 40.0  # Need 40% average to be "stable"

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

            if key == ord("q"):
                break
            elif key == ord("s"):
                reference_frame = frame.copy()
                reference_tensor = process_frame_for_xfeat(reference_frame, device)

                # Get initial feature count
                with torch.no_grad():
                    ref_output = xfeat.detectAndCompute(reference_tensor, top_k=4096)[0]
                    max_matches = len(ref_output["keypoints"])

                print(f"✓ Reference captured! ({max_matches} features detected)")
                fps_history = []
                match_history = []
                yolo_active = False
            elif key == ord("r"):
                reference_frame = None
                reference_tensor = None
                max_matches = 0
                yolo_active = False
                match_history = []
                print("Reference cleared")
                fps_history = []

            # If no reference, just show current frame with instructions
            if reference_frame is None:
                display_frame = draw_info_overlay(
                    frame,
                    0,
                    0,
                    0,
                    False,
                    0,
                    activate_threshold,
                    deactivate_threshold,
                    mode="waiting",
                )
                cv2.imshow("XFeat + YOLOv11 (Smart)", display_frame)
                continue

            # Process current frame
            current_tensor = process_frame_for_xfeat(frame, device)

            # Match features between reference and current frame
            with torch.no_grad():
                mkpts_0, mkpts_1 = xfeat.match_xfeat(
                    reference_tensor, current_tensor, top_k=4096
                )

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

            # Calculate stability (variance-based: lower variance = more stable)
            if len(match_history) >= 3:
                avg_ratio = np.mean(match_history)
                variance = np.var(match_history)
                # Stability score: high when variance is low and average is decent
                # Normalize variance impact (lower variance = higher stability)
                stability = min(100, avg_ratio * (1 - min(1, variance / 1000)))
            else:
                stability = match_ratio

            # Hysteresis logic: different thresholds for activation vs deactivation
            if not yolo_active:
                # To activate: need good ratio AND stability
                should_activate = (
                    match_ratio >= activate_threshold
                    and stability >= stability_threshold
                )
                if should_activate:
                    print(
                        f"✓ YOLO ACTIVATED - Ratio: {match_ratio:.1f}% | Stability: {stability:.1f}%"
                    )
                    yolo_active = True
            else:
                # To deactivate: need ratio to drop significantly (hysteresis)
                should_deactivate = match_ratio < deactivate_threshold
                if should_deactivate:
                    print(f"✗ YOLO DEACTIVATED - Ratio dropped to {match_ratio:.1f}%")
                    yolo_active = False

            # Create base visualization with matches
            match_vis = draw_matches(reference_frame, frame, mkpts_0, mkpts_1)

            # Run YOLO if active
            if yolo_active:
                results = yolo_model(frame, verbose=False)
                # Draw detections on the current frame (right side)
                match_vis = draw_yolo_detections(
                    match_vis, results, offset_x=reference_frame.shape[1]
                )

            # Calculate FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history) if fps_history else fps

            # Add info overlay
            match_vis = draw_info_overlay(
                match_vis,
                num_matches,
                max_matches,
                avg_fps,
                yolo_active,
                stability,
                activate_threshold,
                deactivate_threshold,
                mode="matching",
            )

            # Display
            cv2.imshow("XFeat + YOLOv11 (Smart)", match_vis)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if fps_history:
            print(f"\nAverage FPS: {np.mean(fps_history):.1f}")
        print("Done!")


if __name__ == "__main__":
    main()

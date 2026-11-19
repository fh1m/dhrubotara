"""
XFeat Real-time Webcam Feature Matching
Press 's' to capture reference frame, then see live matches against it
"""

import cv2
import torch
import numpy as np
import time
import sys

sys.path.insert(0, "accelerated_features")
from modules.xfeat import XFeat


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


def draw_info_overlay(image, num_matches, fps, mode="waiting"):
    """Draw information overlay on the image"""
    overlay = image.copy()
    height, width = image.shape[:2]

    # Semi-transparent dark background for text
    if mode == "waiting":
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
    else:
        cv2.rectangle(overlay, (10, 10), (400, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Draw text information
    font = cv2.FONT_HERSHEY_SIMPLEX

    if mode == "waiting":
        cv2.putText(
            image, "Press S to capture reference", (20, 40), font, 0.7, (0, 255, 255), 2
        )
        cv2.putText(image, "Press Q to quit", (20, 70), font, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(
            image, f"Matches: {num_matches}", (20, 40), font, 0.8, (0, 255, 0), 2
        )
        cv2.putText(image, f"FPS: {fps:.1f}", (20, 75), font, 0.7, (255, 255, 0), 2)
        cv2.putText(
            image,
            "S: New reference | R: Reset | Q: Quit",
            (20, 110),
            font,
            0.5,
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

    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 50)
    print("Controls:")
    print("  S - Capture reference frame")
    print("  R - Reset (clear reference)")
    print("  Q - Quit")
    print("=" * 50 + "\n")

    # State variables
    reference_frame = None
    reference_tensor = None
    fps_history = []

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
                print("Reference frame captured!")
                fps_history = []  # Reset FPS history
            elif key == ord("r"):
                reference_frame = None
                reference_tensor = None
                print("Reference cleared")
                fps_history = []

            # If no reference, just show current frame with instructions
            if reference_frame is None:
                display_frame = draw_info_overlay(frame, 0, 0, mode="waiting")
                cv2.imshow("XFeat Feature Matching", display_frame)
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

            # Calculate FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history) if fps_history else fps

            # Create visualization
            match_vis = draw_matches(reference_frame, frame, mkpts_0, mkpts_1)

            # Add info overlay
            match_vis = draw_info_overlay(
                match_vis, num_matches, avg_fps, mode="matching"
            )

            # Display
            cv2.imshow("XFeat Feature Matching", match_vis)

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

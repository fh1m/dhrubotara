#!/usr/bin/env python3
"""
Visual Navigation System - Main Entry Point
XFeat + YOLOv11 + Optical Flow + Pixhawk IMU Sensor Fusion
"""

import cv2
import torch
import numpy as np
import time
import sys

# Add accelerated_features to path
sys.path.insert(0, "../accelerated_features")

from modules.xfeat import XFeat
from ultralytics import YOLO

# Import subsystems
from subsystems.pixhawk import PixhawkIMU
from subsystems.xfeat import process_frame_for_xfeat, estimate_camera_guidance, fuse_visual_imu_guidance
from subsystems.detection import extract_object_info, estimate_guidance_from_objects
from subsystems.optical_flow import estimate_guidance_from_optical_flow
from subsystems.dashboard import (
    PerformanceMonitor,
    draw_unified_window,
    switch_tab
)

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
    cap = cv2.VideoCapture(0)

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
            
            # Tab switching (5 tabs now)
            elif key == ord("1"):
                switch_tab(0)
                print("Switched to CENTRAL tab")
            elif key == ord("2"):
                switch_tab(1)
                print("Switched to VISION tab")
            elif key == ord("3"):
                switch_tab(2)
                print("Switched to IMU/FUSION tab")
            elif key == ord("4"):
                switch_tab(3)
                print("Switched to FLOW tab")
            elif key == ord("5"):
                switch_tab(4)
                print("Switched to PERFORMANCE tab")

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

            # YOLO is ALWAYS ACTIVE - run on every frame
            yolo_active = True
            curr_results = yolo_model(frame, verbose=False)

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
            
            # Update performance monitor with trajectory
            ekf_vel_mag = 0.0
            if imu_data is not None:
                ekf = imu_data.get_ekf_data()
                if ekf['age'] < 1.0:
                    ekf_vel_mag = np.sqrt(ekf['vel_x']**2 + ekf['vel_y']**2 + ekf['vel_z']**2)
            
            performance_monitor.update(avg_fps, confidence, num_matches, ekf_vel_mag, 
                                      dx if found else 0, dy if found else 0, 
                                      (dz_scale - 1.0) if found else 0)

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

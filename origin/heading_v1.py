import cv2
import numpy as np
import argparse
import math
from collections import deque
import time


class HeadingEstimator:
    def __init__(self):
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )

        # Initialize tracking variables
        self.tracks = []
        self.track_len = 20
        self.frame_idx = 1
        self.prev_gray = None

        # Heading estimation variables
        self.heading_history = deque(maxlen=30)  # Store last 30 heading estimates
        self.velocity_history = deque(maxlen=30)

        # Colors for visualization
        self.colors = np.random.randint(0, 255, (100, 3))

    def detect_features(self, gray):
        """Detect good features to track using Shi-Tomasi corner detector"""
        corners = cv2.goodFeaturesToTrack(gray, **self.feature_params)
        return corners

    def lucas_kanade_flow(self, prev_gray, gray, p0):
        """Calculate optical flow using Lucas-Kanade method"""
        if p0 is None or len(p0) == 0:
            return None, None, None

        # Calculate optical flow
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None, **self.lk_params
        )

        # Select good points
        good_new = p1[status == 1]
        good_old = p0[status == 1]

        return good_new, good_old, status

    def horn_schunck_flow(self, prev_gray, gray):
        """Calculate dense optical flow using Horn-Schunck method (Farneback implementation)"""
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        return flow

    def block_matching_flow(self, prev_gray, gray, block_size=32, search_range=32):
        """Calculate optical flow using simple block matching"""
        h, w = gray.shape
        flow = np.zeros((h, w, 2), dtype=np.float32)

        # Create grid of blocks
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                template = prev_gray[y : y + block_size, x : x + block_size]

                # Define search area
                y_min = max(0, y - search_range)
                y_max = min(h - block_size, y + search_range)
                x_min = max(0, x - search_range)
                x_max = min(w - block_size, x + search_range)

                search_area = gray[y_min:y_max, x_min:x_max]

                # Template matching
                result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                # Calculate displacement
                dx = max_loc[0] + x_min - x
                dy = max_loc[1] + y_min - y

                # Fill flow for this block
                flow[y : y + block_size, x : x + block_size, 0] = dx
                flow[y : y + block_size, x : x + block_size, 1] = dy

        return flow

    def estimate_heading_from_flow(self, flow):
        """Estimate heading change from optical flow field"""
        if flow is None:
            return 0, 0, 0

        # Calculate flow statistics
        flow_magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        flow_angle = np.arctan2(flow[:, :, 1], flow[:, :, 0]) * 180 / np.pi

        # Filter out small movements (noise)
        threshold = np.percentile(flow_magnitude, 75)
        mask = flow_magnitude > threshold

        if not np.any(mask):
            return 0, 0, 0

        # Calculate dominant flow direction
        valid_angles = flow_angle[mask]
        valid_magnitudes = flow_magnitude[mask]

        # Weighted average of flow angles
        if len(valid_angles) > 0:
            # Convert to unit vectors and average
            x_components = np.cos(valid_angles * np.pi / 180) * valid_magnitudes
            y_components = np.sin(valid_angles * np.pi / 180) * valid_magnitudes

            avg_x = np.mean(x_components)
            avg_y = np.mean(y_components)

            # Calculate heading change (opposite of flow direction for camera motion)
            heading_change = math.atan2(-avg_y, -avg_x) * 180 / np.pi
            velocity = math.sqrt(avg_x**2 + avg_y**2)
            confidence = (
                np.mean(valid_magnitudes) / np.max(flow_magnitude)
                if np.max(flow_magnitude) > 0
                else 0
            )
        else:
            heading_change = 0
            velocity = 0
            confidence = 0

        return heading_change, velocity, confidence

    def estimate_heading_from_features(self, good_new, good_old):
        """Estimate heading from feature point movements"""
        if good_new is None or good_old is None or len(good_new) < 5:
            return 0, 0, 0

        # Calculate movement vectors
        movements = good_new - good_old

        # Filter out outliers using median
        movement_magnitudes = np.sqrt(movements[:, 0] ** 2 + movements[:, 1] ** 2)
        median_magnitude = np.median(movement_magnitudes)

        # Keep movements within 2x median
        valid_mask = movement_magnitudes < 2 * median_magnitude
        valid_movements = movements[valid_mask]

        if len(valid_movements) < 3:
            return 0, 0, 0

        # Calculate average movement (opposite for camera motion)
        avg_movement = -np.mean(valid_movements, axis=0)

        # Convert to heading and velocity
        heading_change = math.atan2(avg_movement[1], avg_movement[0]) * 180 / np.pi
        velocity = np.linalg.norm(avg_movement)
        confidence = len(valid_movements) / len(good_new)

        return heading_change, velocity, confidence

    def draw_flow_field(self, img, flow, step=16):
        """Draw optical flow field on image"""
        h, w = img.shape[:2]
        y, x = (
            np.mgrid[step // 2 : h : step, step // 2 : w : step]
            .reshape(2, -1)
            .astype(int)
        )
        fx, fy = flow[y, x].T

        # Create line endpoints
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)

        # Draw flow vectors
        for (x1, y1), (x2, y2) in lines:
            if abs(x2 - x1) > 1 or abs(y2 - y1) > 1:  # Only draw significant movements
                cv2.arrowedLine(img, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

        return img

    def draw_features(self, img, good_new, good_old):
        """Draw tracked features and their trajectories"""
        if good_new is None or good_old is None:
            return img

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            color = self.colors[i % len(self.colors)].tolist()
            cv2.line(img, (a, b), (c, d), color, 2)
            cv2.circle(img, (a, b), 3, color, -1)

        return img

    def process_frame(self, frame):
        """Process a single frame and return visualization with heading information"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()

        if self.prev_gray is not None:
            # Method 1: Lucas-Kanade Optical Flow with feature tracking
            if self.frame_idx % 5 == 0:  # Detect new features every 5 frames
                features = self.detect_features(gray)
                if features is not None:
                    self.tracks = [
                        tr[-1:] for tr in self.tracks
                    ]  # Keep only latest point
                    for x, y in features.reshape(-1, 2):
                        self.tracks.append([[x, y]])

            # Track existing features
            if self.tracks:
                img0, img1 = self.prev_gray, gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                good_new, good_old, status = self.lucas_kanade_flow(img0, img1, p0)

                if good_new is not None:
                    # Update tracks
                    good_tracks = []
                    for i, (tr, (x, y)) in enumerate(
                        zip(self.tracks, good_new.reshape(-1, 2))
                    ):
                        if status[i] == 1:
                            tr.append([x, y])
                            if len(tr) > self.track_len:
                                tr.pop(0)
                            good_tracks.append(tr)
                    self.tracks = good_tracks

                    # Estimate heading from features
                    lk_heading, lk_velocity, lk_confidence = (
                        self.estimate_heading_from_features(good_new, good_old)
                    )

                    # Draw features
                    vis = self.draw_features(vis, good_new, good_old)
            else:
                lk_heading, lk_velocity, lk_confidence = 0, 0, 0

            # Method 2: Horn-Schunck (Farneback) Dense Optical Flow
            flow_dense = self.horn_schunck_flow(self.prev_gray, gray)
            hs_heading, hs_velocity, hs_confidence = self.estimate_heading_from_flow(
                flow_dense
            )

            # Method 3: Block Matching
            flow_bm = self.block_matching_flow(self.prev_gray, gray)
            bm_heading, bm_velocity, bm_confidence = self.estimate_heading_from_flow(
                flow_bm
            )

            # Combine estimates (weighted by confidence)
            total_confidence = lk_confidence + hs_confidence + bm_confidence
            if total_confidence > 0:
                combined_heading = (
                    lk_heading * lk_confidence
                    + hs_heading * hs_confidence
                    + bm_heading * bm_confidence
                ) / total_confidence
                combined_velocity = (
                    lk_velocity * lk_confidence
                    + hs_velocity * hs_confidence
                    + bm_velocity * bm_confidence
                ) / total_confidence
            else:
                combined_heading, combined_velocity = 0, 0

            # Update history
            self.heading_history.append(combined_heading)
            self.velocity_history.append(combined_velocity)

            # Calculate smoothed heading
            smoothed_heading = (
                np.mean(list(self.heading_history)[-10:])
                if len(self.heading_history) > 0
                else 0
            )

            # Draw flow field (subsampled for visibility)
            vis = self.draw_flow_field(vis, flow_dense, step=20)

            # Add information overlay
            self.draw_info_overlay(
                vis,
                lk_heading,
                lk_velocity,
                lk_confidence,
                hs_heading,
                hs_velocity,
                hs_confidence,
                bm_heading,
                bm_velocity,
                bm_confidence,
                combined_heading,
                combined_velocity,
                smoothed_heading,
            )

        self.prev_gray = gray.copy()
        self.frame_idx += 1
        return vis

    def draw_info_overlay(
        self,
        img,
        lk_h,
        lk_v,
        lk_c,
        hs_h,
        hs_v,
        hs_c,
        bm_h,
        bm_v,
        bm_c,
        combined_h,
        combined_v,
        smoothed_h,
    ):
        """Draw information overlay on the image"""
        h, w = img.shape[:2]

        # Create semi-transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (420, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1

        # Draw method results
        y_offset = 30
        cv2.putText(
            img, "AUV Heading Estimation", (15, y_offset), font, 0.7, (0, 255, 255), 2
        )

        y_offset += 25
        cv2.putText(
            img,
            f"Lucas-Kanade: {lk_h:.1f}° | Vel: {lk_v:.2f} | Conf: {lk_c:.2f}",
            (15, y_offset),
            font,
            scale,
            (0, 255, 0),
            thickness,
        )

        y_offset += 20
        cv2.putText(
            img,
            f"Horn-Schunck: {hs_h:.1f}° | Vel: {hs_v:.2f} | Conf: {hs_c:.2f}",
            (15, y_offset),
            font,
            scale,
            (255, 0, 0),
            thickness,
        )

        y_offset += 20
        cv2.putText(
            img,
            f"Block Match:  {bm_h:.1f}° | Vel: {bm_v:.2f} | Conf: {bm_c:.2f}",
            (15, y_offset),
            font,
            scale,
            (0, 0, 255),
            thickness,
        )

        y_offset += 25
        cv2.putText(
            img,
            f"Combined Heading: {combined_h:.1f}°",
            (15, y_offset),
            font,
            0.6,
            (255, 255, 0),
            2,
        )

        y_offset += 20
        cv2.putText(
            img,
            f"Smoothed Heading: {smoothed_h:.1f}°",
            (15, y_offset),
            font,
            0.6,
            (255, 255, 255),
            2,
        )

        y_offset += 20
        cv2.putText(
            img,
            f"Combined Velocity: {combined_v:.2f}",
            (15, y_offset),
            font,
            scale,
            (255, 255, 255),
            thickness,
        )

        # Draw heading compass
        self.draw_compass(img, smoothed_h, w - 120, 120)

        # Draw velocity history graph
        self.draw_velocity_graph(img, 15, h - 100, 200, 80)

    def draw_compass(self, img, heading, cx, cy):
        """Draw a simple compass showing current heading"""
        radius = 50

        # Draw compass circle
        cv2.circle(img, (cx, cy), radius, (100, 100, 100), 2)
        cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)

        # Draw cardinal directions
        cv2.putText(
            img,
            "N",
            (cx - 5, cy - radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img,
            "E",
            (cx + radius + 5, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        # Draw heading arrow
        angle_rad = (
            (heading - 90) * np.pi / 180
        )  # Convert to radians, offset for north up
        end_x = int(cx + radius * 0.8 * np.cos(angle_rad))
        end_y = int(cy + radius * 0.8 * np.sin(angle_rad))

        cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (0, 255, 255), 3, tipLength=0.3)

        # Display heading value
        cv2.putText(
            img,
            f"{heading:.1f}°",
            (cx - 25, cy + radius + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

    def draw_velocity_graph(self, img, x, y, width, height):
        """Draw velocity history graph"""
        if len(self.velocity_history) < 2:
            return

        # Draw graph background
        cv2.rectangle(img, (x, y), (x + width, y + height), (50, 50, 50), -1)
        cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 1)

        # Draw velocity history
        velocities = list(self.velocity_history)
        if len(velocities) > 1:
            max_vel = max(velocities) if max(velocities) > 0 else 1

            for i in range(1, len(velocities)):
                x1 = x + int((i - 1) * width / len(velocities))
                y1 = y + height - int(velocities[i - 1] / max_vel * height)
                x2 = x + int(i * width / len(velocities))
                y2 = y + height - int(velocities[i] / max_vel * height)

                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label
        cv2.putText(
            img,
            "Velocity",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )


def main():
    parser = argparse.ArgumentParser(
        description="AUV Heading Estimation using Optical Flow"
    )
    parser.add_argument(
        "--input", "-i", help="Input video file (default: camera)", default=None
    )
    parser.add_argument(
        "--camera", "-c", type=int, help="Camera index (default: 0)", default=0
    )
    args = parser.parse_args()

    # Initialize heading estimator
    estimator = HeadingEstimator()

    # Open video source
    if args.input:
        cap = cv2.VideoCapture(args.input)
        print(f"Processing video: {args.input}")
    else:
        cap = cv2.VideoCapture(args.camera)
        print(f"Using camera: {args.camera}")

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Video FPS: {fps}")

    # Processing loop
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.input:  # If it's a video file, restart from beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # Process frame
        result = estimator.process_frame(frame)

        # Calculate and display processing FPS
        frame_count += 1
        if frame_count % 30 == 0:  # Update every 30 frames
            elapsed_time = time.time() - start_time
            processing_fps = frame_count / elapsed_time
            cv2.putText(
                result,
                f"Processing FPS: {processing_fps:.1f}",
                (10, result.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Display result
        cv2.imshow("AUV Heading Estimation", result)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):  # Reset tracking
            estimator.tracks = []
            estimator.heading_history.clear()
            estimator.velocity_history.clear()
            print("Reset tracking")
        elif key == ord(" "):  # Pause/resume
            cv2.waitKey(0)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames")
    if len(estimator.heading_history) > 0:
        avg_heading = np.mean(list(estimator.heading_history))
        print(f"Average heading change: {avg_heading:.2f}°")


if __name__ == "__main__":
    main()

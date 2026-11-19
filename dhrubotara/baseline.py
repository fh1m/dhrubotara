"""Archived numerical kernels, unchanged; see origin and provenance.json.

The Torch/XFeat preprocessing and live telemetry adapters are not imported here.
"""
import cv2
import numpy as np

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

def estimate_guidance_from_optical_flow(ref_gray, curr_gray, prev_points=None):
    """
    Estimate camera guidance using Lucas-Kanade optical flow
    Returns: (dx, dy, dz_scale, confidence, found, tracked_points)
    """
    if ref_gray is None or curr_gray is None:
        return 0, 0, 1.0, 0.0, False, None

    if prev_points is None or len(prev_points) < 10:
        feature_params = dict(
            maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7
        )
        prev_points = cv2.goodFeaturesToTrack(ref_gray, mask=None, **feature_params)

        if prev_points is None:
            return 0, 0, 1.0, 0.0, False, None

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    try:
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            ref_gray, curr_gray, prev_points, None, **lk_params
        )

        if next_points is None:
            return 0, 0, 1.0, 0.0, False, None

        good_prev = prev_points[status == 1]
        good_next = next_points[status == 1]

        if len(good_prev) < 5:
            return 0, 0, 1.0, 0.0, False, None

        displacement = good_next - good_prev
        dx = np.median(displacement[:, 0])
        dy = np.median(displacement[:, 1])

        flow_magnitude = np.linalg.norm(displacement, axis=1)
        avg_flow = np.median(flow_magnitude)

        if avg_flow > 2.0:
            scale = 1.0 + (avg_flow - 2.0) / 50.0
        else:
            scale = 1.0

        flow_std = np.std(flow_magnitude)
        consistency = 1.0 / (1.0 + flow_std / 10.0)
        confidence = min(100, (len(good_prev) / 200.0) * 100 * consistency)

        return -dx, -dy, scale, confidence, True, good_next.reshape(-1, 1, 2)

    except Exception as e:
        return 0, 0, 1.0, 0.0, False, None

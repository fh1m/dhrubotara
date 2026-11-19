#!/usr/bin/env python3
"""
POC: Feature-based heading estimation + Hough-line based orientation cue.
Usage:
  python poc_homography_with_hough.py --camera 0
  python poc_homography_with_hough.py --input sample.mp4 --scale 0.6 --use_clahe
"""

import cv2
import numpy as np
import argparse
import math
import time


def normalize_angle_deg(a):
    """Normalize angle in degrees to [-180, 180)"""
    a = (a + 180) % 360 - 180
    return a


def circular_mean_deg(angles_deg, weights=None):
    """Compute circular mean of angles (degrees). Returns degrees in [-180,180)."""
    if len(angles_deg) == 0:
        return None
    a = np.deg2rad(angles_deg)
    if weights is None:
        sx = np.mean(np.cos(a))
        sy = np.mean(np.sin(a))
    else:
        w = np.asarray(weights, dtype=float)
        sx = np.sum(np.cos(a) * w) / (np.sum(w) + 1e-9)
        sy = np.sum(np.sin(a) * w) / (np.sum(w) + 1e-9)
    mean_ang = math.degrees(math.atan2(sy, sx))
    return normalize_angle_deg(mean_ang)


def extract_hough_angles(lines):
    """Given lines from HoughLinesP (x1,y1,x2,y2), return array of angles (deg) of each segment."""
    if lines is None or len(lines) == 0:
        return np.array([])
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        ang = math.degrees(math.atan2(dy, dx))  # [-180,180)
        angles.append(ang)
    return np.array(angles)


def dominant_angle_and_confidence(angles_deg, bandwidth_deg=10.0):
    """
    Compute dominant angle (circular mean of angles within the densest bin) and a confidence:
      - confidence = fraction of angles within +/- bandwidth_deg of dominant angle.
    """
    if angles_deg.size == 0:
        return None, 0.0
    # compute circular mean as initial guess
    mean_ang = circular_mean_deg(angles_deg)
    # find how many are within bandwidth
    diffs = np.vectorize(normalize_angle_deg)(angles_deg - mean_ang)
    within = np.abs(diffs) <= bandwidth_deg
    if np.sum(within) == 0:
        # fallback: use global circular mean with low confidence
        return mean_ang, 0.1
    dom_mean = circular_mean_deg(angles_deg[within])
    conf = float(np.sum(within)) / float(len(angles_deg))
    return dom_mean, conf


def yaw_from_rotation(R):
    """Yaw from 3x3 rotation R (degrees)."""
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def approx_intrinsics_from_shape(shape, fx_scale=0.9):
    h, w = shape[:2]
    fx = fy = fx_scale * max(w, h)
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def draw_compact_text(img, lines, origin=(8, 20), line_height=16):
    x0, y0 = origin
    hbox = line_height * len(lines) + 8
    cv2.rectangle(img, (x0 - 6, y0 - 18), (x0 + 380, y0 - 18 + hbox), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        y = y0 + i * line_height
        cv2.putText(
            img,
            line,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", help="video file (default: camera)", default=None)
    ap.add_argument("--camera", "-c", type=int, default=0)
    ap.add_argument(
        "--scale", type=float, default=1.0, help="downscale factor (0.5 saves cpu)"
    )
    ap.add_argument(
        "--use_clahe", action="store_true", help="apply CLAHE to improve contrast"
    )
    ap.add_argument(
        "--hough_thresh", type=int, default=60, help="Hough threshold (votes)"
    )
    ap.add_argument("--hough_minlen", type=int, default=40, help="min line length (px)")
    ap.add_argument("--hough_maxgap", type=int, default=8, help="max gap (px)")
    ap.add_argument("--ransac", type=float, default=8.0, help="RANSAC reproj threshold")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input if args.input else args.camera)
    if not cap.isOpened():
        print("ERROR: cannot open source")
        return

    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    ret, prev_frame = cap.read()
    if not ret:
        print("ERROR: empty first frame")
        return
    if args.scale != 1.0 and 0 < args.scale < 1.0:
        prev_frame = cv2.resize(
            prev_frame,
            (0, 0),
            fx=args.scale,
            fy=args.scale,
            interpolation=cv2.INTER_AREA,
        )
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if args.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        prev_gray_proc = clahe.apply(prev_gray)
    else:
        prev_gray_proc = prev_gray

    K = approx_intrinsics_from_shape(prev_frame.shape)

    last_heading_hom = None
    last_dom_angle = None
    last_combined = None

    frame_idx = 0
    last_time = time.time()
    window = "Hough+Homography Heading"
    can_show = True
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    except Exception:
        can_show = False

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.input:
                break
            else:
                time.sleep(0.01)
                continue
        if args.scale != 1.0 and 0 < args.scale < 1.0:
            frame_proc = cv2.resize(
                frame,
                (0, 0),
                fx=args.scale,
                fy=args.scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            frame_proc = frame
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        gray_proc = clahe.apply(gray) if args.use_clahe else gray

        # ---------- Homography-based (feature) ----------
        kp1, des1 = orb.detectAndCompute(prev_gray_proc, None)
        kp2, des2 = orb.detectAndCompute(gray_proc, None)

        hom_heading = None
        hom_conf = 0.0
        matches_count = 0
        inliers = 0

        if des1 is not None and des2 is not None and len(des1) >= 8 and len(des2) >= 8:
            knn = bf.knnMatch(des1, des2, k=2)
            good = []
            ratio = 0.75
            for m_n in knn:
                if len(m_n) < 2:
                    continue
                m, n = m_n
                if m.distance < ratio * n.distance:
                    good.append(m)
            matches_count = len(good)
            if len(good) >= 8:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, args.ransac)
                if H is not None and mask is not None and int(mask.sum()) >= 4:
                    try:
                        Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
                        # choose a decomposition preferring positive normal z
                        chosen = 0
                        for i, nv in enumerate(Ns):
                            if nv[2] > 0:
                                chosen = i
                                break
                        R = Rs[chosen]
                        hom_heading = yaw_from_rotation(R)
                        inliers = int(mask.sum())
                        hom_conf = inliers / max(1, len(good))
                        last_heading_hom = hom_heading
                    except Exception as e:
                        if args.debug:
                            print("Decomp err:", e)
                else:
                    # fallback to affine if homography fails
                    A, inlier_aff = cv2.estimateAffinePartial2D(
                        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=args.ransac
                    )
                    if A is not None:
                        # A is 2x3: [a b tx; c d ty]
                        a, b, tx = A[0]
                        c, d, ty = A[1]
                        Rmat = np.array([[a, b], [c, d]])
                        U, _, Vt = np.linalg.svd(Rmat)
                        Rn = U @ Vt
                        hom_heading = math.degrees(math.atan2(Rn[1, 0], Rn[0, 0]))
                        if inlier_aff is not None:
                            inliers = int(inlier_aff.sum())
                            hom_conf = inliers / max(1, len(good))
                        last_heading_hom = hom_heading

        # ---------- Hough lines ----------
        # Preprocessing: blur + Canny
        blur = cv2.GaussianBlur(gray_proc, (5, 5), 1.0)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=args.hough_thresh,
            minLineLength=args.hough_minlen,
            maxLineGap=args.hough_maxgap,
        )
        angles = extract_hough_angles(lines)  # array of angles in degrees

        dom_angle = None
        angle_conf = 0.0
        if angles.size > 0:
            dom_angle, angle_conf = dominant_angle_and_confidence(
                angles, bandwidth_deg=12.0
            )
            last_dom_angle = dom_angle

        # ---------- Combine / fuse ----------
        # We have:
        #  - hom_heading (degrees) with hom_conf in [0,1]
        #  - dom_angle (degrees) with angle_conf in [0,1] representing absolute orientation
        # We want a camera yaw (heading) — homography gives camera rotation directly, while line angle gives dominant scene line orientation.
        # For a downward camera where lines are fixed to world, camera yaw change ≈ negative change in dominant line angle.
        fused_heading = None
        if hom_heading is not None and dom_angle is not None:
            # convert dom_angle (scene line orientation) to implied camera yaw:
            # if previous dom angle exists, delta = normalize(dom_angle - prev_dom_angle), implied yaw change = -delta
            # but we don't maintain prev_dom here, so we interpret dom_angle as absolute dominant orientation; if known mapping from line orientation->north exists, you'd map accordingly.
            # For simplicity, fuse homography yaw & dom_angle with a heuristic: prefer homography when hom_conf higher, else use dom_angle as proxy.
            # Use weights:
            w_h = hom_conf + 1e-3
            w_l = angle_conf + 1e-3
            # convert line orientation to "camera heading" by negating (camera rotation causes opposite apparent scene rotation)
            # where we interpret dom_angle as world-aligned axis; camera heading candidate = normalize(-dom_angle)
            candidate_line_heading = normalize_angle_deg(-dom_angle)
            # weighted circular mean between hom_heading and candidate_line_heading
            a1 = np.deg2rad(hom_heading)
            a2 = np.deg2rad(candidate_line_heading)
            sx = w_h * math.cos(a1) + w_l * math.cos(a2)
            sy = w_h * math.sin(a1) + w_l * math.sin(a2)
            fused_heading = normalize_angle_deg(math.degrees(math.atan2(sy, sx)))
            last_combined = fused_heading
        elif hom_heading is not None:
            fused_heading = hom_heading
            last_combined = fused_heading
        elif dom_angle is not None:
            fused_heading = normalize_angle_deg(-dom_angle)
            last_combined = fused_heading

        # ---------- Visualization ----------
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # draw Hough lines in blue
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(vis, (x1, y1), (x2, y2), (200, 100, 20), 1)
        # draw keypoints (small)
        if "kp2" in locals() and kp2:
            for k in kp2[:120]:
                x, y = map(int, k.pt)
                cv2.circle(vis, (x, y), 1, (120, 200, 120), -1)
        # draw matches lightly if we computed 'good'
        if "good" in locals() and len(good) > 0:
            drawN = min(80, len(good))
            for i, m in enumerate(good[:drawN]):
                p2 = tuple(map(int, kp2[m.trainIdx].pt))
                color = (0, 255, 0)
                if (
                    "mask" in locals()
                    and mask is not None
                    and i < len(mask)
                    and mask[i] == 0
                ):
                    color = (0, 0, 255)
                cv2.circle(vis, p2, 2, color, -1)

        # text overlay
        lines_txt = [
            f"Method: hom_conf={hom_conf:.2f} ang_conf={angle_conf:.2f}",
            f"Homog yaw: {last_heading_hom:.1f} deg"
            if last_heading_hom is not None
            else "Homog yaw: N/A",
            f"Hough dom angle: {last_dom_angle:.1f} deg"
            if last_dom_angle is not None
            else "Hough dom angle: N/A",
            f"Fused heading: {last_combined:.1f} deg"
            if last_combined is not None
            else "Fused heading: N/A",
        ]
        vis = draw_compact_text(vis, lines_txt, origin=(10, 22))

        # draw small compass top-right if fused heading exists
        if last_combined is not None:
            Hh, Ww = vis.shape[:2]
            cx = Ww - 60
            cy = 60
            rad = 34
            cv2.circle(vis, (cx, cy), rad, (120, 120, 120), 1)
            ang = math.radians(last_combined - 90)
            ex = int(cx + rad * 0.78 * math.cos(ang))
            ey = int(cy + rad * 0.78 * math.sin(ang))
            cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 200, 200), 2, tipLength=0.3)
            cv2.putText(
                vis,
                f"{last_combined:.0f}deg",
                (cx - 28, cy + rad + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 220, 220),
                1,
            )

        # FPS show
        if frame_idx % 10 == 0:
            now = time.time()
            fps = 10.0 / max(1e-6, now - last_time)
            last_time = now
            cv2.putText(
                vis,
                f"FPS:{fps:.1f}",
                (10, vis.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )

        if can_show:
            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = ord("q")

        if frame_idx % 30 == 0:
            print(
                f"[Frame {frame_idx}] matches={matches_count} inliers={inliers} hom_yaw={last_heading_hom} dom_angle={last_dom_angle} fused={last_combined}"
            )

        if key == ord("q"):
            break
        elif key == ord("p") or key == ord(" "):
            if can_show:
                cv2.waitKey(0)

        # shift prev -> current
        prev_gray = gray.copy()
        prev_gray_proc = gray_proc.copy()
        # store for next homography usage
        prev_gray_proc = gray_proc
        prev_frame = frame_proc
        prev_gray = gray.copy()
        prev_gray_proc = gray_proc

        frame_idx += 1

    cap.release()
    if can_show:
        cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

"""Replay archived image guidance on a file; no camera, telemetry or model downloads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import cv2
import numpy as np

from .baseline import estimate_camera_guidance, estimate_guidance_from_optical_flow


def match_reference(reference, current, detector, matcher):
    """ORB is an explicit CPU baseline, not a substitute claimed to be XFeat."""
    key_ref, desc_ref = reference
    key_now, desc_now = detector.detectAndCompute(current, None)
    if desc_ref is None or desc_now is None or len(desc_ref) < 2:
        return None, None
    pairs = matcher.knnMatch(desc_now, desc_ref, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 8:
        return None, None
    fixed = np.float32([key_ref[m.trainIdx].pt for m in good])
    moving = np.float32([key_now[m.queryIdx].pt for m in good])
    return fixed, moving


def replay(video: Path, output: Path, every: int = 5, max_frames: int | None = None):
    if not video.is_file():
        raise ValueError("Supply an existing video file; camera indices and network streams are not accepted.")
    if every < 1 or (max_frames is not None and max_frames < 2):
        raise ValueError("--every must be positive; --max-frames must be at least two.")
    capture = cv2.VideoCapture(str(video.resolve()))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode {video}")
    output.mkdir(parents=True, exist_ok=True)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Video has no usable frame rate; a time axis cannot be assigned.")
    cv2.setRNGSeed(0)
    orb = cv2.ORB_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    reference = previous = None
    rows = []
    samples = []
    index = -1
    first_shape = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            if max_frames is not None and index >= max_frames:
                break
            if index % every:
                continue
            if first_shape is None:
                first_shape = frame.shape[:2]
            if frame.shape[:2] != first_shape:
                raise ValueError("Frame dimensions changed during replay.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if reference is None:
                reference = orb.detectAndCompute(gray, None)
                previous = gray
                cv2.imwrite(str(output / 'reference.jpg'), frame)
                continue
            fixed, moving = match_reference(reference, gray, orb, matcher)
            guidance = (0.0, 0.0, 1.0, 0.0, False) if fixed is None else estimate_camera_guidance(fixed, moving)
            flow = estimate_guidance_from_optical_flow(previous, gray)
            row = {
                'frame': index, 'time_s': index / fps,
                'reference_matches': 0 if fixed is None else len(fixed),
                'homography_valid': bool(guidance[4]),
                'homography_tx_px': float(guidance[0]) if guidance[4] else None,
                'homography_ty_px': float(guidance[1]) if guidance[4] else None,
                'homography_scale': float(guidance[2]) if guidance[4] else None,
                'homography_inlier_pct': float(guidance[3]) if guidance[4] else None,
                'flow_valid': bool(flow[4]),
                'flow_dx_px': float(flow[0]) if flow[4] else None,
                'flow_dy_px': float(flow[1]) if flow[4] else None,
                'archived_flow_scale': float(flow[2]) if flow[4] else None,
                'flow_score_pct': float(flow[3]) if flow[4] else None,
            }
            rows.append(row)
            if len(samples) < 6 and index >= (len(samples) + 1) * 120:
                canvas = frame.copy()
                if fixed is not None:
                    _, mask = cv2.findHomography(moving, fixed, cv2.RANSAC, 5.0)
                    if mask is not None:
                        for p, q, valid in zip(moving, fixed, mask.ravel()):
                            if valid:
                                cv2.line(canvas, tuple(np.int32(p)), tuple(np.int32(q)), (180, 210, 80), 1)
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 52), (22, 26, 30), -1)
                cv2.putText(canvas, f'{index / fps:.2f}s | ORB reference matches: {row["reference_matches"]}', (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)
                cv2.putText(canvas, 'Image registration only. No metric pose or depth.', (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 195, 230), 1)
                samples.append(canvas)
            previous = gray
    finally:
        capture.release()
    if not rows:
        raise ValueError("Video did not provide two sampled frames.")
    with (output / 'frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with video.open('rb') as stream:
        input_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    summary = {
        'input_file': video.name,
        'input_sha256': input_sha256,
        'fps': fps, 'sample_every_frames': every,
        'frame_size': [first_shape[1], first_shape[0]],
        'evaluated_pairs': len(rows),
        'homography_valid_pairs': sum(r['homography_valid'] for r in rows),
        'flow_valid_pairs': sum(r['flow_valid'] for r in rows),
        'feature_backend': 'OpenCV ORB; not XFeat',
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__},
        'reference_policy': 'First sampled frame for homography; previous sampled frame for optical flow',
        'ground_truth': None,
        'interpretation': 'Registration success is not navigation accuracy; archived scales are dimensionless heuristics.',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if samples:
        blank = np.zeros_like(samples[0])
        while len(samples) % 2:
            samples.append(blank)
        sheet = np.vstack([np.hstack(samples[i:i + 2]) for i in range(0, len(samples), 2)])
        cv2.imwrite(str(output / 'replay.jpg'), sheet)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--output', default=Path('results/replay'), type=Path)
    parser.add_argument('--every', default=5, type=int)
    parser.add_argument('--max-frames', type=int)
    args = parser.parse_args()
    try:
        summary = replay(args.video, args.output, args.every, args.max_frames)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

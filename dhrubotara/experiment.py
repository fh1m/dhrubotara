"""Controlled image warps, not simulated physical camera ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .baseline import estimate_camera_guidance, estimate_guidance_from_optical_flow


def controlled_cases():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (360, 480), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    x, y = np.meshgrid(np.linspace(60, 420, 9), np.linspace(60, 300, 7))
    reference_points = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    cases = []
    for tx in (0, 5, 10, 20):
        affine = np.array([[1, 0, tx], [0, 1, 0]], dtype=np.float32)
        current = cv2.warpAffine(image, affine, (480, 360), borderMode=cv2.BORDER_REFLECT)
        moving_points = cv2.transform(reference_points[None], affine)[0]
        homography = estimate_camera_guidance(reference_points, moving_points)
        flow = estimate_guidance_from_optical_flow(image, current)
        cases.append({
            'known_image_translation_x_px': tx,
            'known_image_scale': 1.0,
            'homography_valid': bool(homography[4]),
            'homography_tx_px': float(homography[0]),
            'homography_scale': float(homography[2]),
            'flow_valid': bool(flow[4]),
            'flow_dx_px': float(flow[0]),
            'archived_flow_scale': float(flow[2]),
            'flow_score_pct': float(flow[3]),
        })
    flat = np.zeros_like(image)
    missing = estimate_guidance_from_optical_flow(flat, flat)
    return {
        'input': 'Seeded 480x360 texture; exact 2-D translations; no physical camera model',
        'seed': 42,
        'cases': cases,
        'textureless_input': {'valid': bool(missing[4]), 'score_pct': float(missing[3])},
        'interpretation': 'Lateral image translation alone raises the archived flow scale. That number is not an independent depth measurement.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/controlled.json'))
    args = parser.parse_args()
    cv2.setRNGSeed(0)
    result = controlled_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from dhrubotara.baseline import estimate_camera_guidance, estimate_guidance_from_optical_flow
from dhrubotara.experiment import controlled_cases
from dhrubotara.replay import replay


class GuidanceBoundaries(unittest.TestCase):
    def test_reference_mapping_reverses_an_exact_image_translation(self):
        x, y = np.meshgrid(np.arange(20, 220, 30), np.arange(20, 180, 30))
        fixed = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
        current = fixed + np.array([13, -7], dtype=np.float32)
        dx, dy, scale, _, valid = estimate_camera_guidance(fixed, current)
        self.assertTrue(valid)
        self.assertAlmostEqual(dx, -13, places=4)
        self.assertAlmostEqual(dy, 7, places=4)
        self.assertAlmostEqual(scale, 1, places=4)

    def test_lateral_motion_exposes_the_archived_non_depth_scale(self):
        cases = controlled_cases()['cases']
        translated = next(case for case in cases if case['known_image_translation_x_px'] == 10)
        self.assertAlmostEqual(translated['homography_scale'], 1, places=4)
        self.assertAlmostEqual(translated['flow_dx_px'], -10, delta=0.2)
        self.assertGreater(translated['archived_flow_scale'], 1.1)
        self.assertGreater(translated['flow_score_pct'], 90)

    def test_blank_image_does_not_become_a_valid_observation(self):
        blank = np.zeros((120, 160), dtype=np.uint8)
        _, _, _, score, valid, points = estimate_guidance_from_optical_flow(blank, blank)
        self.assertFalse(valid)
        self.assertEqual(score, 0)
        self.assertIsNone(points)

    def test_file_replay_keeps_failed_estimates_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'blank.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 120))
            self.assertTrue(writer.isOpened(), 'MJPG encoding is required for the recorded-input regression')
            for _ in range(3):
                writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
            writer.release()
            summary = replay(video, root / 'out', every=1)
            self.assertEqual(summary['evaluated_pairs'], 2)
            self.assertEqual(summary['flow_valid_pairs'], 0)
            self.assertEqual(summary['homography_valid_pairs'], 0)
            with (root / 'out' / 'frames.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(row['flow_dx_px'] == '' and row['homography_tx_px'] == '' for row in rows))
            self.assertIsNone(json.loads((root / 'out' / 'summary.json').read_text())['ground_truth'])

    def test_non_file_input_is_rejected_before_video_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'existing video file'):
                replay(Path(directory) / 'camera-zero-does-not-exist', Path(directory) / 'out')


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np

from streamtimelens.observer.signals import LiteSignalObserver


def zero_flow(previous, current):
    return np.zeros((*current.shape, 2), dtype=np.float32)


def translation_flow(previous, current):
    magnitude = float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))) / 20.0)
    flow = np.zeros((*current.shape, 2), dtype=np.float32)
    flow[..., 0] = magnitude
    return flow


class LiteVisualSignalsTest(unittest.TestCase):
    def test_static_translation_hard_cut_and_gradient_have_expected_direction(self):
        observer = LiteSignalObserver(
            hard_cut_hsv_threshold=0.6, hard_cut_ssim_threshold=0.5,
            flow_backend=translation_flow,
        )
        base = np.zeros((32, 32, 3), dtype=np.uint8)
        base[8:20, 4:12] = (240, 40, 20)
        initial = observer.observe(base, 0)
        static = observer.observe(base.copy(), 1)
        translated = np.roll(base, 8, axis=1)
        motion = observer.observe(translated, 2)
        cut = observer.observe(np.full_like(base, (20, 230, 80)), 3)
        gradient = observer.observe(np.full_like(base, (20, 220, 80)), 4)
        self.assertIn("initial_frame", initial.statuses)
        self.assertEqual(static.hsv_distance, 0)
        self.assertAlmostEqual(static.ssim_change, 0)
        self.assertGreater(motion.flow_mean, static.flow_mean)
        self.assertGreater(motion.ssim_change, static.ssim_change)
        self.assertTrue(cut.hard_cut)
        self.assertLess(gradient.hsv_distance, cut.hsv_distance)

    def test_missing_black_and_resolution_change_are_explicit(self):
        observer = LiteSignalObserver(flow_backend=zero_flow)
        missing = observer.observe(None, 0)
        black = observer.observe(np.zeros((8, 8, 3), dtype=np.uint8), 1)
        resized = observer.observe(np.zeros((12, 10, 3), dtype=np.uint8), 2)
        self.assertEqual(missing.statuses, ("missing_frame",))
        self.assertIn("black_frame", black.statuses)
        self.assertIn("initial_frame", black.statuses)
        self.assertIn("resolution_change", resized.statuses)
        self.assertIsNotNone(resized.hsv_distance)

    def test_unavailable_farneback_is_reported_not_silently_zeroed(self):
        def unavailable(_previous, _current):
            raise RuntimeError("opencv_unavailable")

        observer = LiteSignalObserver(flow_backend=unavailable)
        frame = np.ones((8, 8, 3), dtype=np.uint8) * 100
        observer.observe(frame, 0)
        result = observer.observe(frame, 1)
        self.assertIn("opencv_unavailable", result.statuses)
        self.assertIsNone(result.flow_mean)


if __name__ == "__main__":
    unittest.main()

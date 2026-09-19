import json
import math
import unittest

from streamtimelens.observer.normalizer import OnlineNormalizer


class OnlineNormalizerTest(unittest.TestCase):
    def test_current_score_uses_only_past_state_and_warmup_threshold(self):
        normalizer = OnlineNormalizer(
            alpha=0.5, warmup_samples=2, epsilon=1e-6,
            fixed_thresholds={"motion": 0.5}, z_threshold=2,
        )
        first = normalizer.observe_one("motion", 0.4)
        second = normalizer.observe_one("motion", 0.6)
        third = normalizer.observe_one("motion", 1.0)
        self.assertFalse(first.active)
        self.assertTrue(second.active)
        self.assertTrue(third.ready)
        self.assertAlmostEqual(third.reference_mean, 0.5)
        self.assertAlmostEqual(third.reference_variance, 0.01)
        self.assertAlmostEqual(third.z_score, 0.5 / math.sqrt(0.010001))

    def test_constant_sequence_has_no_nan(self):
        normalizer = OnlineNormalizer(alpha=0.1, warmup_samples=1, epsilon=1e-5)
        values = [normalizer.observe_one("constant", 3.0) for _ in range(20)]
        self.assertTrue(all(item.z_score is None or math.isfinite(item.z_score) for item in values))
        self.assertTrue(all(item.z_score == 0 for item in values[1:]))

    def test_snapshot_restore_is_continuous_and_json_roundtrips(self):
        original = OnlineNormalizer(alpha=0.2, warmup_samples=2, fixed_thresholds={"x": 3})
        for value in (1, 2, 4):
            original.observe_one("x", value)
        restored = OnlineNormalizer.restore(json.loads(json.dumps(original.snapshot())))
        left = original.observe_one("x", 8)
        right = restored.observe_one("x", 8)
        self.assertEqual(left, right)
        self.assertEqual(original.snapshot(), restored.snapshot())


if __name__ == "__main__":
    unittest.main()

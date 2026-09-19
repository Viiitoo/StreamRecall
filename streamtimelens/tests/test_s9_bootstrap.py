import unittest

from streamtimelens.evaluation.bootstrap import (
    BootstrapConfig, PairedMetricObservation, paired_video_bootstrap,
)


def observation(method, video, sample, value, budget=1024):
    return PairedMetricObservation(method, budget, video, sample, value)


class PairedBootstrapTest(unittest.TestCase):
    def test_video_level_bootstrap_is_seeded_and_saves_config_not_draws(self):
        rows = [
            observation("a", "v1", "q1", .8), observation("b", "v1", "q1", .5),
            observation("a", "v1", "q2", .6), observation("b", "v1", "q2", .5),
            observation("a", "v2", "q3", .4), observation("b", "v2", "q3", .5),
        ]
        config = BootstrapConfig(seed=7, resamples=10_000)
        first = paired_video_bootstrap(rows, method_a="a", method_b="b", config=config)
        second = paired_video_bootstrap(rows, method_a="a", method_b="b", config=config)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["delta_a_minus_b"], .05)
        self.assertEqual(first["configuration"]["resamples"], 10_000)
        self.assertNotIn("draws", first)
        self.assertLessEqual(first["ci_low"], first["delta_a_minus_b"])
        self.assertGreaterEqual(first["ci_high"], first["delta_a_minus_b"])

    def test_budget_and_pair_mismatch_fail(self):
        with self.assertRaisesRegex(ValueError, "same byte budget"):
            paired_video_bootstrap([
                observation("a", "v", "q", 1, 1), observation("b", "v", "q", 1, 2),
            ], method_a="a", method_b="b")
        with self.assertRaisesRegex(ValueError, "not paired"):
            paired_video_bootstrap([
                observation("a", "v", "q1", 1), observation("b", "v", "q2", 1),
            ], method_a="a", method_b="b")


if __name__ == "__main__":
    unittest.main()

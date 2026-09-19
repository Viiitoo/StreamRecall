import unittest

from streamtimelens.evaluation.long_video import (
    LongVideoConfig, MemoryAgingObservation, memory_aging_report,
    select_long_video_configs,
)


HASH = "a" * 64


class LongVideoExtensionTest(unittest.TestCase):
    def test_exactly_two_pareto_full_and_two_baselines_are_selected(self):
        rows = [
            LongVideoConfig("f1", "full", HASH, .9, True, True),
            LongVideoConfig("f2", "full", HASH, .8, True, True),
            LongVideoConfig("f3", "full", HASH, .7, True, False),
            LongVideoConfig("b1", "semantic", HASH, .85, False, True),
            LongVideoConfig("b2", "oasis", HASH, .75, False, True),
            LongVideoConfig("b3", "uniform", HASH, .65, False, True),
        ]
        selected = select_long_video_configs(rows)
        self.assertEqual(selected["full_config_ids"], ["f1", "f2"])
        self.assertEqual(selected["baseline_config_ids"], ["b1", "b2"])
        self.assertEqual(len(selected["selected"]), 4)
        self.assertEqual(selected["selection_source"], "frozen_independent_dev_only")

    def test_memory_aging_has_explicit_empty_and_length_bins(self):
        report = memory_aging_report([
            MemoryAgingObservation("f", "v", 600, .5, .6, True),
            MemoryAgingObservation("f", "v", 600, .4, .5, False, "fixed_0.25"),
        ])
        self.assertEqual(report["length_bins"]["short:<60s"]["count"], 0)
        self.assertEqual(report["length_bins"]["long:>=300s"]["video_count"], 1)
        self.assertEqual(report["fixed_0.25"]["count"], 1)

    def test_invalid_memory_aging_observation_fails(self):
        with self.assertRaisesRegex(ValueError, "rates"):
            MemoryAgingObservation("f", "v", 10, 2, .5, True)

    def test_insufficient_frozen_choices_fail(self):
        with self.assertRaisesRegex(ValueError, "two Pareto"):
            select_long_video_configs([
                LongVideoConfig("f", "full", HASH, 1, True, True),
                LongVideoConfig("b", "base", HASH, 1, False, True),
            ])


if __name__ == "__main__":
    unittest.main()

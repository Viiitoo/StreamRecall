import unittest

from streamtimelens.evaluation.groups import GroupedVTGExample, evaluate_grouped_vtg
from streamtimelens.evaluation.metrics import VTGMetricExample


def row(query_id, cohort, *, eligible=True, lag="[0,0.1)", duration=30, gt=(0, 2)):
    return GroupedVTGExample(
        VTGMetricExample(query_id, "v" + query_id, gt, gt, "ok"),
        cohort, eligible, lag, duration,
    )


class GroupMetricsTest(unittest.TestCase):
    def test_cohort_lag_and_length_tables_include_counts_and_empty_groups(self):
        tables = evaluate_grouped_vtg([
            row("1", "natural", duration=30, gt=(0, 2)),
            row("2", "fixed_0.25", duration=100, gt=(0, 10)),
            row("3", "natural", eligible=False, duration=400, gt=(0, 30)),
        ])
        self.assertEqual(tables["cohort"]["natural"]["count"], 1)
        self.assertEqual(tables["cohort"]["fixed_0.25"]["video_count"], 1)
        self.assertEqual(tables["lag"]["[0.5,1)"]["count"], 0)
        self.assertEqual(tables["video_length"]["long:>=300s"]["count"], 0)
        self.assertEqual(tables["event_length"]["medium:[5,20)s"]["count"], 1)

    def test_invalid_duration_bins_fail(self):
        with self.assertRaisesRegex(ValueError, "boundaries"):
            evaluate_grouped_vtg([], video_boundaries_s=(60, 10))
        with self.assertRaisesRegex(ValueError, "cohort"):
            GroupedVTGExample(
                VTGMetricExample("q", "v", (0, 1), None, "not_found"),
                "bad", True, "[0,0.1)", 10,
            )


if __name__ == "__main__":
    unittest.main()

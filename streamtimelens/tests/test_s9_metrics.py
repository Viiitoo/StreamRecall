import unittest

from timelens.utils import iou as official_iou

from streamtimelens.evaluation.metrics import (
    VTGMetricExample, evaluate_vtg, temporal_iou,
)


class OfficialMetricsTest(unittest.TestCase):
    def test_iou_and_aggregate_match_official_definition(self):
        spans = [((1, 4), (2, 5)), ((0, 2), (0, 2)), ((1, 2), (4, 5))]
        for gt, predicted in spans:
            self.assertAlmostEqual(temporal_iou(gt, predicted), official_iou(gt, predicted))
        metrics = evaluate_vtg([
            VTGMetricExample("q1", "v1", (0, 2), (0, 2), "ok"),
            VTGMetricExample("q2", "v1", (0, 2), (0, 1), "fallback"),
            VTGMetricExample("q3", "v2", (0, 2), None, "not_found"),
        ])
        self.assertAlmostEqual(metrics.miou, 50.0)
        self.assertAlmostEqual(metrics.recall_at_05, 100 * 2 / 3)
        self.assertAlmostEqual(metrics.recall_at_07, 100 / 3)
        self.assertEqual(metrics.invalid_or_not_found_count, 1)
        self.assertEqual(metrics.video_count, 2)
        self.assertAlmostEqual(metrics.end_signed_error_s, -0.5)

    def test_invalid_not_found_and_empty_are_zero(self):
        metrics = evaluate_vtg([
            VTGMetricExample("q", "v", (1, 2), (1, 2), "error"),
        ])
        self.assertEqual(metrics.miou, 0)
        self.assertIsNone(metrics.start_abs_error_s)
        self.assertEqual(evaluate_vtg([]).count, 0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_vtg([
                VTGMetricExample("q", "v", (1, 2), None, "not_found"),
                VTGMetricExample("q", "v", (1, 2), None, "not_found"),
            ])


if __name__ == "__main__":
    unittest.main()

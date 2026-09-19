import unittest

from streamtimelens.evaluation.runner import evaluate_predictions
from streamtimelens.protocol.arrival import build_arrival_plan


class EvaluationJoinTest(unittest.TestCase):
    def test_verified_plan_gt_is_joined_outside_prediction_values(self):
        plan = build_arrival_plan([
            {"video_id": "v", "query_id": "q", "query": "door", "gt_span": [.5, 1], "duration": 4},
        ], [.25, .5])
        predictions = [
            {"query_id": "q", "video_id": "v", "rho_q": .25,
             "status": "not_found", "span": None},
            {"query_id": "q", "video_id": "v", "rho_q": .5,
             "status": "ok", "span": [.5, 1]},
        ]
        result = evaluate_predictions(plan, predictions)
        self.assertEqual(result["overall_natural"]["count"], 2)
        self.assertEqual(result["overall_natural"]["miou"], 50)
        self.assertEqual(result["groups"]["cohort"]["fixed_0.25"]["count"], 1)

    def test_missing_prediction_and_gt_leak_fail(self):
        plan = build_arrival_plan([
            {"video_id": "v", "query_id": "q", "query": "door", "gt_span": [1, 2], "duration": 4},
        ], [.5])
        with self.assertRaisesRegex(ValueError, "missing"):
            evaluate_predictions(plan, [])
        with self.assertRaisesRegex(ValueError, "ground truth"):
            evaluate_predictions(plan, [{
                "query_id": "q", "video_id": "v", "rho_q": .5,
                "status": "ok", "span": [1, 2], "gt_span": [1, 2],
            }])


if __name__ == "__main__":
    unittest.main()

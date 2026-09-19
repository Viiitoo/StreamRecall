import unittest

from streamtimelens.protocol.oracle import (
    OraclePrediction, OracleRunner, build_oracle_examples, evaluate_oracle_predictions,
    evaluate_oracle_conditions, oracle_gate_decision,
)
from streamtimelens.refiner.oracle_inference import TimeLensOracleRunner


class OracleDatasetTest(unittest.TestCase):
    def setUp(self):
        self.record = {
            "query_id": "q1", "video_id": "v1", "query": "opens door", "gt_span": [20, 30],
            "duration_s": 60, "fps": 10, "total_num_frames": 600,
        }

    def test_expands_gt_margins_frame_counts_and_strategies(self):
        examples = build_oracle_examples([self.record])
        self.assertEqual(len(examples), 12)
        for example in examples:
            self.assertEqual(len(example.frame_indices), example.k_frames)
            self.assertEqual(len(set(example.frame_indices)), example.k_frames)
            self.assertEqual(tuple(sorted(example.timestamps_s)), example.timestamps_s)
            self.assertGreaterEqual(example.timestamps_s[0], example.crop_span[0])
            self.assertLessEqual(example.timestamps_s[-1], example.crop_span[1])
        margin_two = next(item for item in examples if item.margin_s == 2 and item.k_frames == 16 and item.strategy == "uniform")
        self.assertEqual(margin_two.crop_span, (18.0, 32.0))

    def test_boundary_heavy_places_more_samples_near_edges(self):
        examples = build_oracle_examples([self.record], margins_s=[10], frame_counts=[16])
        boundary = next(item for item in examples if item.strategy == "boundary-heavy")
        middle = sum(1 for timestamp in boundary.timestamps_s if 17.5 < timestamp < 32.5)
        self.assertEqual(middle, 0)

    def test_metrics_and_p0_gate(self):
        examples = build_oracle_examples([self.record], margins_s=[2], frame_counts=[16], strategies=["uniform"])
        example = examples[0]
        predictions = [
            OraclePrediction(example.example_id, "dense_crop", (20, 30), "ok"),
            OraclePrediction(example.example_id, "sparse_adapter", (20.1, 30.1), "ok"),
            OraclePrediction(example.example_id, "offline_timelens", (19, 31), "ok"),
        ]
        result = evaluate_oracle_predictions(examples, predictions)
        self.assertAlmostEqual(result["metrics"]["dense_crop"]["miou"], 1.0)  # type: ignore[index]
        self.assertTrue(result["p0_gate"]["passed"])  # type: ignore[index]
        self.assertEqual(oracle_gate_decision(result)["decision"], "go")

    def test_gate_decision_distinguishes_fix_and_stop(self):
        examples = build_oracle_examples(
            [self.record], margins_s=[2], frame_counts=[16], strategies=["uniform"],
        )
        example = examples[0]
        fix = evaluate_oracle_predictions(examples, [
            OraclePrediction(example.example_id, "dense_crop", example.gt_span, "ok"),
            OraclePrediction(example.example_id, "sparse_adapter", (21.5, 31.5), "ok"),
            OraclePrediction(example.example_id, "offline_timelens", example.gt_span, "ok"),
        ])
        self.assertEqual(oracle_gate_decision(fix)["decision"], "fix")
        stop = evaluate_oracle_predictions(examples, [
            OraclePrediction(example.example_id, "dense_crop", None, "failed"),
            OraclePrediction(example.example_id, "sparse_adapter", None, "failed"),
            OraclePrediction(example.example_id, "offline_timelens", None, "failed"),
        ])
        self.assertEqual(oracle_gate_decision(stop)["decision"], "stop")

    def test_gate_uses_strictest_sampling_interval(self):
        examples = build_oracle_examples(
            [self.record], margins_s=[2, 10], frame_counts=[16], strategies=["uniform"],
        )
        predictions = []
        for example in examples:
            predictions.extend([
                OraclePrediction(example.example_id, "dense_crop", example.gt_span, "ok"),
                OraclePrediction(example.example_id, "sparse_adapter", (21.2, 31.2), "ok"),
                OraclePrediction(example.example_id, "offline_timelens", example.gt_span, "ok"),
            ])
        result = evaluate_oracle_predictions(examples, predictions)
        self.assertAlmostEqual(
            result["p0_gate"]["bias_tolerance_s"], min(item.sampling_interval_s for item in examples),  # type: ignore[index]
        )
        self.assertFalse(result["p0_gate"]["passed"])  # type: ignore[index]

    def test_gate_rejects_missing_and_duplicate_prediction_matrix_entries(self):
        example = build_oracle_examples(
            [self.record], margins_s=[2], frame_counts=[16], strategies=["uniform"],
        )[0]
        dense = OraclePrediction(example.example_id, "dense_crop", example.gt_span, "ok")
        sparse = OraclePrediction(example.example_id, "sparse_adapter", example.gt_span, "ok")
        offline = OraclePrediction(example.example_id, "offline_timelens", example.gt_span, "ok")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            evaluate_oracle_predictions([example], [dense, sparse])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_oracle_predictions([example], [dense, sparse, offline, dense])

    def test_condition_metrics_preserve_every_sampling_axis(self):
        examples = build_oracle_examples(
            [self.record], margins_s=[2, 5], frame_counts=[16],
        )
        predictions = [
            OraclePrediction(example.example_id, mode, example.gt_span, "ok")
            for example in examples for mode in (
                "dense_crop", "sparse_adapter", "offline_timelens",
            )
        ]
        conditions = evaluate_oracle_conditions(examples, predictions)
        self.assertEqual(len(conditions), 4)
        self.assertIn("margin_2s__frames_16__uniform", conditions)
        self.assertTrue(all(row["p0_gate"]["passed"] for row in conditions.values()))

    def test_runner_invokes_all_three_modes(self):
        example = build_oracle_examples([self.record], margins_s=[2], frame_counts=[16], strategies=["uniform"])[0]
        calls = []
        def inference(mode):
            def run(item):
                calls.append(mode)
                return OraclePrediction(item.example_id, mode, item.gt_span, "ok")
            return run
        predictions, metrics = OracleRunner(
            inference("dense_crop"), inference("sparse_adapter"), inference("offline_timelens")
        ).run([example])
        self.assertEqual(calls, ["dense_crop", "sparse_adapter", "offline_timelens"])
        self.assertEqual(len(predictions), 3)
        self.assertTrue(metrics["p0_gate"]["passed"])  # type: ignore[index]

    def test_concrete_runner_uses_shared_service_and_global_sparse_frames(self):
        example = build_oracle_examples([self.record], margins_s=[2], frame_counts=[16], strategies=["uniform"])[0]
        class Service:
            def __init__(self):
                self.calls = 0
            def generate(self, messages, videos, max_new_tokens):
                self.calls += 1
                metadata = videos[0][1]
                self.assert_global = metadata["frames_indices"][::2]
                return "The event happens in 20 - 30 seconds"
        service = Service()
        loader_calls = []
        def load_frames(item, indices):
            import numpy as np
            loader_calls.append(tuple(indices))
            return [np.zeros((2, 2, 3), dtype=np.uint8) for _ in indices]
        predictions, metrics = TimeLensOracleRunner(service, load_frames).run([example])  # type: ignore[arg-type]
        self.assertEqual(service.calls, 3)
        self.assertEqual(len(loader_calls), 3)
        self.assertEqual(len(predictions), 3)
        self.assertTrue(metrics["p0_gate"]["passed"])  # type: ignore[index]

    def test_concrete_runner_resumes_without_repeating_completed_modes(self):
        examples = build_oracle_examples(
            [self.record], margins_s=[2], frame_counts=[16], strategies=["uniform"],
        )
        example = examples[0]

        class Service:
            def __init__(self):
                self.calls = 0

            def generate(self, messages, videos, max_new_tokens):
                self.calls += 1
                return "The event happens in 20 - 30 seconds"

        service = Service()

        def load_frames(item, indices):
            import numpy as np
            return [np.zeros((2, 2, 3), dtype=np.uint8) for _ in indices]

        existing = [OraclePrediction(
            example.example_id, "dense_crop", example.gt_span, "ok", "cached",
        )]
        checkpointed = []
        predictions, metrics = TimeLensOracleRunner(service, load_frames).run(
            examples, existing_predictions=existing,
            prediction_callback=checkpointed.append,
        )  # type: ignore[arg-type]
        self.assertEqual(service.calls, 2)
        self.assertEqual(len(checkpointed), 2)
        self.assertEqual(len(predictions), 3)
        self.assertTrue(metrics["p0_gate"]["passed"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from streamtimelens.evaluation.writer_feasibility import (
    evaluate_writer_records, feasibility_decision, semantic_reservoir_oracle_coverage,
)


def row(model, chunk, span=(0, 2), gpu=1.0):
    return {
        "model": model, "chunk_id": chunk, "segment": [0, 2],
        "sampled_timestamps": [0, 1, 2], "gt_spans": [[0.5, 1.5]], "gpu_s": gpu,
        "raw_output": json.dumps({"segment": [0, 2], "events": [{
            "summary": "event", "actors": [], "actions": [], "objects": [], "scene": "room",
            "span": list(span), "phase": "complete", "visual_support": [0, 1, 2],
        }]}),
    }


class WriterFeasibilityTest(unittest.TestCase):
    def test_paired_metrics_and_stop_decision(self):
        records = [row("timelens", "a", (0.5, 1.5)), row("timelens", "b", (0.5, 1.5)),
                   row("qwen", "a", (0, 2), 2), row("qwen", "b", (0, 2), 2)]
        metrics = evaluate_writer_records(records, expected_chunks=2)
        self.assertEqual(metrics["models"]["timelens"]["json_valid_rate"], 1)
        self.assertGreater(metrics["models"]["timelens"]["mean_gt_coverage_iou"],
                           metrics["models"]["qwen"]["mean_gt_coverage_iou"])
        self.assertEqual(feasibility_decision(metrics)["selected_writer"], "timelens")
        self.assertEqual(feasibility_decision(metrics, semantic_oracle_coverage=1)["decision"], "go")

    def test_mismatched_or_duplicate_fixed_chunks_fail(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            evaluate_writer_records([row("a", "one"), row("b", "two")], expected_chunks=None)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_writer_records([row("a", "one"), row("a", "one")], expected_chunks=None)

    def test_semantic_reservoir_oracle_uses_only_fixed_sample_endpoints(self):
        records = [row("a", "one"), row("b", "one")]
        self.assertAlmostEqual(semantic_reservoir_oracle_coverage(records), 0.5)
        conflicting = row("b", "one")
        conflicting["sampled_timestamps"] = [0, 0.5, 2]
        with self.assertRaisesRegex(ValueError, "disagree"):
            semantic_reservoir_oracle_coverage([row("a", "one"), conflicting])


if __name__ == "__main__":
    unittest.main()

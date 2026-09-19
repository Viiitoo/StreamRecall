import unittest

from streamtimelens.evaluation.diagnostics import (
    PipelineDiagnosticExample, evaluate_pipeline_diagnostics,
)


class DiagnosticsTest(unittest.TestCase):
    def test_writer_retrieval_boundary_refinement_and_trace_rates(self):
        cards = ({
            "id": "hit", "t_start": 1, "t_end": 3,
            "boundary_cache": {"left_hit": True, "right_hit": False},
        }, {
            "id": "miss", "t_start": 8, "t_end": 9, "boundary_cache": {},
        })
        result = evaluate_pipeline_diagnostics([
            PipelineDiagnosticExample(
                "q", (1, 3), cards, ("miss", "hit"), ((8, 9), (1, 3)),
                (0, 4), (1, 3),
            ),
        ], [
            {"kind": "writer_called", "parse_status": "valid"},
            {"kind": "writer_called", "parse_status": "fallback"},
            {"kind": "nodes_merged"}, {"kind": "payload_evicted"},
        ])
        self.assertEqual(result["writer_coverage_mean_iou"], 1)
        self.assertEqual(result["card_recall"]["1"], 0)
        self.assertEqual(result["card_recall"]["5"], 1)
        self.assertEqual(result["candidate_recall"]["5"], 1)
        self.assertEqual(result["left_boundary_hit_rate"], 1)
        self.assertEqual(result["right_boundary_hit_rate"], 0)
        self.assertGreater(result["post_minus_pre_refinement_mean_iou"], 0)
        self.assertEqual(result["writer_parse_rates"]["valid"], .5)
        self.assertEqual(result["merge_per_writer_call"], .5)

    def test_empty_and_duplicate_examples_are_explicit(self):
        result = evaluate_pipeline_diagnostics([], [])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["writer_calls"], 0)
        example = PipelineDiagnosticExample("q", (0, 1), (), (), (), None, None)
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_pipeline_diagnostics([example, example], [])


if __name__ == "__main__":
    unittest.main()

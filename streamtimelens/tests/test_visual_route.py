import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from streamtimelens.config import (
    BudgetConfig, MethodConfig, ModelConfig, ProtocolConfig, ResolvedConfig,
)
from streamtimelens.evaluation.dev_selection import VisualDevRun, select_visual_dev_configs
from streamtimelens.evaluation.diagnostics import (
    VisualDiagnosticExample, evaluate_visual_diagnostics,
)
from streamtimelens.evaluation.formal_report import FormalRunSummary
from streamtimelens.evaluation.formal_visual import (
    FormalVisualObservation, canonical_formal_rho, frozen_visual_grids, paired_formal_bootstrap,
    select_activitynet_configs, summarize_formal_group,
    summarize_long_video_observations, timelens_bench_records,
)
from streamtimelens.evaluation.freeze import freeze_visual_configuration
from streamtimelens.evaluation.visual_audit import VisualAuditRun, audit_visual_cache_runs
from streamtimelens.evaluation.visual_refiner_gate import build_visual_refiner_gate
from streamtimelens.evaluation.visual_v2 import (
    oracle_candidate_for_condition, select_candidate_margin_configs,
    summarize_candidate_margin,
)
from streamtimelens.observer.clip_encoder import FrozenCLIPEncoder, serialize_embedding
from streamtimelens.protocol.snapshot import SnapshotWriter
from streamtimelens.protocol.types import Budget, FramePacket, VideoMeta
from streamtimelens.retrieval.frame_candidates import build_frame_candidates, coarse_prediction
from streamtimelens.stream.baseline_runner import BaselineIngestConfig, RawBaselineIngestor


class Backend:
    def encode_images(self, images):
        return np.asarray([[1.0, float(index + 1)] for index, _ in enumerate(images)])

    def encode_texts(self, texts):
        return np.asarray([[1.0, 1.0] for _ in texts])


def jpeg(value):
    output = BytesIO()
    Image.new("RGB", (8, 8), (value, 0, 0)).save(output, format="JPEG")
    return output.getvalue()


def resolved(method="semantic_reservoir"):
    return ResolvedConfig(
        ProtocolConfig(), BudgetConfig(1024 * 1024, 1),
        MethodConfig(method, "fixed_clip_0_5fps", "none", "clip", "visual"),
        ModelConfig(clip_model="clip"),
    )


def visual_parameters(enabled=False):
    return {
        "writer": "none", "clip_model": "clip", "clip_revision": "rev",
        "clip_sha256": "a" * 64, "jpeg_short_edge": 224, "jpeg_quality": 85,
        "embedding_precision": "fp16", "anchor_fraction": .25, "top_k": 8,
        "expand_neighbors": 1, "merge_gap_s": 4, "coarse_margin_s": 10,
        "timelens_enabled": enabled,
    }


class VisualRouteTest(unittest.TestCase):
    def test_formal_preparation_separates_gt_from_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "v.mp4").write_bytes(b"video")
            videos, queries, annotations, arrival = timelens_bench_records(
                "bench", {"v": {"duration": 10, "spans": [[1, 2]], "queries": ["event"]}},
                root,
            )
        self.assertEqual(videos[0]["video_id"], "v")
        self.assertNotIn("gt_span", queries[0])
        self.assertEqual(annotations[0]["gt_span"], [1.0, 2.0])
        self.assertTrue(arrival)
        self.assertEqual({row.rho_q for row in arrival}, {.25, .5, .75, 1.0})

    def test_formal_rho_uses_canonical_arrival_ratio(self):
        self.assertEqual(canonical_formal_rho(.7500000000000001, .75), .75)
        with self.assertRaisesRegex(ValueError, "does not match plan"):
            canonical_formal_rho(.74, .75)

    def test_formal_preparation_clamps_only_integer_endpoint_rounding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "v.mp4").write_bytes(b"video")
            _, _, annotations, _ = timelens_bench_records(
                "bench", {"v": {"duration": 9.9, "spans": [[8, 10]], "queries": ["event"]}},
                root,
            )
            self.assertEqual(annotations[0]["gt_span"], [8.0, 9.9])
            with self.assertRaisesRegex(ValueError, "invalid TimeLens-Bench query"):
                timelens_bench_records(
                    "bench", {"v": {"duration": 9.1, "spans": [[8, 11]], "queries": ["event"]}},
                    root,
                )

    def test_frozen_formal_grids_retain_exact_config_ids(self):
        frozen = freeze_visual_configuration(
            {"winner": resolved("uniform_raw")},
            {"passed": True, "selected_config_ids": ["winner"]},
            {"passed": True}, {"passed": False, "decision": "disable"},
            {"winner": visual_parameters(False)},
        )
        grids = frozen_visual_grids(frozen)
        self.assertEqual(grids[1024 * 1024][0]["config_id"], "winner")

    def test_formal_group_reports_metrics_diagnostics_and_bootstrap(self):
        row = FormalVisualObservation(
            "cfg", "bench", 262144, "v", "q", 1.0, "natural", "[0,0.1)",
            10.0, (1.0, 3.0), (1.0, 3.0), "ok", (2.0,), ((1.0, 3.0),),
            100, 1.0, 2.0, .1, .2,
        )
        summary = summarize_formal_group([row])
        self.assertEqual(summary["miou"], 100.0)
        self.assertEqual(summary["candidate_recall"]["1"], 1.0)
        self.assertEqual(summary["bootstrap_miou"]["ci_low"], 1.0)
        paired = paired_formal_bootstrap([row], [row])
        self.assertEqual(paired["delta_miou"], 0.0)
        self.assertEqual(paired["ci_high"], 0.0)

    def test_v6_selects_one_v5_winner_per_frozen_budget(self):
        first = freeze_visual_configuration(
            {"a": resolved("uniform_raw")},
            {"passed": True, "selected_config_ids": ["a"]},
            {"passed": True}, {"passed": False, "decision": "disable"},
            {"a": visual_parameters(False)},
        )
        second_config = resolved("uniform_raw")
        second_config = ResolvedConfig(
            second_config.protocol, BudgetConfig(262144, 1), second_config.method,
            second_config.model,
        )
        second = freeze_visual_configuration(
            {"b": second_config}, {"passed": True, "selected_config_ids": ["b"]},
            {"passed": True}, {"passed": False, "decision": "disable"},
            {"b": visual_parameters(False)},
        )
        first["configs"].update(second["configs"])
        selection = select_activitynet_configs(first, [
            {"config_id": "a", "budget_bytes": 1048576, "cohort": "natural", "miou": 50},
            {"config_id": "b", "budget_bytes": 262144, "cohort": "natural", "miou": 40},
        ])
        self.assertEqual({row["config_id"] for row in selection["selected"]}, {"a", "b"})

    def test_v6_reports_length_aging_and_amortized_cost(self):
        row = FormalVisualObservation(
            "cfg", "activitynet", 262144, "v", "q", 1.0, "natural", "[0,0.1)",
            350.0, (1.0, 3.0), (1.0, 3.0), "ok", (2.0,), ((1.0, 3.0),),
            100, 2.0, 3.0, .1, .2, True, 4, 2,
        )
        report = summarize_long_video_observations([row])["configs"]["cfg"]
        self.assertEqual(report["length_bins"]["long:>=300s"]["miou"], 1.0)
        self.assertEqual(report["natural"]["earliest_anchor_retention_rate"], 1.0)
        self.assertAlmostEqual(report["natural"]["amortized_gpu_s_per_query"], 1.1)

    def test_coarse_margin_and_ranked_frame_diagnostics_are_bounded(self):
        rows = {
            "a.jpg": {"timestamp_s": 1, "frame_index": 1,
                      "clip_embedding": serialize_embedding([1, 0])},
            "b.jpg": {"timestamp_s": 4, "frame_index": 4,
                      "clip_embedding": serialize_embedding([.8, .2])},
        }
        candidates = build_frame_candidates([1, 0], rows, top_k=2, merge_gap_s=4)
        self.assertEqual([item.rank for item in candidates[0].retrieved_frames], [1, 2])
        prediction = coarse_prediction(candidates, margin_s=10, upper_bound_s=5)
        self.assertEqual(prediction.span, (0.0, 5))

    def test_candidate_margin_expands_refiner_window_without_changing_default(self):
        rows = {
            "a.jpg": {"timestamp_s": 5, "frame_index": 5,
                      "clip_embedding": serialize_embedding([1, 0])},
        }
        default = build_frame_candidates([1, 0], rows, upper_bound_s=10)
        expanded = build_frame_candidates(
            [1, 0], rows, candidate_margin_s=2, upper_bound_s=6,
        )
        self.assertAlmostEqual(default[0].end_s - default[0].start_s, .001)
        self.assertEqual(expanded[0].span, (3.0, 6.0))
        with self.assertRaisesRegex(ValueError, "parameters"):
            build_frame_candidates([1, 0], rows, candidate_margin_s=-1)

    def test_v2_candidate_margin_requires_recall_gate_and_positive_paired_ci(self):
        rows = [{
            "sample_id": "q@1", "video_id": "v", "budget_bytes": 1048576,
            "gt_span": (3, 5), "upper_bound_s": 10, "candidate_spans": ((4, 4.001),),
        }]
        summary = summarize_candidate_margin(rows, margin_s=1, coarse_margin_s=0)
        self.assertEqual(summary["candidate_recall_at_5"], 1)
        self.assertEqual(summary["baseline_candidate_recall_at_5"], 0)
        selected = select_candidate_margin_configs([{
            **summary, "config_id": "v2", "budget_bytes": 1048576,
        }])
        self.assertTrue(selected["passed"])
        self.assertEqual(selected["selected_config_ids"], ["v2"])

    def test_v2_oracle_gate_uses_recorded_sampling_tolerance(self):
        candidate = oracle_candidate_for_condition({"conditions": {"margin2": {
            "metrics": {
                "dense_crop": {"miou": .4},
                "sparse_adapter": {
                    "miou": .42, "signed_start_bias_s": 1, "signed_end_bias_s": -1,
                },
            },
            "p0_gate": {"passed": False, "bias_tolerance_s": .25},
        }}}, "margin2")
        self.assertEqual(candidate["sampling_interval_s"], .25)
        self.assertFalse(candidate["source_p0_gate_passed"])
        with self.assertRaisesRegex(ValueError, "sampling tolerance"):
            oracle_candidate_for_condition({"conditions": {"bad": {
                "metrics": {
                    "dense_crop": {"miou": .4},
                    "sparse_adapter": {
                        "miou": .4, "signed_start_bias_s": 0, "signed_end_bias_s": 0,
                    },
                },
                "p0_gate": {"bias_tolerance_s": 0},
            }}}, "bad")

    def test_visual_diagnostics_report_retrieval_before_localization(self):
        result = evaluate_visual_diagnostics([VisualDiagnosticExample(
            "q", (2, 4),
            ({"frame_ref": "a", "timestamp_s": 3, "rank": 1},),
            ((1, 5),), (1, 5), (2, 4),
        )])
        self.assertEqual(result["frame_recall"]["1"], 1)
        self.assertEqual(result["candidate_recall"]["1"], 1)
        self.assertGreater(result["post_minus_pre_refinement_mean_iou"], 0)

    def test_visual_selection_gate_and_uniform_fallback(self):
        common = dict(
            budget_bytes=1024 * 1024, frame_recall_at_1=.5, frame_recall_at_5=.8,
            frame_recall_at_8=.9, candidate_recall_at_1=.6,
            candidate_oracle_miou=.7, left_boundary_hit_rate=.8,
            right_boundary_hit_rate=.8, coarse_miou=.4, coarse_recall_at_1=.5,
            snapshot_bytes=100, ingest_gpu_s=1, query_gpu_s=.1,
            query_latency_s=.01, fixed_c025_by_rho={".25": .5, "1": .4},
        )
        uniform = VisualDevRun(
            "u", "uniform_raw", candidate_recall_at_5=.71, **common,
        )
        semantic = VisualDevRun(
            "s", "semantic_reservoir", candidate_recall_at_5=.72, **common,
            paired_bootstrap_vs_uniform={"coarse_iou": {"ci_low": .01}},
        )
        selection = select_visual_dev_configs([uniform, semantic])
        self.assertTrue(selection["passed"])
        self.assertEqual(selection["selected_method"], "semantic_reservoir")

    def test_refiner_gate_can_legitimately_disable(self):
        gate = build_visual_refiner_gate(
            {"dense_miou": .5, "sparse_miou": .44, "sampling_interval_s": 2,
             "start_signed_bias_s": 0, "end_signed_bias_s": 0},
            {"coarse_miou": .4, "refined_miou": .39, "count": 10,
             "valid_count": 8, "fallback_count": 2},
        )
        self.assertEqual(gate["decision"], "disable")
        self.assertAlmostEqual(gate["retrieved_candidate"]["fallback_rate"], .2)

    def test_v2_refiner_gate_freezes_explicit_candidate_margin(self):
        gate = build_visual_refiner_gate(
            {"dense_miou": .4, "sparse_miou": .42, "sampling_interval_s": 2,
             "start_signed_bias_s": 1, "end_signed_bias_s": -1},
            {"coarse_miou": .3, "refined_miou": .31, "count": 2,
             "valid_count": 2, "fallback_count": 0},
            candidate_margin_s=2, max_frames=16,
        )
        self.assertEqual(gate["decision"], "enable")
        self.assertEqual(gate["frozen_setting"]["candidate_margin_s"], 2)

    def test_visual_freeze_accepts_disable_without_writer_gate(self):
        frozen = freeze_visual_configuration(
            {"winner": resolved()}, {"passed": True, "selected_config_ids": ["winner"]},
            {"passed": True}, {"passed": False, "decision": "disable"},
            {"winner": visual_parameters(False)},
        )
        self.assertIsNone(frozen["selected_writer"])
        self.assertEqual(frozen["refiner_decision"], "disable")
        self.assertEqual(len(frozen["configs"]["winner"]["visual_sha256"]), 64)

    def test_formal_summary_allows_writer_metric_to_be_absent(self):
        row = FormalRunSummary(
            method="semantic_reservoir", budget_bytes=262144, rho_q=1,
            cohort="natural", count=1, miou=.5, recall_at_03=.5,
            recall_at_05=.5, recall_at_07=.5, candidate_recall_at_5=.8,
            both_boundary_hit_rate=.6, refinement_delta_iou=0, gpu_s=1,
            realtime_throughput=2, snapshot_bytes=100,
            snapshot_integrity_passed=True, budget_audit_passed=True,
            frame_recall_at_5=.9,
        )
        self.assertIsNone(row.writer_coverage)

    def test_visual_audit_checks_four_snapshots_and_byte_traces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meta = VideoMeta("v", 4, 1, 5)
            encoder = FrozenCLIPEncoder("fixture", backend=Backend(), batch_size=2, visual_fps=1)
            ingestor = RawBaselineIngestor(
                meta, Budget(256 * 1024, 1), BaselineIngestConfig("uniform_raw", 4),
                clip_encoder=encoder,
            )
            packets = [FramePacket(index, index, jpeg(index * 30), 8, 8, video_id="v") for index in range(5)]
            ingestor.run(
                packets, [1, 2, 3, 4], SnapshotWriter(root),
                config={"writer": "none"}, snapshot_prefix="v",
            )
            (root / "v" / "ingest.trace.jsonl").write_text(ingestor.trace.to_jsonl())
            (root / "provenance.json").write_text("{}")
            (root / "git_revision.txt").write_text("revision\n")
            (root / "config.resolved.json").write_text(json.dumps({"writer": "none"}))
            report = audit_visual_cache_runs([
                VisualAuditRun("uniform_raw", 256 * 1024, "v", str(root)),
            ])
        self.assertTrue(report["passed"], report["runs"][0]["failures"])


if __name__ == "__main__":
    unittest.main()

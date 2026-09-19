import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from streamtimelens.config import (
    HybridBrainConfig,
    HybridFallbackConfig,
    HybridRetrievalConfig,
    HybridSnapshotSourceConfig,
    HybridV3Config,
    read_hybrid_v3_config,
)
from streamtimelens.evaluation.hybrid_v3 import summarize_hybrid_v3
from streamtimelens.model_identity import verify_registered_model
from streamtimelens.observer.clip_encoder import serialize_embedding
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, VideoMeta
from streamtimelens.refiner.frame_allocator import allocate_candidate_frames
from streamtimelens.refiner.multicandidate_parse import parse_multicandidate_output
from streamtimelens.refiner.multicandidate_prompt import (
    SPARSE_MULTICANDIDATE_ADAPTIVE_VERSION,
    SPARSE_MULTICANDIDATE_KEEP_VERSION,
    SPARSE_MULTICANDIDATE_TEMPLATE_SHA256,
    build_multicandidate_prompt,
    candidate_labels,
)
from streamtimelens.retrieval.clip_timelens_hybrid import answer_hybrid_snapshot
from streamtimelens.retrieval.frame_candidates import (
    FrameCandidate,
    build_frame_candidates,
    coarse_prediction,
)
from streamtimelens.retrieval.multicandidate import select_diverse_candidates


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT.parent
FIXTURE = ROOT / "tests" / "fixtures" / "frozen_visual_v2_golden.json"


def jpeg(value):
    stream = BytesIO()
    Image.new("RGB", (8, 8), (value, 0, 0)).save(stream, format="JPEG")
    return stream.getvalue()


def make_snapshot(root, timestamps, *, t_q=None, embeddings=None):
    duration = max(30.0, max(timestamps, default=0) + 1)
    t_q = duration if t_q is None else t_q
    frames = []
    metadata = {}
    for index, timestamp in enumerate(timestamps):
        ref = f"f{index:02d}.jpg"
        frames.append((ref, jpeg(index)))
        metadata[ref] = {
            "blob": ref, "timestamp_s": float(timestamp),
            "frame_index": int(timestamp),
        }
        if embeddings is not None:
            metadata[ref]["clip_embedding"] = serialize_embedding(embeddings[index])
    writer = SnapshotWriter(root, source_revision="test")
    manifest = writer.write(
        name="snapshot", t_q=t_q,
        meta=VideoMeta("video", duration, 1.0, int(duration) + 1),
        budget=Budget(1024 * 1024, 1), cards=[], raw_frames=frames,
        raw_metadata=metadata, writer_calls=0, config={"fixture": "hybrid-v3"},
        method="uniform_raw", pixel_only=False,
    )
    return SnapshotReader(root / "snapshot"), manifest


def hybrid_config(snapshot_hash, *, max_frames=16, accept_not_found=False):
    return HybridV3Config(
        1, "clip_timelens_hybrid_v3",
        HybridSnapshotSourceConfig("frozen", "a" * 64, snapshot_hash, True),
        HybridRetrievalConfig(
            "clip", "revision", "b" * 64, 32, 2.0, 0, 2.0, 0.0,
            6, .5, 2.0,
        ),
        HybridBrainConfig(
            "timelens", "timelens", "revision", "c" * 64,
            {
                "model_sha256": "d" * 64,
                "config_sha256": "e" * 64,
                "tokenizer_sha256": "f" * 64,
            },
            "timelens_multicandidate_v1", "sparse_multicandidate_v1",
            max_frames, 64, False, 1, 4096,
        ),
        HybridFallbackConfig("clip_coarse", "clip_coarse", accept_not_found),
    )


class Encoder:
    resource_records = ()

    def encode_text(self, query):
        return np.asarray([1.0, 0.0])

    def resource_dicts(self):
        return []


class Service:
    hashes = {
        "model_sha256": "d" * 64,
        "config_sha256": "e" * 64,
        "tokenizer_sha256": "f" * 64,
    }

    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error
        self.calls = 0
        self.last_call_stats = {}

    def generate(self, messages, videos, max_new_tokens):
        self.calls += 1
        if self.error:
            raise self.error
        self.last_call_stats = {
            "generated_tokens": 12, "peak_cuda_bytes": 100,
            "gpu_time_s": .2,
        }
        return self.answer


class MultiCandidateSelectionTest(unittest.TestCase):
    def test_temporal_nms_is_deterministic_and_trace_visible(self):
        candidates = [
            FrameCandidate("late", 10, 14, .8, ("c",), ("c",)),
            FrameCandidate("overlap", .5, 4.5, .9, ("b",), ("b",)),
            FrameCandidate("best", 0, 4, 1.0, ("a",), ("a",)),
        ]
        first = select_diverse_candidates(
            candidates, max_candidates=6, temporal_nms_iou=.5,
            min_cluster_separation_s=2,
        )
        second = select_diverse_candidates(
            reversed(candidates), max_candidates=6, temporal_nms_iou=.5,
            min_cluster_separation_s=2,
        )
        self.assertEqual([item.candidate_id for item in first.candidates], ["best", "late"])
        self.assertEqual(first, second)
        decision = next(item for item in first.decisions if item["candidate_id"] == "overlap")
        self.assertEqual(decision["reason"], "temporal_nms")
        self.assertEqual(decision["suppressed_by"], "best")

    def test_merge_gap_tolerance_absorbs_nominal_half_fps_rounding(self):
        rows = {
            f"f{index}.jpg": {
                "timestamp_s": timestamp, "frame_index": index,
                "clip_embedding": serialize_embedding([1, .01 * (index + 1)]),
            }
            for index, timestamp in enumerate((0, 2.002, 4.004, 6.006))
        }
        strict = build_frame_candidates(
            [1, 0], rows, top_k=4, merge_gap_s=2, expand_neighbors=0,
            candidate_margin_s=.1, upper_bound_s=8,
        )
        cadence_aware = build_frame_candidates(
            [1, 0], rows, top_k=4, merge_gap_s=2, merge_gap_tolerance_s=.5,
            expand_neighbors=0, candidate_margin_s=.1, upper_bound_s=8,
        )
        self.assertEqual(len(strict), 4)
        self.assertEqual(len(cadence_aware), 1)


class FrameAllocatorTest(unittest.TestCase):
    def test_zero_one_sixteen_and_hard_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot, _ = make_snapshot(Path(temporary), range(24))
            candidates = tuple(
                FrameCandidate(
                    f"c{index}", index * 4, index * 4 + 3, 1 - index / 10,
                    tuple(f"f{value:02d}.jpg" for value in range(index * 4, index * 4 + 4)),
                    (f"f{index * 4:02d}.jpg",),
                )
                for index in range(6)
            )
            empty = allocate_candidate_frames(candidates, snapshot, max_frames=0)
            self.assertEqual(empty.selected_frame_refs, ())
            one = allocate_candidate_frames(candidates, snapshot, max_frames=1)
            self.assertEqual(len(one.selected_frame_refs), 1)
            self.assertEqual(set(one.candidate_frame_refs), {"c0"})
            full = allocate_candidate_frames(candidates, snapshot, max_frames=16)
            self.assertEqual(len(full.selected_frame_refs), 16)
            self.assertEqual(
                full.allocation_reason["target_counts"],
                {"c0": 4, "c1": 4, "c2": 3, "c3": 3, "c4": 1, "c5": 1},
            )
            timestamps = [snapshot.read_frame_metadata()[ref]["timestamp_s"] for ref in full.selected_frame_refs]
            self.assertEqual(timestamps, sorted(timestamps))
            with self.assertRaisesRegex(ValueError, "hard limit"):
                allocate_candidate_frames(candidates, snapshot, max_frames=17)

    def test_rejects_future_and_unknown_snapshot_refs(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot, _ = make_snapshot(Path(temporary), [1, 10], t_q=5)
            future = FrameCandidate("future", 9, 11, 1, ("f01.jpg",), ("f01.jpg",))
            with self.assertRaisesRegex(ValueError, "future frame"):
                allocate_candidate_frames([future], snapshot, max_frames=1)
            unknown = FrameCandidate("unknown", 1, 2, 1, ("elsewhere.jpg",), ("elsewhere.jpg",))
            with self.assertRaisesRegex(ValueError, "outside snapshot"):
                allocate_candidate_frames([unknown], snapshot, max_frames=1)

    def test_boundary_anchored_v2_reserves_global_edges_and_maps_boundary_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot, _ = make_snapshot(Path(temporary), [0, 2, 4, 6])
            candidates = (
                FrameCandidate("left", 0, 3, 1, ("f01.jpg",), ("f01.jpg",)),
                FrameCandidate("right", 3, 6, .9, ("f02.jpg",), ("f02.jpg",)),
            )
            bundle = allocate_candidate_frames(
                candidates, snapshot, max_frames=4, strategy="boundary_anchored_v2",
            )
            self.assertEqual(bundle.selected_frame_refs, (
                "f00.jpg", "f01.jpg", "f02.jpg", "f03.jpg",
            ))
            self.assertIn("f00.jpg", bundle.candidate_frame_refs["left"])
            self.assertIn("f03.jpg", bundle.candidate_frame_refs["right"])
            self.assertEqual(
                bundle.allocation_reason["global_anchor_refs"], ["f00.jpg", "f03.jpg"],
            )


class HybridPromptParseTest(unittest.TestCase):
    def setUp(self):
        self.labels = {"C01": "one", "C02": "two"}
        self.spans = {"one": (1, 3), "two": (5, 8)}
        self.refs = {"one": ("a",), "two": ("b",)}

    def parse(self, answer):
        return parse_multicandidate_output(
            answer, t_q=10, candidate_labels=self.labels,
            candidate_spans=self.spans, candidate_frame_refs=self.refs,
            available_frame_refs=("a", "b"),
        )

    def test_prompt_hash_and_candidate_table_are_stable(self):
        candidate = FrameCandidate("one", 1, 3, .5, ("a",), ("a",))
        prompt = build_multicandidate_prompt(
            version="sparse_multicandidate_v1", query=" opens   door ",
            candidates=[candidate], candidate_frame_refs={"one": ("a",)},
            frame_metadata={"a": {"timestamp_s": 2}}, t_q=10,
        )
        self.assertEqual(
            SPARSE_MULTICANDIDATE_TEMPLATE_SHA256,
            "ae2fa3bc602616785e212769fd31b479ca008dd58772e6c6813cd649110315ef",
        )
        self.assertIn("C01 | coarse=1.000-3.000s", prompt.text)
        self.assertIn("frame_timestamps=2.000s", prompt.text)

    def test_keep_v2_prompt_parser_containment_and_label_permutation(self):
        candidates = [
            FrameCandidate("one", 1, 3, .5, ("a",), ("a",)),
            FrameCandidate("two", 5, 8, .4, ("b",), ("b",)),
        ]
        prompt = build_multicandidate_prompt(
            version=SPARSE_MULTICANDIDATE_KEEP_VERSION, query="event",
            candidates=candidates, candidate_frame_refs=self.refs,
            frame_metadata={"a": {"timestamp_s": 2}, "b": {"timestamp_s": 6}},
            t_q=10, candidate_label_order="reverse_score_v1",
        )
        self.assertIn("KEEP Cxx", prompt.text)
        self.assertEqual(
            candidate_labels(candidates, order="reverse_score_v1"),
            {"C01": "two", "C02": "one"},
        )
        def parse(answer):
            return parse_multicandidate_output(
                answer, t_q=10, candidate_labels=self.labels,
                candidate_spans=self.spans, candidate_frame_refs=self.refs,
                available_frame_refs=("a", "b"),
                prompt_version=SPARSE_MULTICANDIDATE_KEEP_VERSION,
            )
        kept = parse("KEEP C02")
        self.assertEqual((kept.status, kept.span, kept.selected_candidate_id), (
            "keep", (5, 8), "two",
        ))
        self.assertEqual(parse(
            "REFINE C01; The event happens in 1.000 - 3.000 seconds"
        ).status, "ok")
        self.assertEqual(parse(
            "REFINE C01; The event happens in 0.999 - 2.000 seconds"
        ).status, "outside_candidate_envelope")
        self.assertEqual(self.parse("KEEP C01").status, "no_timestamp")

    def test_adaptive_v3_prompt_switches_on_query_visible_frame_count(self):
        candidate = FrameCandidate("one", 1, 3, .5, ("a",), ("a",))
        common = dict(
            version=SPARSE_MULTICANDIDATE_ADAPTIVE_VERSION, query="event",
            candidates=[candidate], candidate_frame_refs={"one": ("a",)},
            frame_metadata={"a": {"timestamp_s": 2}}, t_q=10,
            sparse_keep_max_frames=7,
        )
        sparse = build_multicandidate_prompt(**common, selected_frame_count=7)
        dense = build_multicandidate_prompt(**common, selected_frame_count=8)
        self.assertIn("SPARSE mode (7 <= 7)", sparse.text)
        self.assertIn("REFINE is forbidden", sparse.text)
        self.assertIn("DENSE mode (8 > 7)", dense.text)
        self.assertIn("KEEP is forbidden", dense.text)
        self.assertIn("REFINE; x - y seconds", dense.text)
        parsed = parse_multicandidate_output(
            "KEEP C01", t_q=10, candidate_labels={"C01": "one"},
            candidate_spans={"one": (1, 3)}, candidate_frame_refs={"one": ("a",)},
            available_frame_refs=("a",),
            prompt_version=SPARSE_MULTICANDIDATE_ADAPTIVE_VERSION,
        )
        self.assertEqual(parsed.status, "keep")
        refined = parse_multicandidate_output(
            "REFINE; 1.5 - 2.5 seconds", t_q=10,
            candidate_labels={"C01": "one"}, candidate_spans={"one": (1, 3)},
            candidate_frame_refs={"one": ("a",)}, available_frame_refs=("a",),
            prompt_version=SPARSE_MULTICANDIDATE_ADAPTIVE_VERSION,
        )
        self.assertEqual((refined.status, refined.span), ("ok", (1.5, 2.5)))
        compact = parse_multicandidate_output(
            "REFINE; 1.5-2.5 seconds", t_q=10,
            candidate_labels={"C01": "one"}, candidate_spans={"one": (1, 3)},
            candidate_frame_refs={"one": ("a",)}, available_frame_refs=("a",),
            prompt_version=SPARSE_MULTICANDIDATE_ADAPTIVE_VERSION,
        )
        self.assertEqual((compact.status, compact.span), ("ok", (1.5, 2.5)))

    def test_valid_not_found_and_typed_rejections(self):
        valid = self.parse("Candidate C02; The event happens in 5.5 - 7.0 seconds")
        self.assertEqual(valid.status, "ok")
        self.assertEqual(valid.selected_candidate_id, "two")
        self.assertEqual(self.parse("NOT_FOUND").status, "not_found")
        cases = {
            "The event happens in 1 - 2 seconds": "missing_candidate_id",
            "Candidate C01 and C02; The event happens in 1 - 2 seconds": "multiple_candidate_ids",
            "Candidate C09; The event happens in 1 - 2 seconds": "unknown_candidate_id",
            "Candidate C01; no time": "no_timestamp",
            "Candidate C01; The event happens in 1 - 2 seconds; 2 - 3 seconds": "multiple_spans",
            "Candidate C01; The event happens in 3 - 2 seconds": "invalid_order",
            "Candidate C02; The event happens in 5 - 11 seconds": "out_of_bounds",
            "Candidate C02; The event happens in 1 - 2 seconds": "no_candidate_overlap",
            "Candidate C01; The event happens in 1 - 2 seconds extra": "invalid_format",
        }
        for answer, expected in cases.items():
            with self.subTest(answer=answer):
                self.assertEqual(self.parse(answer).status, expected)
        invalid_refs = parse_multicandidate_output(
            "Candidate C01; The event happens in 1 - 2 seconds",
            t_q=10, candidate_labels=self.labels, candidate_spans=self.spans,
            candidate_frame_refs={"one": ("outside",)}, available_frame_refs=("a",),
        )
        self.assertEqual(invalid_refs.status, "invalid_frame_reference")


class HybridPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.snapshot, manifest = make_snapshot(
            Path(self.temporary.name), [1, 10],
            embeddings=([1, 0], [.9, .1]),
        )
        self.config = hybrid_config(manifest.config_hash)

    def tearDown(self):
        self.temporary.cleanup()

    def answer(self, service, config=None):
        return answer_hybrid_snapshot(
            query_id="q", query="event", snapshot=self.snapshot,
            encoder=Encoder(), service=service, config=config or self.config,
        )

    def test_timelens_can_select_non_top_clip_candidate(self):
        service = Service("Candidate C02; The event happens in 9.0 - 11.0 seconds")
        output = self.answer(service)
        self.assertEqual(service.calls, 1)
        self.assertEqual(output.final_prediction["span"], [9.0, 11.0])
        self.assertEqual(output.selected_candidate_id, output.candidate_ids[1])
        self.assertEqual(output.final_selection_reason, "timelens_multicandidate_valid")
        self.assertFalse(output.fallback_used)
        inference = next(item for item in output.query_trace if item["kind"] == "timelens_inference")
        self.assertEqual(inference["unique_input_frames"], 2)
        self.assertEqual(inference["generated_tokens"], 12)

    def test_keep_v2_preserves_selected_candidate_with_reversed_labels(self):
        config = replace(
            self.config,
            brain=replace(
                self.config.brain,
                readout="timelens_multicandidate_keep_v2",
                prompt_version="sparse_multicandidate_keep_v2",
                frame_allocator="boundary_anchored_v2",
                candidate_label_order="reverse_score_v1",
            ),
        )
        output = self.answer(Service("KEEP C01"), config)
        self.assertEqual(output.selected_candidate_id, output.candidate_ids[1])
        selected = next(
            item for item in output.candidates
            if item["candidate_id"] == output.selected_candidate_id
        )
        self.assertEqual(
            output.final_prediction["span"], [selected["start_s"], selected["end_s"]],
        )
        self.assertEqual(output.timelens_prediction["parse_status"], "keep")
        self.assertEqual(output.final_selection_reason, "timelens_multicandidate_keep")
        self.assertFalse(output.fallback_used)
        score_order = replace(
            config,
            brain=replace(config.brain, candidate_label_order="score_v1"),
        )
        control = self.answer(Service("KEEP C02"), score_order)
        self.assertEqual(output.candidates, control.candidates)
        self.assertEqual(output.selected_frame_refs, control.selected_frame_refs)
        self.assertEqual(output.selected_candidate_id, control.selected_candidate_id)

    def test_adaptive_v3_gate_rejects_sparse_refinement(self):
        config = replace(
            self.config,
            brain=replace(
                self.config.brain,
                readout="timelens_multicandidate_adaptive_v3",
                prompt_version="sparse_multicandidate_adaptive_v3",
                frame_allocator="boundary_anchored_v2",
                sparse_keep_max_frames=2,
            ),
        )
        output = self.answer(
            Service("REFINE; 1.500 - 2.500 seconds"),
            config,
        )
        selected = next(
            item for item in output.candidates
            if item["candidate_id"] == output.selected_candidate_id
        )
        self.assertEqual(output.timelens_prediction["parse_status"], "ok")
        self.assertEqual(output.timelens_prediction["span"], [1.5, 2.5])
        self.assertEqual(
            output.final_prediction["span"],
            [selected["start_s"], selected["end_s"]],
        )
        self.assertEqual(
            output.final_selection_reason,
            "timelens_refinement_rejected_sparse_gate_keep",
        )

    def test_adaptive_v3_gate_rejects_only_strong_dense_shrinkage(self):
        config = replace(
            self.config,
            brain=replace(
                self.config.brain,
                readout="timelens_multicandidate_adaptive_v3",
                prompt_version="sparse_multicandidate_adaptive_v3",
                frame_allocator="boundary_anchored_v2",
                sparse_keep_max_frames=0,
                min_refined_candidate_ratio=.8,
            ),
        )
        rejected = self.answer(Service("REFINE; 1.2 - 1.8 seconds"), config)
        self.assertEqual(
            rejected.final_selection_reason,
            "timelens_refinement_rejected_shrink_gate_keep",
        )
        accepted = self.answer(Service("REFINE; 0.2 - 2.8 seconds"), config)
        self.assertEqual(accepted.final_selection_reason, "timelens_multicandidate_valid")
        self.assertEqual(accepted.final_prediction["span"], [0.2, 2.8])

    def test_invalid_and_exception_preserve_exact_clip_fallback(self):
        for service in (Service("bad output"), Service(error=RuntimeError("oom"))):
            with self.subTest(service=service):
                output = self.answer(service)
                self.assertEqual(output.final_prediction, output.clip_coarse_prediction)
                self.assertTrue(output.fallback_used)
                self.assertEqual(service.calls, 1)

    def test_not_found_policy_and_zero_frame_budget(self):
        rejected = self.answer(Service("NOT_FOUND"))
        self.assertEqual(rejected.final_prediction, rejected.clip_coarse_prediction)
        accepted_config = replace(
            self.config,
            fallback=HybridFallbackConfig("clip_coarse", "clip_coarse", True),
        )
        accepted = self.answer(Service("NOT_FOUND"), accepted_config)
        self.assertEqual(accepted.final_prediction["status"], "NOT_FOUND")
        zero_config = replace(
            self.config, brain=replace(self.config.brain, max_unique_frames=0),
        )
        service = Service("unused")
        zero = self.answer(service, zero_config)
        self.assertEqual(service.calls, 0)
        self.assertEqual(zero.final_prediction, zero.clip_coarse_prediction)

    def test_snapshot_source_hash_is_enforced(self):
        bad = replace(
            self.config,
            snapshot_source=replace(self.config.snapshot_source, snapshot_config_sha256="d" * 64),
        )
        with self.assertRaisesRegex(ValueError, "snapshot config hash"):
            self.answer(Service("NOT_FOUND"), bad)


class HybridEvaluationAndRegressionTest(unittest.TestCase):
    def test_registered_model_identity_rehashes_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            artifact = model / "weights.bin"
            artifact.write_bytes(b"weights")
            file_sha = hashlib.sha256(b"weights").hexdigest()
            files = [{"bytes": 7, "path": "weights.bin", "sha256": file_sha}]
            content_sha = hashlib.sha256(json.dumps(
                files, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")).hexdigest()
            registry = root / "models.json"
            registry.write_text(json.dumps({"models": {"test": {
                "revision": "rev", "content_sha256": content_sha,
                "file_count": 1, "total_bytes": 7, "files": files,
            }}}), encoding="utf-8")
            identity = verify_registered_model(
                registry, model_key="test", model_root=model,
                revision="rev", content_sha256=content_sha, verify_files=True,
            )
            self.assertTrue(identity["files_verified"])
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "file changed"):
                verify_registered_model(
                    registry, model_key="test", model_root=model,
                    revision="rev", content_sha256=content_sha, verify_files=True,
                )

    def test_evaluation_is_past_only_paired_and_reports_reliability(self):
        predictions = [
            {
                "sample_id": "q@1.00", "query_id": "q", "video_id": "v", "rho_q": 1.0,
                "final_prediction": {"span": [2, 4], "status": "ok"},
                "clip_coarse_prediction": {"span": [1, 5], "status": "fallback"},
                "candidates": [{"candidate_id": "c", "start_s": 2, "end_s": 4}],
                "selected_candidate_id": "c", "fallback_used": False,
                "timelens_prediction": {
                    "parse_status": "ok", "status": "ok", "span": [2, 4],
                },
                "query_trace": [{"kind": "frame_allocation", "selected_frames": [
                    {"frame_ref": "f", "timestamp_s": 3, "frame_index": 3},
                ]}],
            },
            {
                "sample_id": "future@0.25", "query_id": "future", "video_id": "v", "rho_q": .25,
                "final_prediction": {"span": None, "status": "NOT_FOUND"},
                "clip_coarse_prediction": {"span": None, "status": "NOT_FOUND"},
            },
        ]
        annotations = [
            {"query_id": "q", "video_id": "v", "duration_s": 10, "gt_span": [2, 4]},
            {"query_id": "future", "video_id": "v", "duration": 10, "gt_span": [5, 6]},
        ]
        result = summarize_hybrid_v3(
            predictions, annotations, budget_bytes=1048576, bootstrap_resamples=10,
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["excluded_future_gt"], 1)
        self.assertEqual(result["hybrid"]["miou"], 100)
        self.assertEqual(
            result["stage_metrics"]["selected_candidate_or_fallback"]["miou"], 100,
        )
        self.assertEqual(
            result["stage_metrics"]["timelens_refined_or_fallback"]["miou"], 100,
        )
        self.assertEqual(result["stage_metrics"]["candidate_oracle"]["miou"], 100)
        self.assertEqual(result["candidate_recall_at_iou_05"]["6"], 1)
        self.assertEqual(result["parse_valid_rate"], 1)
        self.assertGreater(result["paired_miou"]["delta_a_minus_b"], 0)

    def test_evaluation_counts_keep_as_valid_and_reports_score_rank(self):
        predictions = [{
            "sample_id": "q@1.00", "query_id": "q", "video_id": "v", "rho_q": 1.0,
            "final_prediction": {"span": [2, 4], "status": "ok"},
            "clip_coarse_prediction": {"span": [0, 5], "status": "fallback"},
            "candidates": [
                {"candidate_id": "first", "start_s": 0, "end_s": 2},
                {"candidate_id": "kept", "start_s": 2, "end_s": 4},
            ],
            "selected_candidate_id": "kept", "fallback_used": False,
            "timelens_prediction": {
                "parse_status": "keep", "status": "ok", "span": [2, 4],
                "candidate_label": "C01",
            },
            "query_trace": [{"kind": "frame_allocation", "selected_frames": [
                {"frame_ref": "f", "timestamp_s": 3, "frame_index": 3},
            ]}],
        }]
        result = summarize_hybrid_v3(
            predictions,
            [{"query_id": "q", "video_id": "v", "duration_s": 5, "gt_span": [2, 4]}],
            budget_bytes=1048576, bootstrap_resamples=10,
        )
        self.assertEqual(result["parse_valid_rate"], 1)
        self.assertEqual(result["readout_diagnostics"]["keep_rate"], 1)
        self.assertEqual(
            result["readout_diagnostics"]["selected_candidate_score_rank_counts"],
            {"2": 1},
        )
        self.assertEqual(
            result["stage_metrics"]["timelens_refined_or_fallback"]["miou"], 100,
        )

    def test_config_and_frozen_v2_golden_are_unchanged(self):
        config = read_hybrid_v3_config(
            ROOT / "configs" / "exploration" / "clip_timelens_hybrid_v3.yaml"
        )
        self.assertEqual(config.brain.max_unique_frames, 16)
        self.assertEqual(config.brain.frame_allocator, "boundary_anchored_v2")
        self.assertEqual(config.retrieval.top_k, 6)
        self.assertEqual(config.retrieval.merge_gap_tolerance_s, .5)
        self.assertEqual(config.brain.sparse_keep_max_frames, 7)
        self.assertEqual(config.brain.min_refined_candidate_ratio, .8)
        reverse = read_hybrid_v3_config(
            ROOT / "configs" / "exploration" / "clip_timelens_hybrid_v3_reverse_labels.yaml"
        )
        regular_dict = config.canonical_dict()
        reverse_dict = reverse.canonical_dict()
        regular_dict["brain"]["candidate_label_order"] = "reverse_score_v1"
        self.assertEqual(regular_dict, reverse_dict)
        small = read_hybrid_v3_config(
            ROOT / "configs" / "exploration" / "clip_timelens_hybrid_v3_256k.yaml"
        )
        qwen = read_hybrid_v3_config(
            ROOT / "configs" / "exploration" / "clip_qwen_hybrid_v3.yaml"
        )
        self.assertIn("262144", small.snapshot_source.frozen_config_id)
        self.assertEqual(qwen.brain.model_registry_key, "qwen_writer")
        golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
        frozen_bytes = (ROOT / "configs" / "frozen_visual_v2.yaml").read_bytes()
        self.assertEqual(hashlib.sha256(frozen_bytes).hexdigest(), golden["frozen_config_sha256"])
        rows = {
            "a.jpg": {"timestamp_s": 1, "frame_index": 1, "clip_embedding": serialize_embedding([1, 0])},
            "b.jpg": {"timestamp_s": 4, "frame_index": 4, "clip_embedding": serialize_embedding([.8, .2])},
            "c.jpg": {"timestamp_s": 10, "frame_index": 10, "clip_embedding": serialize_embedding([.7, .3])},
        }
        candidates = build_frame_candidates(
            [1, 0], rows, top_k=3, merge_gap_s=2, expand_neighbors=1,
            candidate_margin_s=2, upper_bound_s=12,
        )
        coarse = coarse_prediction(candidates, margin_s=8, upper_bound_s=12)
        self.assertEqual([item.candidate_id for item in candidates], golden["candidate_ids"])
        self.assertEqual([list(item.span) for item in candidates], golden["candidate_spans"])
        self.assertEqual(list(coarse.span), golden["coarse_span"])
        self.assertEqual(list(coarse.evidence_ids), golden["coarse_evidence_ids"])

    def test_query_clis_offer_no_video_argument(self):
        for script in ("answer_hybrid_v3.py", "run_hybrid_v3_matrix.py"):
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--help"],
                check=True, capture_output=True, text=True,
            )
            self.assertNotIn("--video", completed.stdout)


if __name__ == "__main__":
    unittest.main()

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from streamtimelens.baselines.snag_adapt import (
    ProvenancePointGenerator,
    RankedSpan,
    SnAGAdaptConfig,
    SnAGAdaptReader,
    SnAGAdaptWriter,
    SnAGSnapshotReader,
    ranked_spans_close,
)
from streamtimelens.baselines.vendor.snag_model import SnAGTorchBackend
from streamtimelens.baselines.vendor.snag_upstream import (
    UPSTREAM_REVISION,
    load_precomputed_text_feature,
    verify_upstream_checkout,
)
from streamtimelens.config import resolve_config
from streamtimelens.evaluation.snag_parity import (
    compare_ranked_spans,
    compare_raw_traces,
    g0_gate_report,
)
from streamtimelens.protocol.types import VideoMeta


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parents[1] / "ref/snag_release"


def meta():
    return VideoMeta("video", 100, 10, 1000)


class Backend:
    def __init__(self):
        self.calls = 0

    def predict(self, features, items, query):
        self.calls += 1
        score = float(features[int(query), 0])
        item = items[int(query)]
        return [RankedSpan(item.t_start_s, item.t_end_s, score)]


class FakeSnAGModel:
    def encode_text(self, tokens, masks):
        return tokens, masks

    def encode_video(self, video, masks):
        return (video,), (masks,)

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks):
        del text, text_masks
        import torch
        length = fpn[0].shape[-1]
        logits = (torch.arange(length, dtype=torch.float32)[None],)
        offsets = (torch.ones((1, length, 2), dtype=torch.float32) * 0.5,)
        return logits, offsets, fpn_masks


class SnAGAdaptTest(unittest.TestCase):
    def test_repeated_gpu_spans_use_frozen_numeric_tolerance(self):
        left = (RankedSpan(3.35708117, 5.69853115, 0.17406566),)
        right = (RankedSpan(3.35708094, 5.69853163, 0.17406563),)
        self.assertTrue(ranked_spans_close(left, right))
        self.assertFalse(ranked_spans_close(left, (RankedSpan(3.3, 5.7, 0.17),)))

    def test_method_configs_load_through_strict_project_schema(self):
        for name, expected in (
            ("snag_adapt_full", "snag-adapt-input-full"),
            ("snag_adapt_pooled", "snag-adapt-pooled-B"),
        ):
            config = resolve_config(
                protocol_path=ROOT / "configs/protocol.yaml",
                budget_path=ROOT / "configs/budgets/256k.yaml",
                method_path=ROOT / f"configs/methods/{name}.yaml",
            )
            self.assertEqual(config.method.method, expected)

    def test_full_incremental_snapshot_round_trip_and_no_query_writer_api(self):
        self.assertNotIn("query", inspect.signature(SnAGAdaptWriter.ingest_step).parameters)
        writer = SnAGAdaptWriter(meta(), SnAGAdaptConfig(mode="full", storage_dtype="float32"))
        expected = []
        for index in range(5):
            feature = np.asarray([index + 1, index + 2], dtype=np.float32)
            expected.append(feature)
            writer.ingest_step(feature, index * 2, index * 2 + 1.5, f"clip-{index}")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshot"
            manifest = writer.freeze(10, root)
            snapshot = SnAGSnapshotReader(root)
            np.testing.assert_array_equal(snapshot.features(), np.stack(expected))
            self.assertEqual(manifest.state_bytes, sum(path.stat().st_size for path in root.iterdir()))
            self.assertEqual(snapshot.items[-1].t_end_s, 9.5)
            self.assertFalse(any("video_path" in path.name for path in root.iterdir()))

    def test_pooled_store_is_budgeted_and_preserves_absolute_provenance(self):
        writer = SnAGAdaptWriter(meta(), SnAGAdaptConfig(
            mode="pooled", storage_dtype="int8", budget_bytes=4096,
            near_capacity=2, far_capacity=2,
        ))
        for index in range(20):
            writer.ingest_step(np.full(8, index + 1), index * 2, index * 2 + 2, f"clip-{index}")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshot"
            writer.freeze(40, root)
            snapshot = SnAGSnapshotReader(root)
        self.assertLessEqual(snapshot.manifest.state_bytes, 4096)
        self.assertLessEqual(len(snapshot.items), 4)
        self.assertGreater(snapshot.items[0].aggregation_count, 1)
        self.assertEqual(snapshot.items[0].t_start_s, 0)
        self.assertEqual(snapshot.items[-1].t_end_s, 40)

    def test_pooled_store_refuses_to_merge_disjoint_support(self):
        writer = SnAGAdaptWriter(meta(), SnAGAdaptConfig(
            mode="pooled", budget_bytes=100_000, near_capacity=1, far_capacity=1,
        ))
        writer.ingest_step(np.ones(2), 0, 1, "a")
        writer.ingest_step(np.ones(2), 3, 4, "b")
        before = writer.items
        with self.assertRaisesRegex(MemoryError, "disjoint"):
            writer.ingest_step(np.ones(2), 6, 7, "c")
        self.assertEqual([item.source_ids for item in writer.items], [item.source_ids for item in before])

    def test_provenance_decode_uses_seconds_not_packed_indices(self):
        writer = SnAGAdaptWriter(meta(), SnAGAdaptConfig(mode="full"))
        writer.ingest_step(np.ones(2), 10, 14, "a")
        writer.ingest_step(np.ones(2), 12, 16, "b")
        points = ProvenancePointGenerator().level(writer.items, stride=1, level=0)
        spans = ProvenancePointGenerator.decode(points, np.asarray([[1, 1], [0.5, 0.5]]))
        np.testing.assert_allclose(spans, [[10, 14], [13, 15]])

    def test_multiple_queries_start_from_same_read_only_snapshot(self):
        writer = SnAGAdaptWriter(meta(), SnAGAdaptConfig(mode="full"))
        writer.ingest_step(np.asarray([0.25, 1]), 1, 2, "a")
        writer.ingest_step(np.asarray([0.75, 2]), 3, 4, "b")
        backend = Backend()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshot"
            writer.freeze(5, root)
            before = (root / "manifest.sha256").read_bytes()
            snapshot = SnAGSnapshotReader(root)
            self.assertFalse(snapshot.items[0].feature.flags.writeable)
            reader = SnAGAdaptReader(backend)
            first = reader.answer(snapshot, 0)
            second = reader.answer(snapshot, 1)
            after = (root / "manifest.sha256").read_bytes()
        self.assertEqual((first[0].start_s, second[0].start_s), (1, 3))
        self.assertEqual(backend.calls, 2)
        self.assertEqual(before, after)

    def test_rejects_retroactive_freeze_and_tampering(self):
        writer = SnAGAdaptWriter(meta(), SnAGAdaptConfig(mode="full"))
        writer.ingest_step(np.ones(2), 4, 5, "a")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshot"
            with self.assertRaisesRegex(ValueError, "behind"):
                writer.freeze(4.5, root)
            writer.freeze(6, root)
            (root / "features.npy").chmod(0o644)
            with (root / "features.npy").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "integrity"):
                SnAGSnapshotReader(root)

    def test_torch_bridge_calls_split_model_and_decodes_physical_time(self):
        writer = SnAGAdaptWriter(meta(), SnAGAdaptConfig(mode="full"))
        writer.ingest_step(np.ones(2), 10, 14, "a")
        writer.ingest_step(np.ones(2), 12, 16, "b")
        backend = SnAGTorchBackend(FakeSnAGModel(), score_threshold=0)
        spans = backend.predict(
            np.stack([item.feature for item in writer.items]), writer.items,
            (np.ones((2, 3), dtype=np.float32), np.ones((1, 3), dtype=bool)),
        )
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[0].start_s, 13)
        self.assertAlmostEqual(spans[0].end_s, 15)

    def test_torch_bridge_applies_official_padding_and_masks_it_out(self):
        writer = SnAGAdaptWriter(meta(), SnAGAdaptConfig(mode="full"))
        for index in range(3):
            writer.ingest_step(np.ones(2), index, index + 1, str(index))
        backend = SnAGTorchBackend(
            FakeSnAGModel(), score_threshold=0, input_video_length=8,
            minimum_chunk_size=4,
        )
        trace = backend.raw_predict(
            np.stack([item.feature for item in writer.items]),
            (np.ones((2, 3)), np.ones(3, dtype=bool)),
        )
        self.assertEqual((trace.observed_length, trace.input_length), (3, 8))
        self.assertEqual(trace.fpn_shapes[0][-1], 8)
        spans = backend.predict(
            np.stack([item.feature for item in writer.items]), writer.items,
            (np.ones((2, 3)), np.ones(3, dtype=bool)),
        )
        self.assertEqual(len(spans), 3)

    @unittest.skipUnless((UPSTREAM / ".git").exists(), "requires optional pinned SnAG checkout")
    def test_pinned_upstream_verification_and_text_feature_preprocessing(self):
        self.assertEqual(verify_upstream_checkout(UPSTREAM), UPSTREAM.resolve())
        self.assertEqual(len(UPSTREAM_REVISION), 40)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "text.npy"
            np.save(path, np.asarray([[3, 4], [0, 2]], dtype=np.float32))
            value = load_precomputed_text_feature(path, normalize=True, max_length=1)
        self.assertEqual(value.shape, (2, 1))
        np.testing.assert_allclose(value[:, 0], [0.6, 0.8])

    def test_g0_report_requires_raw_post_nms_and_official_metric_parity(self):
        backend = SnAGTorchBackend(FakeSnAGModel())
        trace = backend.raw_predict(
            np.ones((2, 2)), (np.ones((2, 3)), np.ones(3, dtype=bool)),
        )
        raw = compare_raw_traces(trace, trace)
        spans = (RankedSpan(1, 2, 0.5),)
        final = compare_ranked_spans(spans, spans)
        provenance = {
            "upstream_revision": UPSTREAM_REVISION,
            "checkpoint_sha256": "a" * 64,
            "option_sha256": "b" * 64,
        }
        incomplete = g0_gate_report(
            loader_provenance=provenance,
            raw_parity=raw,
            post_nms_parity=None,
            official_metric_parity=None,
        )
        complete = g0_gate_report(
            loader_provenance=provenance,
            raw_parity=raw,
            post_nms_parity=final,
            official_metric_parity={"passed": True},
        )
        self.assertFalse(incomplete["passed"])
        self.assertEqual(incomplete["classification"], "not-a-result")
        self.assertTrue(complete["passed"])


if __name__ == "__main__":
    unittest.main()

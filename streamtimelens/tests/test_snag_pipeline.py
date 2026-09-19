import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from streamtimelens.baselines.snag_adapt import SnAGAdaptConfig, SnAGSnapshotReader
from streamtimelens.baselines.snag_features import (
    FixtureFeatureExtractor,
    SnAGFeatureObservation,
    sampling_cell,
)
from streamtimelens.baselines.snag_config import load_snag_config
from streamtimelens.baselines.snag_physical import (
    SnAGPhysicalQuery,
    SnAGPhysicalReaderConfig,
    SnAGPhysicalTimeBackend,
    SnAGPhysicalTimeModel,
    item_metadata,
    physical_time_loss,
)
from streamtimelens.baselines.snag_stream import (
    SnAGSequentialIngestor,
    snag_runtime_protocol_audit,
)
from streamtimelens.baselines.snag_training import join_development_manifests
from streamtimelens.protocol.types import VideoMeta
from streamtimelens.evaluation.snag_benchmark import (
    SnAGFormalStartAssets,
    audit_snag_formal_start,
    sha256_file,
)
from streamtimelens.evaluation.snag_diagnostics import (
    REQUIRED_G2_ROWS,
    assemble_snag_g2_rows,
    audit_snag_g2_diagnostics,
)


class SnAGPipelineTest(unittest.TestCase):
    def test_sampling_cell_absorbs_frame_rate_jitter_but_preserves_large_gaps(self):
        self.assertEqual(sampling_cell(0.5005, 0.5, 4.0), (0.5, 1.0))
        self.assertEqual(sampling_cell(1.999, 0.5, 4.0), (1.5, 2.0))
        self.assertEqual(sampling_cell(2.1, 0.5, 4.0), (2.0, 2.5))

    def test_join_development_manifests_keeps_gt_out_of_video_rows(self):
        rows = join_development_manifests(
            [{"video_id": "v", "duration_s": 4.0, "path": "v.mp4"}],
            [{"video_id": "v", "query_id": "q", "query": "event"}],
            [{"video_id": "v", "query_id": "q", "gt_span": [1.0, 2.0]}],
        )
        self.assertEqual(rows, [{
            "video_id": "v", "query_id": "q", "query": "event",
            "gt_span": [1.0, 2.0], "duration": 4.0,
        }])
        with self.assertRaisesRegex(ValueError, "do not match"):
            join_development_manifests(
                [{"video_id": "v", "duration_s": 4.0}],
                [{"video_id": "v", "query_id": "q", "query": "event"}],
                [{"video_id": "v", "query_id": "other", "gt_span": [1.0, 2.0]}],
            )

    def _run(self, root: Path):
        observations = [
            SnAGFeatureObservation(np.full(4, index + 1), index, index + 1, str(index), index)
            for index in range(6)
        ]
        extractor = FixtureFeatureExtractor(observations)
        ingestor = SnAGSequentialIngestor(
            extractor,
            SnAGAdaptConfig(
                mode="pooled", storage_dtype="float32", budget_bytes=20_000,
                near_capacity=2, far_capacity=2,
            ),
        )
        return ingestor.run("unused.mp4", VideoMeta("v", 10, 1, 10), (2.5, 5.0), root)

    def test_sequential_ingest_freezes_before_future_observation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self._run(Path(temporary))
            snapshots = [SnAGSnapshotReader(row.path) for row in run.snapshots]
            audit = snag_runtime_protocol_audit(
                run, query_blind_signature=True,
                independent_query_checks=[{"passed": True}],
            )
        self.assertEqual(run.ingest.decode_passes, 1)
        self.assertLessEqual(max(item.t_end_s for item in snapshots[0].items), 2.5)
        self.assertLessEqual(max(item.t_end_s for item in snapshots[1].items), 5.0)
        self.assertTrue(audit["passed"])

    def test_physical_time_model_loss_and_snapshot_backend(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            run = self._run(Path(temporary))
            snapshot = SnAGSnapshotReader(run.snapshots[-1].path)
            config = SnAGPhysicalReaderConfig(
                feature_dim=4, query_dim=3, hidden_dim=16,
                num_heads=4, num_layers=1, fpn_levels=2,
                dropout=0.0, max_output_spans=3,
            )
            model = SnAGPhysicalTimeModel.build(config)
            features = torch.from_numpy(snapshot.features().copy())[None]
            metadata = torch.from_numpy(item_metadata(snapshot.items))[None]
            masks = torch.ones((1, len(snapshot.items)), dtype=torch.bool)
            query = torch.ones((1, 3))
            logits, offsets, levels = model(
                features, metadata, masks, query, torch.tensor([5.0]),
            )
            loss, report = physical_time_loss(
                logits, offsets, levels, masks, torch.tensor([[1.0, 3.0]]),
            )
            loss.backward()
            backend = SnAGPhysicalTimeBackend(model, config)
            spans = backend.predict(
                snapshot.features(), snapshot.items,
                SnAGPhysicalQuery(np.ones(3), snapshot.manifest.t_q),
            )
        self.assertGreater(report["positive_points"], 0)
        self.assertTrue(np.isfinite(float(loss.detach())))
        self.assertGreater(len(spans), 0)
        self.assertLessEqual(max(span.end_s for span in spans), 5.0)

    def test_physical_backend_clips_with_exact_long_query_time(self):
        import torch

        class BoundaryModel(torch.nn.Module):
            def forward(self, features, metadata, masks, query, t_q):
                del features, masks, query, t_q
                logits = (torch.full((1, 1), 10.0, device=metadata.device),)
                offsets = (torch.tensor([[[0.0, 10.0]]], device=metadata.device),)
                return logits, offsets, (metadata,)

        config = SnAGPhysicalReaderConfig(
            feature_dim=4, query_dim=3, hidden_dim=16,
            num_heads=4, num_layers=1, fpn_levels=1,
        )
        backend = SnAGPhysicalTimeBackend(BoundaryModel(), config)
        item = SnAGFeatureObservation(np.ones(4), 228.5, 229.0, "frame", 0)
        state = SnAGAdaptConfig(mode="full")
        with tempfile.TemporaryDirectory() as temporary:
            writer = SnAGSequentialIngestor(FixtureFeatureExtractor([item]), state)
            run = writer.run(
                "unused.mp4", VideoMeta("v", 229.1, 1, 230), (229.0288,), Path(temporary),
            )
            snapshot = SnAGSnapshotReader(run.snapshots[0].path)
            spans = backend.predict(
                snapshot.features(), snapshot.items,
                SnAGPhysicalQuery(np.ones(3), 229.0288),
            )
        self.assertEqual(spans[0].end_s, 229.0288)

    def test_frozen_pooled_config_is_typed_and_hash_stable(self):
        root = Path(__file__).resolve().parents[1]
        config = load_snag_config(root / "configs/snag/pooled_1m_v1.yaml")
        self.assertEqual(config.writer.budget_bytes, 1_048_576)
        self.assertEqual(config.reader.feature_dim, 512)
        self.assertIsInstance(config.training.arrival_ratios, tuple)
        self.assertEqual(len(config.sha256), 64)

    def test_g2_diagnostic_requires_all_three_named_rows(self):
        rows = [{
            "name": name,
            "completed": True,
            "protocol_passed": True,
            "metrics": {"miou": 1, "r1_iou_0.3": 1, "r5_iou_0.3": 1},
            "checkpoint_sha256": "a" * 64,
            "snapshot_manifest_sha256": "b" * 64,
        } for name in REQUIRED_G2_ROWS]
        self.assertTrue(audit_snag_g2_diagnostics(rows)["passed"])
        with self.assertRaisesRegex(ValueError, "exactly"):
            audit_snag_g2_diagnostics(rows[:-1])

    def test_g2_assembly_requires_checkpoint_and_cohort_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query_sha = "q" * 64
            pooled_snapshot_sha = "p" * 64

            def make_run(name, method, checkpoint, snapshot_sha, *, reused=False):
                run = root / name
                run.mkdir()
                (run / "snag_physical_reader.pth").write_bytes(checkpoint)
                (run / "metrics.json").write_text(json.dumps({"overall": {
                    "miou": 1, "r1_iou_0.3": 2, "r5_iou_0.3": 3,
                }}))
                (run / "protocol_audit.json").write_text(json.dumps({"passed": True}))
                provenance = {
                    "configuration": {"method": method},
                    "snapshot_index_sha256": snapshot_sha,
                    "query_manifest_sha256": query_sha,
                }
                if reused:
                    provenance["reused_checkpoint"] = str(root / "full" / "snag_physical_reader.pth")
                (run / "provenance.json").write_text(json.dumps(provenance))
                return run

            full = make_run("full", "snag-adapt-input-full", b"dense", "f" * 64)
            frozen = make_run(
                "frozen", "snag-adapt-pooled-B", b"dense", pooled_snapshot_sha,
                reused=True,
            )
            trained = make_run(
                "trained", "snag-adapt-pooled-B", b"pooled", pooled_snapshot_sha,
            )
            rows = assemble_snag_g2_rows(full, frozen, trained)
            self.assertTrue(audit_snag_g2_diagnostics(rows)["passed"])
            (frozen / "snag_physical_reader.pth").write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "reuse"):
                assemble_snag_g2_rows(full, frozen, trained)

    def test_formal_start_is_per_dataset_and_requires_clean_frozen_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "annotations.json"
            annotation.write_text(json.dumps({
                "v": {"duration": 4, "spans": [[0, 1]], "queries": ["q"]},
            }))
            videos = root / "videos"
            videos.mkdir()
            (videos / "v.mp4").write_bytes(b"v")
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint")
            gate = root / "gate.json"
            gate.write_text(json.dumps({"passed": True}))
            config = root / "frozen.yaml"
            config.write_text(
                "method: snag-adapt-pooled-B\nclassification: strict\nformal_frozen: true\n"
                f"reader:\n  checkpoint: {checkpoint}\n"
                f"  checkpoint_sha256: {sha256_file(checkpoint)}\n"
            )
            rows = [SnAGFormalStartAssets(
                name, annotation, videos, config, checkpoint, gate, gate, gate, gate, gate,
            ) for name in ("charades", "activitynet", "qvhighlights", "mad")]
            report = audit_snag_formal_start(rows, worktree_clean=True)
            dirty = audit_snag_formal_start(rows, worktree_clean=False)
        self.assertTrue(report["any_formal_test_can_start"])
        self.assertEqual(len(report["ready_datasets"]), 4)
        self.assertFalse(dirty["any_formal_test_can_start"])


if __name__ == "__main__":
    unittest.main()

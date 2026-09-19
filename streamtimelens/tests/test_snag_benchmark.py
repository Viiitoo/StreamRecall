import json
import tempfile
import unittest
from pathlib import Path

from streamtimelens.evaluation.snag_benchmark import (
    SnAGDatasetAssets,
    audit_snag_assets,
    snag_protocol_audit,
)


class SnAGBenchmarkAuditTest(unittest.TestCase):
    def test_counts_past_only_cohort_and_blocks_missing_runtime_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = {
                "v": {"duration": 8, "spans": [[1, 2], [5, 7]], "queries": ["a", "b"]},
            }
            annotation = root / "annotations.json"
            annotation.write_text(json.dumps(rows), encoding="utf-8")
            videos = root / "videos"
            videos.mkdir()
            (videos / "v.mp4").write_bytes(b"fixture")
            assets = []
            for name in ("charades", "activitynet", "qvhighlights", "mad"):
                assets.append(SnAGDatasetAssets(
                    name, annotation, videos, None, None, None,
                    upstream_recipe=name != "qvhighlights",
                ))
            report = audit_snag_assets(assets)
        self.assertEqual(report["datasets"][0]["counts"], {
            "videos": 1, "queries": 2, "observations": 8, "eligible": 5,
        })
        self.assertEqual(report["ready_dataset_count"], 0)
        self.assertFalse(report["benchmark_table_update_allowed"])
        self.assertFalse(snag_protocol_audit(report)["passed"])

    def test_ready_requires_every_dataset_and_all_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "annotations.json"
            annotation.write_text(json.dumps({
                "v": {"duration": 4, "spans": [[0, 1]], "queries": ["q"]},
            }), encoding="utf-8")
            directory = root / "assets"
            directory.mkdir()
            (directory / "v.mp4").write_bytes(b"v")
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"model")
            assets = [SnAGDatasetAssets(
                name, annotation, directory, directory, checkpoint, directory, True,
            ) for name in ("charades", "activitynet", "qvhighlights", "mad")]
            report = audit_snag_assets(assets)
        self.assertEqual(report["ready_dataset_count"], 4)
        self.assertTrue(report["benchmark_table_update_allowed"])
        # Asset readiness alone is not a runtime protocol pass.
        self.assertFalse(snag_protocol_audit(report)["passed"])


if __name__ == "__main__":
    unittest.main()

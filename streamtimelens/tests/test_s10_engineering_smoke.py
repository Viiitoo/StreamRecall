import tempfile
import unittest
from pathlib import Path

from streamtimelens.evaluation.smoke import run_engineering_smoke
from streamtimelens.protocol.snapshot import SnapshotReader


class EngineeringSmokeTest(unittest.TestCase):
    def test_short_medium_long_all_internal_paths_have_full_artifacts(self):
        methods = (
            "uniform_raw", "semantic_reservoir", "vst_summary", "oasis_adapt",
            "evidence_fixed", "full",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = run_engineering_smoke(root, video_count=3, methods=methods)
            for method in methods:
                self.assertTrue((root / method / "predictions.jsonl").is_file())
                self.assertTrue((root / method / "metrics.json").is_file())
                self.assertTrue((root / method / "sample_resources.jsonl").is_file())
                reader = SnapshotReader(
                    root / method / "smoke-002" / "snapshots" / "rho_1.00",
                )
                self.assertLessEqual(reader.manifest.state_bytes, reader.manifest.budget_bytes)
        self.assertEqual(summary["runs"], 18)
        self.assertEqual(summary["snapshots"], 72)

    def test_invalid_method_fails(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "matrix"):
            run_engineering_smoke(Path(temporary), video_count=1, methods=("bad",))


if __name__ == "__main__":
    unittest.main()

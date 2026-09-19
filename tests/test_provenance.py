import json
import tempfile
import unittest
from pathlib import Path

from baas.provenance import (
    RESULTS_ROOT,
    code_version,
    versioned_result_path,
    write_artifact_manifest,
    write_provenance,
)


class ProvenanceTest(unittest.TestCase):
    def test_result_path_is_grouped_by_commit_and_is_idempotent(self):
        original = RESULTS_ROOT / "baas" / "smoke" / "sample.jsonl"
        expected = RESULTS_ROOT / code_version() / "baas" / "smoke" / "sample.jsonl"
        self.assertEqual(versioned_result_path(original), expected)
        self.assertEqual(versioned_result_path(expected), expected)

    def test_result_path_outside_results_is_rejected(self):
        with self.assertRaises(ValueError):
            versioned_result_path(Path("/tmp/not-a-versioned-result.jsonl"))

    def test_write_provenance_freezes_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "run"
            provenance_path = write_provenance(
                output_dir,
                configuration={"budget": 2048, "seed": 42},
                config_filename="config.resolved.json",
                command=["test-command"],
            )
            metadata = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["code"]["commit"], code_version())
            self.assertEqual(metadata["configuration"], {"budget": 2048, "seed": 42})
            self.assertEqual(metadata["configuration_file"], "config.resolved.json")
            self.assertTrue((output_dir / "config.resolved.json").is_file())
            self.assertEqual((output_dir / "git_revision.txt").read_text().strip(), code_version())

    def test_resolved_configuration_replaces_source_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.yaml"
            source.write_text("budget: 2048\n", encoding="utf-8")
            output_dir = root / "run"
            write_provenance(
                output_dir,
                config_path=source,
                configuration={"budget": 2048, "execution": {"dataset": "charades-timelens"}},
            )
            snapshot = (output_dir / "config.resolved.yaml").read_text(encoding="utf-8")
            metadata = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
            self.assertIn("dataset: charades-timelens", snapshot)
            self.assertIn("configuration_source_sha256", metadata)
            self.assertEqual(metadata["configuration"]["execution"]["dataset"], "charades-timelens")

    def test_artifact_manifest_hashes_all_completed_files_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "nested" / "b.txt").write_text("bb", encoding="utf-8")
            manifest_path = write_artifact_manifest(root)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["self_excluded"], "artifact_manifest.json")
            self.assertEqual(payload["file_count"], 2)
            self.assertEqual(
                [row["path"] for row in payload["files"]], ["a.txt", "nested/b.txt"],
            )
            (root / "link").symlink_to(root / "a.txt")
            with self.assertRaises(ValueError):
                write_artifact_manifest(root)


if __name__ == "__main__":
    unittest.main()

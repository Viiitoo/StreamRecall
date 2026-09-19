import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from streamtimelens.config import read_resolved_config, resolve_config, write_resolved_config
from streamtimelens.provenance import collect_provenance, git_metadata, write_run_provenance


ROOT = Path(__file__).resolve().parents[1]


class ProvenanceTest(unittest.TestCase):
    def _config(self):
        return resolve_config(protocol_path=ROOT / "configs/protocol.yaml", budget_path=ROOT / "configs/budgets/1m.yaml",
                              method_path=ROOT / "configs/methods/full.yaml")

    def test_resolved_config_round_trip_and_provenance_output(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_resolved_config(config, root / "config.yaml")
            self.assertEqual(read_resolved_config(root / "config.yaml").sha256, config.sha256)
            provenance = write_run_provenance(root / "run", config, model_revisions={"writer": "r1"}, data_revisions={"dev": "d1"})
            data = json.loads(provenance.read_text())
            self.assertEqual(data["resolved_config_sha256"], config.sha256)
            self.assertTrue(data["started_at_utc"] and data["ended_at_utc"])
            self.assertEqual(data["models"]["writer"], "r1")
            self.assertTrue((root / "run" / "git_revision.txt").read_text().strip())

    def test_git_metadata_reports_clean_dirty_and_unavailable_honestly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(git_metadata(root)["status"], "unavailable")
            subprocess.run(["git", "init", str(root)], check=True, stdout=subprocess.DEVNULL)
            (root / "a.txt").write_text("a")
            subprocess.run(["git", "-C", str(root), "add", "a.txt"], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "init"], check=True, stdout=subprocess.DEVNULL)
            self.assertFalse(git_metadata(root)["dirty"])
            (root / "a.txt").write_text("changed")
            self.assertTrue(git_metadata(root)["dirty"])
        self.assertIn("code", collect_provenance(self._config()))


if __name__ == "__main__":
    unittest.main()

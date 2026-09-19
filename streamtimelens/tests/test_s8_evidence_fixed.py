import unittest
from pathlib import Path

import yaml

from streamtimelens.baselines.evidence_fixed import validate_evidence_fixed_pair
from streamtimelens.stream.runner import IngestConfig, StreamingIngestor


class EvidenceFixedTest(unittest.TestCase):
    def test_checked_in_configs_only_change_method_and_trigger(self):
        root = Path(__file__).resolve().parents[1] / "configs" / "methods"
        full = yaml.safe_load((root / "streamtimelens.yaml").read_text(encoding="utf-8"))
        fixed = yaml.safe_load((root / "evidence_fixed.yaml").read_text(encoding="utf-8"))
        validate_evidence_fixed_pair(full, fixed)
        self.assertIs(StreamingIngestor, StreamingIngestor)
        full_config = IngestConfig(method_name="streamtimelens", trigger_mode="joint")
        fixed_config = IngestConfig(method_name="evidence_fixed", trigger_mode="periodic")
        differences = {
            key for key in vars(full_config)
            if getattr(full_config, key) != getattr(fixed_config, key)
        }
        self.assertEqual(differences, {"method_name", "trigger_mode"})

    def test_non_trigger_drift_is_rejected(self):
        full = {"method": "streamtimelens", "trigger": "joint", "writer": "same"}
        fixed = {"method": "evidence-fixed", "trigger": "periodic", "writer": "different"}
        with self.assertRaisesRegex(ValueError, "non-trigger"):
            validate_evidence_fixed_pair(full, fixed)


if __name__ == "__main__":
    unittest.main()

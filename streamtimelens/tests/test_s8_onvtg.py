import tempfile
import unittest
from pathlib import Path

from streamtimelens.baselines.onvtg_query_known import (
    QUERY_KNOWN_PROTOCOL, OnVTGReference, adapt_official_onvtg_rows,
    assert_protocol_table,
)


class OnVTGAdapterTest(unittest.TestCase):
    def test_official_rows_are_query_known_and_cannot_mix_with_main_table(self):
        predictions = adapt_official_onvtg_rows([
            {"query_id": "q", "video_id": "v", "pred_span": [1, 2], "score": .8},
        ])
        row = predictions[0].to_dict()
        self.assertEqual(row["protocol"], QUERY_KNOWN_PROTOCOL)
        self.assertEqual(row["method"], "onvtg-official")
        assert_protocol_table([row], QUERY_KNOWN_PROTOCOL)
        with self.assertRaisesRegex(ValueError, "mixes"):
            assert_protocol_table([row, {"protocol": "delayed_query_v1"}], QUERY_KNOWN_PROTOCOL)

    def test_reference_command_uses_frozen_official_entrypoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            (root / "scripts" / "eval.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text("x: 1\n", encoding="utf-8")
            reference = OnVTGReference(root, config, root / "best.pth", "sha", "hash", "features-v1")
            command = reference.evaluation_command("0,1")
        self.assertIn("scripts/eval.sh", command[3])
        self.assertEqual(command[:3], ("env", "CUDA_VISIBLE_DEVICES=0,1", "bash"))

    def test_duplicate_official_query_fails(self):
        row = {"query_id": "q", "video_id": "v", "span": [1, 2], "confidence": .5}
        with self.assertRaisesRegex(ValueError, "unique"):
            adapt_official_onvtg_rows([row, row])


if __name__ == "__main__":
    unittest.main()

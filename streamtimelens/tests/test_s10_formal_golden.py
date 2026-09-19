import json
import tempfile
import unittest
from pathlib import Path

from streamtimelens.evaluation.golden import assert_golden_matches, run_formal_golden


FIXTURE = Path(__file__).parent / "fixtures" / "formal_golden_v1.json"


class FormalGoldenRegressionTest(unittest.TestCase):
    def test_one_video_four_rho_three_queries_matches_frozen_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            actual = run_formal_golden(Path(temporary))
        assert_golden_matches(actual, FIXTURE)
        self.assertEqual(len(actual["snapshot_hashes"]), 4)
        self.assertEqual(len(actual["predictions"]), 12)

    def test_changed_fixture_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = run_formal_golden(root / "run")
            changed = dict(actual)
            changed["case"] = "silent-change"
            path = root / "changed.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "intentional"):
                assert_golden_matches(actual, path)


if __name__ == "__main__":
    unittest.main()

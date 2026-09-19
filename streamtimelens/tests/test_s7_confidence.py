import unittest

from streamtimelens.config import ProtocolConfig
from streamtimelens.retrieval.confidence import (
    ConfidenceConfig, ConfidenceFeatures, calibrate_confidence,
)


class ConfidenceTest(unittest.TestCase):
    def test_only_frozen_features_drive_continuous_confidence(self):
        config = ConfidenceConfig(not_found_threshold=0.4)
        good = calibrate_confidence(ConfidenceFeatures(0.8, 0.4, 1.0, "ok"), config)
        poor = calibrate_confidence(ConfidenceFeatures(-1.0, 0.0, 0.0, "failure"), config)
        self.assertGreater(good.confidence, poor.confidence)
        self.assertFalse(good.not_found)
        self.assertTrue(poor.not_found)
        self.assertGreaterEqual(good.confidence, 0)
        self.assertLessEqual(good.confidence, 1)

    def test_invalid_features_and_threshold_fail(self):
        with self.assertRaisesRegex(ValueError, "completeness"):
            ConfidenceFeatures(1, 0, 2, "ok")
        with self.assertRaisesRegex(ValueError, "temperature"):
            ConfidenceConfig(temperature=0)
        with self.assertRaisesRegex(ValueError, "confidence calibration"):
            ProtocolConfig(not_found_threshold=1.1)


if __name__ == "__main__":
    unittest.main()

import unittest

from streamtimelens.observer.quota import WriterQuota


class WriterQuotaTest(unittest.TestCase):
    def test_one_two_and_four_calls_per_minute(self):
        for rate, interval in ((1, 60), (2, 30), (4, 15)):
            quota = WriterQuota(rate)
            successes = [quota.try_consume(timestamp) for timestamp in (0, interval, interval * 2)]
            self.assertEqual(successes, [True, True, True], rate)

    def test_repeated_timestamp_and_long_pause_never_exceed_capacity(self):
        quota = WriterQuota(4)
        self.assertTrue(quota.try_consume(0))
        self.assertFalse(quota.try_consume(0))
        self.assertEqual(quota.last_reason, "call_ceiling")
        self.assertTrue(quota.try_consume(600))
        self.assertFalse(quota.try_consume(600))
        self.assertEqual(quota.last_reason, "token_bucket")

    def test_short_video_rounding_rule_and_initial_token_are_explicit(self):
        quota = WriterQuota(4)
        self.assertEqual([quota.max_calls(t) for t in (0, 14.9, 15, 59.9, 60)], [1, 1, 2, 4, 5])
        cold = WriterQuota(2, initial_tokens=0)
        self.assertFalse(cold.try_consume(0))
        self.assertTrue(cold.try_consume(30))
        with self.assertRaises(ValueError):
            WriterQuota(1, initial_tokens=2)

    def test_non_monotonic_time_is_rejected(self):
        quota = WriterQuota(1)
        quota.try_consume(10)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            quota.try_consume(9)


if __name__ == "__main__":
    unittest.main()

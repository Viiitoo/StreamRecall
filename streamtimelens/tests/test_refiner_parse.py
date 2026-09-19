import unittest

from streamtimelens.refiner.parse import parse_refiner_output


class RefinerParseTest(unittest.TestCase):
    def test_valid_decimal_and_multiple_span_uses_first(self):
        result = parse_refiner_output("The event happens in 1.5 - 2.5 seconds; maybe 3 - 4.", t_q=5, candidate=(1, 3))
        self.assertTrue(result.valid)
        self.assertEqual(result.span, (1.5, 2.5))
        self.assertTrue(result.multiple_span)

    def test_typed_failures_do_not_create_fallback_predictions(self):
        cases = [
            ("no time here", "no_timestamp"),
            ("3 - 2 seconds", "invalid_order"),
            ("3 - 6 seconds", "out_of_bounds"),
            ("3 - 4 seconds", "no_candidate_overlap"),
            ("1e1 - 2e1 seconds", "out_of_bounds"),
        ]
        for answer, status in cases:
            with self.subTest(answer=answer):
                result = parse_refiner_output(answer, t_q=5, candidate=(1, 2))
                self.assertEqual(result.status, status)
                self.assertIsNone(result.span)


if __name__ == "__main__":
    unittest.main()

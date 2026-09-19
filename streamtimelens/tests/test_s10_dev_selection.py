import unittest

from streamtimelens.evaluation.dev_selection import DevRun, select_dev_configs


class DevSelectionTest(unittest.TestCase):
    def test_diagnostic_order_pareto_and_explicit_keep(self):
        strong = DevRun("strong", .8, .8, .7, .1, 10, 2)
        dominated = DevRun("dominated", .7, .7, .6, .0, 12, 1)
        diagnostic = DevRun("diagnostic", .1, .1, .1, -.2, 100, .1, diagnostic_keep=True)
        result = select_dev_configs([dominated, diagnostic, strong])
        self.assertEqual(result["selection_order"][0], "writer_coverage_and_candidate_recall")
        self.assertIn("strong", result["pareto_config_ids"])
        self.assertNotIn("dominated", result["pareto_config_ids"])
        self.assertIn("diagnostic", result["selected_config_ids"])

    def test_duplicate_and_invalid_runs_fail(self):
        row = DevRun("x", 0, 0, 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "unique"):
            select_dev_configs([row, row])
        with self.assertRaisesRegex(ValueError, "rates"):
            DevRun("x", 2, 0, 0, 0, 0, 0)


if __name__ == "__main__":
    unittest.main()

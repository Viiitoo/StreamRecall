import unittest

from streamtimelens.evaluation.formal_report import FormalRunSummary, build_formal_report


def row(method, rho, cohort, *, miou=.5, gpu=1, snapshot=100, audited=True):
    return FormalRunSummary(
        method, 262144, rho, cohort, 10, miou, .8, .7, .6,
        .7, .8, .5, .1, gpu, 2, snapshot, audited, audited,
    )


class FormalReportTest(unittest.TestCase):
    def test_four_tables_require_all_rhos_and_fixed_cohort(self):
        rows = [
            row(method, rho, cohort, miou=.6 if method == "a" else .5,
                gpu=1 if method == "a" else 2)
            for method in ("a", "b") for rho in (.25, .5, .75, 1.0)
            for cohort in ("natural", "fixed_0.25")
        ]
        report = build_formal_report(rows, expected_methods=("a", "b"))
        self.assertEqual(set(report), {
            "accuracy_table", "diagnostic_table", "resource_table", "pareto_table",
        })
        self.assertEqual(len(report["accuracy_table"]), 16)
        self.assertEqual([item["method"] for item in report["pareto_table"]], ["a"])

    def test_audit_and_incomplete_matrix_fail_before_report(self):
        with self.assertRaisesRegex(ValueError, "integrity"):
            build_formal_report([row("a", .25, "natural", audited=False)], expected_methods=("a",))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_formal_report([row("a", .25, "natural")], expected_methods=("a",))


if __name__ == "__main__":
    unittest.main()

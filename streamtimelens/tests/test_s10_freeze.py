import unittest

from streamtimelens.config import (
    BudgetConfig, MethodConfig, ModelConfig, ProtocolConfig, ResolvedConfig,
)
from streamtimelens.evaluation.freeze import freeze_dev_configuration


def config():
    return ResolvedConfig(
        ProtocolConfig(), BudgetConfig(262144, 1),
        MethodConfig("streamtimelens", "joint", "writer", "hybrid", "boundary"),
        ModelConfig(),
    )


class FreezeTest(unittest.TestCase):
    def test_all_gate_hashes_prompts_budgets_and_policy_are_frozen(self):
        frozen = freeze_dev_configuration(
            {"winner": config()}, {"selected_config_ids": ["winner"]},
            {"p0_gate": {"passed": True}},
            {"decision": "go", "selected_writer": "timelens-7b"},
        )
        self.assertEqual(frozen["status"], "frozen")
        self.assertEqual(frozen["selected_writer"], "timelens-7b")
        self.assertEqual(frozen["configs"]["winner"]["resolved"]["budget"]["memory_bytes"], 262144)
        self.assertEqual(len(frozen["prompt_hashes"]["writer_template_sha256"]), 64)
        self.assertIn("rerun", frozen["post_freeze_policy"].lower())

    def test_missing_gate_writer_or_config_refuses_freeze(self):
        with self.assertRaisesRegex(ValueError, "Oracle"):
            freeze_dev_configuration(
                {"x": config()}, {"selected_config_ids": ["x"]},
                {"p0_gate": {"passed": False}}, {"decision": "go", "selected_writer": "x"},
            )
        with self.assertRaisesRegex(ValueError, "writer"):
            freeze_dev_configuration(
                {"x": config()}, {"selected_config_ids": ["x"]},
                {"p0_gate": {"passed": True}}, {"decision": "blocked"},
            )


if __name__ == "__main__":
    unittest.main()

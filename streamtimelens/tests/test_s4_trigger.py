import unittest

from streamtimelens.observer.trigger import JointTrigger


class ForbiddenSignal:
    def __float__(self):
        raise AssertionError("disabled ablation read a forbidden signal")


class JointTriggerTest(unittest.TestCase):
    def test_each_ablation_only_reads_allowed_signals(self):
        periodic = JointTrigger("periodic", periodic_interval_s=1, minimum_gap_s=0)
        self.assertIsNotNone(periodic.propose(0, lite_score=ForbiddenSignal(), semantic_score=ForbiddenSignal()))

        cut_motion = JointTrigger("cut-motion", threshold=0.5, minimum_gap_s=0)
        proposal = cut_motion.propose(0, lite_score=0.8, semantic_score=ForbiddenSignal())
        self.assertEqual(proposal.reasons, ("signal:lite",))

        semantic = JointTrigger("semantic", threshold=0.5, minimum_gap_s=0)
        proposal = semantic.propose(0, lite_score=ForbiddenSignal(), semantic_score=0.8, hard_cut=True)
        self.assertEqual(proposal.reasons, ("signal:semantic",))

    def test_hard_cut_minimum_gap_and_max_gap_are_separate_rules(self):
        trigger = JointTrigger("joint", threshold=10, minimum_gap_s=5, max_gap_s=10)
        hard_cut = trigger.propose(0, lite_score=0, semantic_score=0, hard_cut=True)
        self.assertIn("rule:hard_cut", hard_cut.reasons)
        self.assertIsNone(trigger.propose(1, lite_score=100, semantic_score=100))
        self.assertEqual(trigger.last_decision_reasons, ("rule:minimum_gap",))
        # The safety max-gap rule overrides both the high threshold and min-gap.
        max_gap = trigger.propose(10, lite_score=0, semantic_score=0)
        self.assertEqual(max_gap.reasons, ("rule:max_gap", "signal:age"))

    def test_writer_mark_resets_age_rule(self):
        trigger = JointTrigger("semantic", threshold=2, minimum_gap_s=0, max_gap_s=10)
        self.assertIsNone(trigger.propose(0, semantic_score=0))
        self.assertIsNotNone(trigger.propose(10, semantic_score=0))
        trigger.mark_written(10)
        self.assertIsNone(trigger.propose(19, semantic_score=0))
        self.assertIsNotNone(trigger.propose(20, semantic_score=0))


if __name__ == "__main__":
    unittest.main()

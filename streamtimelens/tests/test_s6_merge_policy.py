import unittest

import numpy as np

from streamtimelens.memory.card_store import CardStore
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.merge_policy import MergeCandidateQueue
from streamtimelens.observer.clip_encoder import serialize_embedding


def card(card_id, start, vector=(1.0, 0.0)):
    return EvidenceCard(
        card_id, 0, start, start + 1, card_id,
        normalized_text=card_id,
        text_embedding=serialize_embedding(np.asarray(vector), "fp16"),
    )


class MergeCandidateQueueTest(unittest.TestCase):
    def test_only_adjacent_candidates_and_complete_reason_are_returned(self):
        store = CardStore()
        for item in (card("a", 0), card("b", 2), card("c", 4, (0.0, 1.0))):
            store.insert(item)
        queue = MergeCandidateQueue()
        queue.rebuild(store)
        result = queue.pop_best(store)
        self.assertEqual((result.left_id, result.right_id), ("a", "b"))
        self.assertAlmostEqual(result.reason.semantic_similarity, 1.0, places=4)
        self.assertEqual(result.reason.temporal_gap_s, 1.0)
        self.assertAlmostEqual(
            result.reason.score,
            result.reason.semantic_term - result.reason.gap_penalty - result.reason.boundary_penalty,
        )

    def test_node_update_and_changed_adjacency_invalidate_stale_entries(self):
        store = CardStore()
        for item in (card("a", 0), card("c", 4)):
            store.insert(item)
        queue = MergeCandidateQueue()
        queue.rebuild(store)
        queue.invalidate("a")
        self.assertIsNone(queue.pop_best(store))
        store.insert(card("b", 2))
        queue.rebuild(store)
        candidates = [queue.pop_best(store), queue.pop_best(store)]
        self.assertEqual(
            {(item.left_id, item.right_id) for item in candidates if item is not None},
            {("a", "b"), ("b", "c")},
        )

    def test_card_ids_are_the_deterministic_tie_break(self):
        store = CardStore()
        for item in (card("a", 0), card("b", 2), card("c", 4)):
            store.insert(item)
        queue = MergeCandidateQueue(gap_weight=0, boundary_weight=0)
        queue.rebuild(store)
        self.assertEqual(queue.pop_best(store).left_id, "a")


if __name__ == "__main__":
    unittest.main()

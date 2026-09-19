import unittest

from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.forest import EventForest


def card(card_id, start, summary_size=8):
    return EvidenceCard(card_id, 0, start, start + 1, card_id * summary_size)


class EventForestMergeTest(unittest.TestCase):
    def test_soft_merge_retains_children_and_accounts_every_node(self):
        forest = EventForest(64 * 1024, debug=True)
        forest.insert(card("a", 0))
        forest.insert(card("b", 2))
        before_count = len(forest.store)
        record = forest.merge_best()
        self.assertEqual(record.mode, "soft")
        self.assertEqual(len(forest.store), before_count + 1)
        parent = forest.store.get(record.parent_id)
        self.assertEqual(parent.child_ids, ["a", "b"])
        self.assertEqual(len(forest.roots), 1)
        self.assertGreater(forest.accounting_components()["forest_topology"], 0)

    def test_hard_merge_recursively_removes_query_payload_without_dangling_ids(self):
        forest = EventForest(64 * 1024)
        for item in (card("a", 0), card("b", 2)):
            forest.insert(item)
        soft = forest.merge_best()
        forest.insert(card("c", 4))
        hard = forest.merge_best(force_hard=True)
        self.assertEqual(hard.mode, "hard")
        self.assertEqual(hard.nodes_removed, 4)
        self.assertNotIn(soft.parent_id, forest.store)
        self.assertEqual(len(forest.store), 1)
        root = forest.roots[0]
        self.assertEqual(root.child_ids, [])
        self.assertEqual(root.compacted_child_count, 4)
        forest.assert_invariants(current_stream_time=5)

    def test_restore_reconstructs_roots_and_order(self):
        forest = EventForest(64 * 1024)
        forest.insert(card("a", 0))
        forest.insert(card("b", 2))
        forest.merge_best()
        rows = forest.store.rows()
        restored = EventForest(64 * 1024)
        restored.restore(reversed(rows))
        self.assertEqual(tuple(card.id for card in restored.cards), forest.store.ids)
        self.assertEqual(tuple(card.id for card in restored.roots), tuple(card.id for card in forest.roots))
        restored.assert_invariants(current_stream_time=3)

    def test_invariant_suite_rejects_budget_raw_child_embedding_time_and_cycle(self):
        too_large = EventForest(256)
        too_large.insert(card("large", 0, summary_size=100))
        with self.assertRaisesRegex(AssertionError, "limit"):
            too_large.assert_invariants(current_stream_time=1)

        future = EventForest(64 * 1024)
        future.insert(card("future", 10))
        with self.assertRaisesRegex(AssertionError, "history"):
            future.assert_invariants(current_stream_time=1)

        missing_raw = EventForest(64 * 1024)
        raw_card = card("raw", 0)
        raw_card.raw_ref_ids = ["missing.jpg"]
        raw_card.raw_ref_status = {"missing.jpg": "available"}
        missing_raw.insert(raw_card)
        with self.assertRaisesRegex(AssertionError, "raw refs"):
            missing_raw.assert_invariants(
                current_stream_time=1, allowed_raw_refs={"different.jpg"},
            )

        missing_embedding = EventForest(64 * 1024, require_embeddings=True)
        missing_embedding.insert(card("plain", 0))
        with self.assertRaisesRegex(AssertionError, "embedding"):
            missing_embedding.assert_invariants(current_stream_time=1)

        dangling = EventForest(64 * 1024)
        parent = card("parent", 0)
        parent.child_ids = ["missing"]
        dangling.insert(parent)
        with self.assertRaisesRegex(AssertionError, "dangling"):
            dangling.assert_invariants(current_stream_time=1)

        cyclic = EventForest(64 * 1024)
        first, second = card("a", 0), card("b", 2)
        first.child_ids, second.child_ids = ["b"], ["a"]
        cyclic.insert(first)
        cyclic.insert(second)
        with self.assertRaisesRegex(AssertionError, "cycle"):
            cyclic.assert_invariants(current_stream_time=3)


if __name__ == "__main__":
    unittest.main()

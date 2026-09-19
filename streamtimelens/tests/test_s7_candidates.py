import unittest

from streamtimelens.retrieval.candidates import expand_temporal_candidates
from streamtimelens.retrieval.mmr import MMRChoice


def row(card_id, start, end, *, children=(), raw=True):
    return {
        "id": card_id, "t_start": start, "t_end": end, "child_ids": list(children),
        "raw_ref_ids": [card_id + ".jpg"] if raw else [],
        "raw_ref_status": {card_id + ".jpg": "available"} if raw else {},
    }


def choice(card_id, score=1.0):
    return MMRChoice(card_id, score, score, 0.0, 0.0)


class CandidateExpansionTest(unittest.TestCase):
    def test_leaf_neighbors_parent_children_and_future_guard(self):
        rows = [
            row("a", 0, 2), row("b", 2.5, 4), row("far", 8, 9),
            row("c1", 10, 12), row("c2", 12.5, 15, raw=False),
            row("parent", 10, 15, children=("c1", "c2"), raw=False),
            row("future", 19, 21),
        ]
        candidates = expand_temporal_candidates(
            [choice("b"), choice("parent"), choice("future")], rows, t_q=20,
        )
        leaf = next(item for item in candidates if item.diagnostics["source_card_id"] == "b")
        parent = next(item for item in candidates if item.diagnostics["source_card_id"] == "parent")
        self.assertEqual(leaf.contributing_card_ids, ("a", "b"))
        self.assertEqual(parent.contributing_card_ids, ("c1", "c2"))
        self.assertEqual(parent.raw_completeness, 1.0)
        self.assertTrue(all(candidate.end_s <= 20 for candidate in candidates))
        self.assertNotIn("future", {value for item in candidates for value in item.contributing_card_ids})

    def test_missing_selected_card_and_duplicate_rows_fail(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            expand_temporal_candidates([choice("x")], [row("a", 0, 1)], t_q=2)
        with self.assertRaisesRegex(ValueError, "unique"):
            expand_temporal_candidates([choice("a")], [row("a", 0, 1), row("a", 1, 2)], t_q=2)


if __name__ == "__main__":
    unittest.main()

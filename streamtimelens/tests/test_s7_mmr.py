import unittest

from streamtimelens.observer.clip_encoder import serialize_embedding
from streamtimelens.retrieval.mmr import hierarchy_mmr
from streamtimelens.retrieval.ranker import RankedCard


def row(card_id, vector, *, children=(), mode="leaf"):
    return {
        "id": card_id, "text_embedding": serialize_embedding(vector),
        "child_ids": list(children), "merge_mode": mode,
    }


class HierarchyMMRTest(unittest.TestCase):
    def test_masks_relatives_penalizes_overlap_and_builds_prefixes_once(self):
        rows = [
            row("parent", [1, 0], children=("child",), mode="soft"),
            row("child", [1, 0]), row("overlap", [0.99, 0.01]),
            row("diverse", [0, 1]), row("late", [0.5, 0.5]),
        ]
        ranked = [
            RankedCard("parent", 1.0, (0, 10), 1, {}),
            RankedCard("child", 0.99, (0, 5), 0, {}),
            RankedCard("overlap", 0.98, (0, 5), 0, {}),
            RankedCard("diverse", 0.75, (20, 25), 0, {}),
            RankedCard("late", 0.70, (30, 35), 0, {}),
        ]
        result = hierarchy_mmr(
            ranked, rows, semantic_penalty=0.3, temporal_penalty=0.3,
        )
        self.assertEqual(result.choices[0].card_id, "parent")
        self.assertNotIn("child", [choice.card_id for choice in result.choices])
        self.assertEqual(result.choices[1].card_id, "diverse")
        self.assertEqual(result.prefixes[1], ("parent",))
        self.assertEqual(result.prefixes[5], tuple(choice.card_id for choice in result.choices))
        self.assertEqual(result.prefixes[8], result.prefixes[5])

    def test_dangling_hierarchy_and_duplicate_ids_fail(self):
        ranked = [RankedCard("a", 1, (0, 1), 0, {})]
        with self.assertRaisesRegex(ValueError, "dangling"):
            hierarchy_mmr(
                ranked, [row("a", [1, 0], children=("missing",))],
                semantic_penalty=0, temporal_penalty=0,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            hierarchy_mmr(
                ranked, [row("a", [1, 0]), row("a", [1, 0])],
                semantic_penalty=0, temporal_penalty=0,
            )


if __name__ == "__main__":
    unittest.main()

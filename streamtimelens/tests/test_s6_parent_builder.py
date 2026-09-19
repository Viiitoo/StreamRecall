import unittest

import numpy as np

from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.parent_builder import build_parent_card
from streamtimelens.observer.clip_encoder import deserialize_embedding, serialize_embedding


def child(card_id, start, end, vector, **kwargs):
    return EvidenceCard(
        card_id, 0, start, end, f"summary {card_id}",
        support_timestamps=kwargs.pop("support_timestamps", [start, end]),
        text_embedding=serialize_embedding(np.asarray(vector), "fp16"),
        source_chunk_ids=[card_id], **kwargs,
    )


class ParentBuilderTest(unittest.TestCase):
    def test_union_limits_weighted_pooling_and_soft_children(self):
        first = child(
            "a", 0, 1, (1.0, 0.0), actors=["Alice", "BOB"],
            raw_ref_ids=["a.jpg"], raw_ref_status={"a.jpg": "available"},
            boundary_cache={"left_frame_ids": ["a.jpg"], "left_hit": True},
        )
        second = child(
            "b", 2, 5, (0.0, 1.0), actors=["alice", "Carol"],
            raw_ref_ids=["b.jpg"], raw_ref_status={"b.jpg": "available"},
            boundary_cache={"right_frame_ids": ["b.jpg"], "right_hit": True},
        )
        parent = build_parent_card([second, first], hard=False, max_actors=2)
        self.assertEqual((parent.t_start, parent.t_end), (0, 5))
        self.assertEqual(parent.child_ids, ["a", "b"])
        self.assertEqual(parent.actors, ["Alice", "BOB"])
        self.assertEqual(parent.raw_ref_ids, ["a.jpg", "b.jpg"])
        self.assertEqual(parent.merge_mode, "soft")
        pooled = deserialize_embedding(parent.text_embedding)
        self.assertGreater(pooled[1], pooled[0])
        self.assertEqual(parent.writer_model_revision, "pooled-no-generation")
        self.assertEqual(parent.generated_tokens, 0)

    def test_hard_parent_compacts_ids_without_inventing_raw_refs(self):
        first = child("a", 0, 1, (1, 0), raw_ref_ids=["a.jpg"])
        second = child("b", 1, 2, (1, 0), raw_ref_ids=["b.jpg"])
        parent = build_parent_card(
            [first, second], hard=True, compacted_child_count=7,
        )
        self.assertEqual(parent.child_ids, [])
        self.assertEqual(parent.compacted_child_count, 7)
        self.assertEqual(set(parent.raw_ref_ids), {"a.jpg", "b.jpg"})
        self.assertEqual(parent.merge_mode, "hard")


if __name__ == "__main__":
    unittest.main()

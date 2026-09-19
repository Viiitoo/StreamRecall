import unittest

import numpy as np

from streamtimelens.config import ProtocolConfig
from streamtimelens.observer.clip_encoder import serialize_embedding
from streamtimelens.retrieval.ranker import RankWeights, hybrid_rank_cards


def card(card_id, vector, *, level=0, left=False, right=False, visual=None):
    return {
        "id": card_id, "t_start": 1.0 + level, "t_end": 2.0 + level,
        "level": level, "text_embedding": serialize_embedding(vector),
        "visual_centroid": None if visual is None else serialize_embedding(visual),
        "raw_ref_ids": ["x.jpg"] if left or right else [],
        "raw_ref_status": {"x.jpg": "available"} if left or right else {},
        "boundary_cache": {"left_hit": left, "right_hit": right, "both_hit": left and right},
    }


class HybridRankerTest(unittest.TestCase):
    def test_hybrid_components_and_tie_break_are_persistable(self):
        cards = [
            card("parent", [1, 0], level=1, left=True, right=True, visual=[0, 1]),
            card("leaf", [0.8, 0.2], visual=[1, 0]),
        ]
        weights = RankWeights(text=1.0, visual=0.2, boundary_bonus=0.1, parent_penalty=0.2)
        ranked = hybrid_rank_cards(
            np.asarray([1, 0]), cards, weights=weights,
            query_visual_embedding=np.asarray([1, 0]), upper_bound_s=5,
        )
        self.assertEqual(ranked[0].card_id, "leaf")
        self.assertEqual(set(ranked[0].diagnostics), {
            "text_cosine", "visual_cosine", "raw_boundary_completeness",
            "text_component", "visual_component", "boundary_component",
            "parent_penalty", "total",
        })
        self.assertEqual(ranked[1].diagnostics["raw_boundary_completeness"], 1.0)

    def test_missing_embedding_future_span_and_invalid_config_fail(self):
        with self.assertRaisesRegex(ValueError, "text embedding"):
            hybrid_rank_cards([1, 0], [{"id": "x", "t_start": 0, "t_end": 1}], weights=RankWeights())
        with self.assertRaisesRegex(ValueError, "outside"):
            hybrid_rank_cards([1, 0], [card("future", [1, 0])], weights=RankWeights(), upper_bound_s=1.5)
        with self.assertRaisesRegex(ValueError, "rank weights"):
            ProtocolConfig(rank_text_weight=-1)


if __name__ == "__main__":
    unittest.main()

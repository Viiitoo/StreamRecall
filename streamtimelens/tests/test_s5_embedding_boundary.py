import unittest

import numpy as np

from streamtimelens.memory.boundary_allocator import BoundaryCandidate, BoundaryFrameAllocator
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.raw_cache import RawFrameCache
from streamtimelens.observer.clip_encoder import deserialize_embedding
from streamtimelens.protocol.types import FramePacket
from streamtimelens.retrieval.embedder import CardTextEmbedder, card_embedding_text, query_embedding_text


def packet(index, timestamp=None, payload=None):
    return FramePacket(
        float(index if timestamp is None else timestamp), index,
        payload or bytes([index + 1]) * (10 + index), 1, 1, video_id="v",
    )


class TextEmbeddingTest(unittest.TestCase):
    def test_card_and_request_use_same_template_family_and_fp16_unit_vectors(self):
        class Backend:
            def encode(self, texts):
                return np.asarray([[len(text), index + 1, 2] for index, text in enumerate(texts)], dtype=np.float32)

        cards = [
            EvidenceCard("a", 0, 0, 1, "door", actors=["person"], actions=["open"], objects=["door"]),
            EvidenceCard("b", 0, 1, 2, "walk", actors=["person"], actions=["walk"]),
        ]
        embedder = CardTextEmbedder("minilm-fixture", revision="rev", backend=Backend())
        vectors = embedder.embed_cards(cards)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        self.assertEqual(cards[0].text_embedding["dtype"], "fp16")
        self.assertAlmostEqual(float(np.linalg.norm(deserialize_embedding(cards[0].text_embedding))), 1, places=5)
        self.assertEqual(cards[0].writer_provenance["text_embedding"]["model_name"], "minilm-fixture")
        self.assertIn("video evidence", card_embedding_text(cards[0]))
        self.assertIn("video evidence", query_embedding_text("opens a door"))
        self.assertAlmostEqual(float(np.linalg.norm(embedder.encode_query("opens a door"))), 1, places=6)


class BoundaryAllocatorTest(unittest.TestCase):
    def setUp(self):
        self.store = RawFrameCache()
        self.candidates = []
        for index in range(9):
            ref = self.store.add(packet(index), owner="active_segment")
            self.candidates.append(BoundaryCandidate(ref.frame_id, ref.timestamp_s, novelty=index / 10))

    def test_endpoint_nearest_overlap_internal_and_status_metadata(self):
        card = EvidenceCard("c", 0, 3.5, 4.5, "short", left_uncertainty_s=2, right_uncertainty_s=1)
        allocation = BoundaryFrameAllocator(self.store).allocate(card, self.candidates, budget_bytes=10_000)
        self.assertLessEqual(len(allocation.left_frame_ids), 4)
        self.assertLessEqual(len(allocation.right_frame_ids), 4)
        self.assertLessEqual(len(allocation.internal_frame_ids), 2)
        self.assertTrue(allocation.both_hit)
        self.assertTrue(set(allocation.left_frame_ids) & set(allocation.right_frame_ids))
        for frame_id in allocation.selected_frame_ids:
            self.assertIn("card:c", self.store.owners(frame_id))
            self.assertEqual(card.raw_ref_status[frame_id], "available")

    def test_zero_budget_one_frame_and_release_are_explicit(self):
        allocator = BoundaryFrameAllocator(self.store)
        card = EvidenceCard("zero", 0, 4, 4, "instant", left_uncertainty_s=3)
        allocation = allocator.allocate(card, [self.candidates[4]], budget_bytes=0)
        self.assertEqual(allocation.selected_frame_ids, ())
        self.assertEqual(card.raw_ref_status[self.candidates[4].frame_id], "budget_rejected")
        admitted = EvidenceCard("one", 0, 4, 4, "instant", left_uncertainty_s=3)
        allocation = allocator.allocate(admitted, [self.candidates[4]], budget_bytes=100)
        self.assertEqual(len(allocation.selected_frame_ids), 1)
        allocator.release_card(admitted)
        self.assertEqual(admitted.raw_ref_status[allocation.selected_frame_ids[0]], "evicted")


if __name__ == "__main__":
    unittest.main()

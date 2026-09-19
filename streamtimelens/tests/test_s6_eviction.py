import unittest

from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.eviction import UtilityEvictor, UtilityWeights
from streamtimelens.memory.forest import EventForest
from streamtimelens.memory.raw_cache import RawFrameCache
from streamtimelens.protocol.types import FramePacket


def card(card_id, start, *, summary_size=30, raw_ref=None):
    kwargs = {}
    if raw_ref:
        kwargs.update(
            raw_ref_ids=[raw_ref], raw_ref_status={raw_ref: "available"},
            boundary_cache={
                "left_frame_ids": [raw_ref], "right_frame_ids": [raw_ref],
                "internal_frame_ids": [], "left_hit": True, "right_hit": True,
                "both_hit": True,
            },
        )
    return EvidenceCard(card_id, 0, start, start + 0.5, card_id * summary_size, **kwargs)


class UtilityEvictionTest(unittest.TestCase):
    def test_utility_has_all_four_terms_and_no_age_reward(self):
        forest = EventForest(64 * 1024)
        forest.insert(card("old", 0))
        forest.insert(card("new", 100))
        weights = UtilityWeights(novelty=1, boundary=1, inverse_density=1, has_raw=1)
        evictor = UtilityEvictor(weights)
        old = evictor.score(forest.store.get("old"), forest.store)
        new = evictor.score(forest.store.get("new"), forest.store)
        self.assertEqual(old, new)
        self.assertEqual(
            old.total,
            old.novelty + old.boundary + old.inverse_density + old.has_raw,
        )

    def test_raw_is_evicted_before_card_and_status_is_explicit(self):
        raw = RawFrameCache()
        frame = FramePacket(0, 0, b"x" * 200, 1, 1, video_id="v")
        cached = raw.add(frame, owner="card:a")
        forest = EventForest(64 * 1024, raw_cache=raw)
        forest.insert(card("a", 0, raw_ref=cached.frame_id))
        record = forest.evictor.evict_raw(forest)
        self.assertEqual(record.kind, "raw")
        self.assertIn("a", forest.store)
        self.assertEqual(forest.store.get("a").raw_ref_status[cached.frame_id], "evicted")
        self.assertNotIn(cached.frame_id, forest.store.get("a").raw_ref_ids)

    def test_log_buckets_keep_earliest_anchor_during_repeated_card_eviction(self):
        forest = EventForest(128 * 1024)
        for index, start in enumerate((0, 0.1, 2, 3, 8, 9, 32, 33)):
            forest.insert(card(f"c{index}", start))
        earliest = forest.roots[0].id
        for _ in range(6):
            record = forest.evictor.evict_card(forest)
            self.assertIsNotNone(record)
        self.assertIn(earliest, forest.store)
        self.assertEqual(len(forest.roots), 2)


if __name__ == "__main__":
    unittest.main()

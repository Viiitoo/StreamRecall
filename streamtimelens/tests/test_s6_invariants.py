import random
import tempfile
import unittest
from pathlib import Path

from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.forest import EventForest
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, VideoMeta


def card(index):
    start = index * 0.25
    return EvidenceCard(
        f"c{index:06d}", 0, start, start + 0.2, f"event {index % 17}",
        actors=[f"actor-{index % 5}"], actions=[f"action-{index % 11}"],
        source_chunk_ids=[f"chunk-{index:06d}"], normalized_text=f"event {index % 17}",
    )


class ForestInvariantPropertyTest(unittest.TestCase):
    def test_random_insert_merge_evict_snapshot_restore_for_1000_updates(self):
        rng = random.Random(20260829)
        limit = 48 * 1024
        forest = EventForest(limit, debug=True)
        current_time = 0.0
        next_id = 0
        with tempfile.TemporaryDirectory() as temporary:
            writer = SnapshotWriter(Path(temporary))
            for step in range(1000):
                operation = rng.randrange(5)
                if operation <= 2 or not forest.roots:
                    item = card(next_id)
                    next_id += 1
                    current_time = item.t_end
                    forest.insert(item)
                elif operation == 3 and len(forest.roots) >= 2:
                    forest.merge_best(force_hard=bool(rng.randrange(2)))
                else:
                    forest.evictor.evict_card(forest)
                forest.enforce_budget(current_stream_time=current_time)
                if step % 100 == 99:
                    manifest = writer.write(
                        name=f"s{step}", t_q=current_time,
                        meta=VideoMeta("v", max(300.0, current_time), 4, 1200),
                        budget=Budget(512 * 1024, 1), cards=forest.store.rows(),
                        raw_frames=[], raw_metadata={}, writer_calls=0,
                        config={"property_seed": 20260829},
                    )
                    restored = EventForest(limit, debug=True)
                    restored.restore(SnapshotReader(Path(temporary) / f"s{step}").read_cards())
                    self.assertEqual(manifest.writer_calls, 0)
                    forest = restored
                forest.assert_invariants(current_stream_time=current_time)

    def test_ten_thousand_step_stream_stays_bounded_and_retains_early_coverage(self):
        limit = 12 * 1024
        forest = EventForest(limit)
        peak = 0
        for index in range(10_000):
            item = card(index)
            forest.insert(item)
            while len(forest.roots) > 8:
                forest.merge_best()
            forest.enforce_budget(current_stream_time=item.t_end)
            peak = max(peak, forest.state_bytes)
        self.assertLessEqual(peak, limit)
        self.assertTrue(forest.cards)
        # At least one compact root continues to cover the first logarithmic
        # time bucket instead of turning the memory into a recent-only cache.
        self.assertTrue(any(item.t_start < 1.0 for item in forest.roots))
        forest.assert_invariants(current_stream_time=card(9999).t_end)


if __name__ == "__main__":
    unittest.main()

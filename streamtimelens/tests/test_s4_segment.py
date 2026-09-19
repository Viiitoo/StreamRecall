import unittest

import numpy as np

from streamtimelens.memory.raw_cache import RawFrameCache
from streamtimelens.observer.segment import ActiveSegmentReservoir
from streamtimelens.observer.trigger import JointTrigger
from streamtimelens.protocol.types import FramePacket


def packet(timestamp):
    index = int(timestamp)
    return FramePacket(timestamp, index, bytes([index % 256]), 1, 1, video_id="v")


class ActiveSegmentReservoirTest(unittest.TestCase):
    def test_capacity_is_selected_from_remaining_budget(self):
        store = RawFrameCache()
        segment = ActiveSegmentReservoir(store, remaining_budget_bytes=32 * 100, bytes_per_frame_estimate=100)
        self.assertEqual(segment.capacity, 32)
        self.assertEqual(segment.resize_for_budget(1600), 16)
        self.assertEqual(segment.resize_for_budget(799), 8)

    def test_sixty_second_segment_has_true_endpoints_and_is_max_gap_writable(self):
        store = RawFrameCache()
        segment = ActiveSegmentReservoir(store, remaining_budget_bytes=32 * 1024, bytes_per_frame_estimate=1024)
        trigger = JointTrigger("joint", threshold=100, minimum_gap_s=0, max_gap_s=60)
        for timestamp in range(61):
            embedding = np.array((1.0, timestamp / 60.0 + 0.01), dtype=np.float32)
            segment.observe(packet(timestamp), embedding)
            proposal = trigger.propose(timestamp, lite_score=0, semantic_score=0)
        self.assertEqual(segment.span, (0.0, 60.0))
        self.assertLessEqual(len(segment), 32)
        self.assertIn("rule:max_gap", proposal.reasons)
        self.assertTrue(segment.writable)
        self.assertEqual((segment.writer_packets()[0].timestamp_s, segment.writer_packets()[-1].timestamp_s), (0.0, 60.0))

    def test_writer_retains_one_second_overlap_and_releases_other_refs(self):
        store = RawFrameCache()
        segment = ActiveSegmentReservoir(store, remaining_budget_bytes=32 * 1024, bytes_per_frame_estimate=1024)
        for timestamp in range(61):
            frame = packet(timestamp)
            store.add(frame, owner="now_ring")
            segment.observe(frame, novelty_score=float(timestamp))
            if timestamp < 59:
                store.release(frame.frame_index.__format__("09d") + ".jpg", "now_ring")
        old_ids = set(segment.frame_ids)
        kept = set(segment.mark_written(60))
        self.assertTrue(kept)
        self.assertTrue(all(int(frame_id[:9]) >= 59 for frame_id in kept))
        for frame_id in old_ids - kept:
            self.assertNotIn("active_segment", store.owners(frame_id))
        for frame_id in kept:
            self.assertIn("active_segment", store.owners(frame_id))
            self.assertIn("now_ring", store.owners(frame_id))


if __name__ == "__main__":
    unittest.main()

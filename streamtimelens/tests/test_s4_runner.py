import unittest

from streamtimelens.protocol.types import Budget, FramePacket, VideoMeta
from streamtimelens.stream.runner import IngestConfig, StreamingIngestor


def packet(timestamp):
    index = int(timestamp)
    return FramePacket(timestamp, index, bytes([index % 256]), 1, 1, video_id="v")


class S4RunnerIntegrationTest(unittest.TestCase):
    def test_sixty_second_max_gap_write_has_endpoints_and_overlap_refs(self):
        ingestor = StreamingIngestor(
            VideoMeta("v", 60, 1, 61), Budget(256 * 1024, 1),
            IngestConfig(
                decision_interval_s=120, max_gap_s=60, minimum_gap_s=0,
                segment_bytes_per_frame=1024,
            ),
        )
        for timestamp in range(61):
            ingestor.observe(packet(timestamp))
        self.assertEqual(ingestor.ledger.writer_calls, 1)
        self.assertEqual((ingestor.cards[0].t_start, ingestor.cards[0].t_end), (0.0, 60.0))
        self.assertEqual(ingestor.segment.span, (59.0, 60.0))
        for frame_id in ingestor.segment.frame_ids:
            self.assertIn("active_segment", ingestor.raw.owners(frame_id))
            self.assertIn("now_ring", ingestor.raw.owners(frame_id))
        max_gap = [row for row in ingestor.trace if row["kind"] == "writer_called"]
        self.assertIn("rule:max_gap", max_gap[0]["trigger_reasons"])

    def test_quota_drops_are_traced_with_admission_reason(self):
        ingestor = StreamingIngestor(
            VideoMeta("v", 4, 1, 5), Budget(128 * 1024, 1),
            IngestConfig(
                decision_interval_s=1, minimum_gap_s=0, max_gap_s=60,
                segment_bytes_per_frame=1024,
            ),
        )
        for timestamp in range(5):
            ingestor.observe(packet(timestamp))
        drops = [row for row in ingestor.trace if row["kind"] == "trigger_dropped"]
        self.assertTrue(any(row["reason"].startswith("writer_quota:") for row in drops))


if __name__ == "__main__":
    unittest.main()

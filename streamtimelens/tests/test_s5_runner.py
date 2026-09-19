import json
import unittest

from streamtimelens.protocol.types import Budget, FramePacket, VideoMeta
from streamtimelens.stream.runner import IngestConfig, StreamingIngestor
from streamtimelens.writer.card_builder import build_evidence_cards
from streamtimelens.writer.parse import parse_writer_output
from streamtimelens.writer.prompts import build_writer_prompt
from streamtimelens.writer.timelens_writer import WriterCallResult


def packet(timestamp):
    return FramePacket(timestamp, int(timestamp), bytes([int(timestamp) + 1]) * 20, 1, 1, video_id="v")


class S5RunnerIntegrationTest(unittest.TestCase):
    def test_valid_multi_event_writer_embedding_boundary_and_trace_are_integrated(self):
        class Writer:
            def write(inner, chunk_id, frames, meta):
                span = (frames[0].timestamp_s, frames[-1].timestamp_s)
                support = [frame.timestamp_s for frame in frames]
                raw = json.dumps({"segment": list(span), "events": [{
                    "summary": "person moves", "actors": ["person"], "actions": ["move"],
                    "objects": [], "scene": "room", "span": list(span), "phase": "complete",
                    "visual_support": support,
                }]})
                parsed = parse_writer_output(raw, segment=span, sampled_timestamps=support)
                prompt = build_writer_prompt(segment=span, sampled_timestamps=support)
                cards = build_evidence_cards(
                    parsed, source_chunk_id=chunk_id, segment=span, prompt=prompt,
                    writer_revision="fixture", generated_tokens=5,
                )
                return WriterCallResult(tuple(cards), "valid", raw, prompt, {"generated_tokens": 5})

        class Embedder:
            def embed_cards(inner, cards):
                for card in cards:
                    card.text_embedding = {"format_version": 1, "dtype": "fp16", "length": 1,
                                           "scale": None, "data_b64": "ADw="}

        ingestor = StreamingIngestor(
            VideoMeta("v", 3, 1, 4), Budget(128 * 1024, 60),
            IngestConfig(decision_interval_s=1, minimum_gap_s=0, max_gap_s=60,
                         segment_bytes_per_frame=1024),
            evidence_writer=Writer(), text_embedder=Embedder(),
        )
        for timestamp in range(3):
            ingestor.observe(packet(timestamp))
        self.assertTrue(ingestor.cards)
        self.assertIsNotNone(ingestor.cards[0].text_embedding)
        self.assertTrue(ingestor.cards[0].boundary_cache["both_hit"])
        called = [row for row in ingestor.trace if row["kind"] == "writer_called"]
        self.assertEqual(called[0]["parse_status"], "valid")
        self.assertIn("raw_output", called[0])
        self.assertTrue(called[0]["boundary_allocations"])


if __name__ == "__main__":
    unittest.main()

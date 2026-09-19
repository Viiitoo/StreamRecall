import tempfile
import unittest
from pathlib import Path

import numpy as np

from streamtimelens.baselines.vst_summary import (
    VST_MAX_NEW_TOKENS, VSTSummaryIngestor, VSTSummaryMemory, VSTIngestConfig,
    SummaryCall, build_vst_prompt,
)
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.raw_cache import RawFrameCache
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, FramePacket, VideoMeta
from streamtimelens.retrieval.embedder import CardTextEmbedder


class Backend:
    def encode(self, texts):
        return np.asarray([[len(text), 1] for text in texts], dtype=np.float32)


class Writer:
    def summarize(self, frames, meta):
        del meta
        text = f"visible segment {frames[0].timestamp_s}-{frames[-1].timestamp_s}"
        return SummaryCall(text, text, 4, {"component": "fixture", "status": "ok"})


def packet(index):
    return FramePacket(float(index), index, bytes([index + 1]) * 20, 1, 1, video_id="v")


class VSTSummaryTest(unittest.TestCase):
    def test_prompt_is_free_text_query_blind_and_token_bounded(self):
        prompt = build_vst_prompt((1, 3), [1, 2, 3])
        self.assertIn("free text", prompt)
        self.assertIn("do not output JSON", " ".join(prompt.split()))
        self.assertNotIn("actors/actions/objects", prompt)
        self.assertNotIn("query", prompt.lower())
        self.assertEqual(VST_MAX_NEW_TOKENS, 256)
        with self.assertRaisesRegex(ValueError, "inside"):
            build_vst_prompt((1, 3), [0, 2])

    def test_fifo_and_first_recent_have_distinct_bounded_eviction(self):
        fifo = VSTSummaryMemory(10_000, RawFrameCache(), policy="fifo")
        first_recent = VSTSummaryMemory(10_000, RawFrameCache(), policy="first_recent")
        for memory in (fifo, first_recent):
            for index in range(3):
                memory.insert(EvidenceCard(str(index), 0, index, index + 1, "x" * 20))
            memory.limit_bytes = memory.state_bytes - 1
            memory.enforce()
        self.assertEqual([card.id for card in fifo.cards], ["1", "2"])
        self.assertEqual([card.id for card in first_recent.cards], ["0", "2"])
        self.assertLessEqual(fifo.state_bytes, fifo.limit_bytes)
        self.assertIn("summary_payload_and_embedding", fifo.accounting())

    def test_single_pass_runner_uses_shared_budget_and_common_card_snapshot(self):
        meta = VideoMeta("v", 10, 1, 10)
        embedder = CardTextEmbedder("fixture", revision="rev", backend=Backend())
        ingestor = VSTSummaryIngestor(
            meta, Budget(65_536, 60), VSTIngestConfig(segment_interval_s=2, max_frames=8),
            text_embedder=embedder, writer=Writer(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = ingestor.run(
                [packet(index) for index in range(10)], [5, 10], SnapshotWriter(root),
                config={"method": "vst_summary"}, snapshot_prefix="v",
            )
            reader = SnapshotReader(root / "v" / "rho_0.50")
            cards = reader.read_cards()
        self.assertEqual(len(manifests), 2)
        self.assertEqual(reader.manifest.method, "vst_summary")
        self.assertEqual(reader.manifest.embedder["model_name"], "fixture")
        self.assertTrue(cards)
        self.assertTrue(all(card["prompt_version"] == "vst_observation_v1" for card in cards))
        self.assertLessEqual(reader.manifest.state_bytes, reader.manifest.budget_bytes)
        self.assertGreater(ingestor.ledger.writer_calls, 0)


if __name__ == "__main__":
    unittest.main()

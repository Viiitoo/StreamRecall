import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from streamtimelens.baselines.oasis_adapt import (
    OASISAdaptIngestor, OASISIngestConfig, OASISMemory,
)
from streamtimelens.baselines.vst_summary import SummaryCall
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
        text = f"summary {frames[0].frame_index}-{frames[-1].frame_index}"
        return SummaryCall(text, text, 3, {"component": "fixture", "status": "ok"})


def packet(index):
    return FramePacket(float(index), index, bytes([index + 1]) * 16, 1, 1, video_id="v")


class OASISAdaptTest(unittest.TestCase):
    def test_adjacent_merge_keeps_all_soft_nodes_accounted(self):
        raw = RawFrameCache()
        memory = OASISMemory(100_000, raw, max_roots=2)
        embedder = CardTextEmbedder("fixture", revision="rev", backend=Backend())
        leaves = [EvidenceCard(str(i), 0, i * 2, i * 2 + 1, f"event {i}") for i in range(3)]
        embedder.embed_cards(leaves)
        for card in leaves:
            memory.insert(card)
        self.assertEqual(len(memory.forest.roots), 2)
        self.assertEqual(len(memory.cards), 4)
        parent = next(card for card in memory.cards if card.level == 1)
        self.assertEqual(len(parent.child_ids), 2)
        self.assertEqual(memory.state_bytes, sum(memory.accounting().values()))
        memory.forest.assert_invariants(current_stream_time=5)

    def test_online_runner_fixed_32_cap_snapshot_and_no_video_preload_api(self):
        self.assertNotIn("video_path", inspect.signature(OASISAdaptIngestor.run).parameters)
        meta = VideoMeta("v", 12, 1, 12)
        embedder = CardTextEmbedder("fixture", revision="rev", backend=Backend())
        ingestor = OASISAdaptIngestor(
            meta, Budget(100_000, 60),
            OASISIngestConfig(segment_interval_s=2, max_frames=32, max_roots=2),
            text_embedder=embedder, writer=Writer(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ingestor.run(
                [packet(index) for index in range(12)], [6, 12], SnapshotWriter(root),
                config={"method": "oasis_adapt"}, snapshot_prefix="v",
            )
            reader = SnapshotReader(root / "v" / "rho_1.00")
            rows = reader.read_cards()
        self.assertEqual(reader.manifest.method, "oasis_adapt")
        self.assertTrue(any(row["level"] > 0 for row in rows))
        self.assertLessEqual(reader.manifest.state_bytes, reader.manifest.budget_bytes)
        self.assertTrue(any(row["kind"] == "nodes_merged" for row in ingestor.trace))

    def test_invalid_root_cap_fails(self):
        with self.assertRaisesRegex(ValueError, "hierarchy"):
            OASISIngestConfig(max_roots=0)


if __name__ == "__main__":
    unittest.main()

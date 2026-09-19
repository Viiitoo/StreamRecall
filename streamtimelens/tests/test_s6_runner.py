import tempfile
import unittest
from pathlib import Path

from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, FramePacket, VideoMeta
from streamtimelens.stream.runner import IngestConfig, StreamingIngestor


def packet(timestamp):
    return FramePacket(timestamp, int(timestamp), bytes([int(timestamp) + 1]) * 20, 1, 1, video_id="v")


class S6RunnerIntegrationTest(unittest.TestCase):
    def test_forest_merge_trace_and_topology_enter_snapshot_not_trace(self):
        ingestor = StreamingIngestor(
            VideoMeta("v", 8, 1, 9), Budget(256 * 1024, 60),
            IngestConfig(
                decision_interval_s=1, minimum_gap_s=0, max_gap_s=60,
                segment_bytes_per_frame=1024, forest_max_roots=2, forest_debug=True,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = ingestor.run(
                [packet(value) for value in range(9)], [8], SnapshotWriter(root),
                config={"forest_max_roots": 2}, snapshot_prefix="v",
            )
            reader = SnapshotReader(root / "v" / "rho_1.00")
            rows = reader.read_cards()
        merges = [row for row in ingestor.trace if row["kind"] == "card_merged"]
        self.assertTrue(merges)
        self.assertTrue(any(row["merge_reason"]["semantic_similarity"] >= 0 for row in merges))
        ids = {row["id"] for row in rows}
        self.assertTrue(all(set(row["child_ids"]) <= ids for row in rows))
        self.assertLessEqual(len(ingestor.forest.roots), 2)
        self.assertLessEqual(manifests[8.0].state_bytes, 256 * 1024)
        self.assertNotIn("trace", reader.manifest.allowed_files)


if __name__ == "__main__":
    unittest.main()

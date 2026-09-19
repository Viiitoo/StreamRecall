import tempfile
import unittest
from pathlib import Path

import numpy as np

from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, VideoMeta
from streamtimelens.retrieval.embedder import (
    EMBEDDING_TEMPLATE_SHA256,
    EMBEDDING_TEMPLATE_VERSION,
    load_snapshot_query_embedder,
    timed_query_embedding,
)


class Backend:
    def encode(self, texts):
        return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float32)


def snapshot(root: Path, *, embedder=None) -> SnapshotReader:
    SnapshotWriter(root).write(
        name="s", t_q=5.0,
        meta=VideoMeta("v", 10.0, 2.0, 20),
        budget=Budget(16_384, 1), cards=[], raw_frames=[], writer_calls=0,
        config={}, embedder=embedder,
    )
    return SnapshotReader(root / "s")


class QueryEmbeddingTest(unittest.TestCase):
    def test_exact_snapshot_embedder_and_timing(self):
        metadata = {
            "model_name": "fixture/minilm", "revision": "fixed-rev",
            "template_version": EMBEDDING_TEMPLATE_VERSION,
            "template_sha256": EMBEDDING_TEMPLATE_SHA256, "dtype": "fp16",
        }
        with tempfile.TemporaryDirectory() as temporary:
            reader = snapshot(Path(temporary), embedder=metadata)
            embedder = load_snapshot_query_embedder(reader, backend=Backend())
            result = timed_query_embedding(embedder, "opens a door")
        self.assertAlmostEqual(float(np.linalg.norm(result.vector)), 1.0, places=6)
        self.assertEqual(result.resource["component"], "query_embedding")
        self.assertEqual(result.resource["status"], "ok")

    def test_mismatch_and_missing_metadata_are_rejected(self):
        metadata = {
            "model_name": "fixture/minilm", "revision": "fixed-rev",
            "template_version": EMBEDDING_TEMPLATE_VERSION,
            "template_sha256": EMBEDDING_TEMPLATE_SHA256, "dtype": "fp16",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = snapshot(root / "declared", embedder=metadata)
            with self.assertRaisesRegex(ValueError, "exactly match"):
                load_snapshot_query_embedder(reader, model_name_or_path="other", backend=Backend())
            missing = snapshot(root / "missing")
            with self.assertRaisesRegex(ValueError, "does not declare"):
                load_snapshot_query_embedder(missing, backend=Backend())


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from streamtimelens.observer.clip_encoder import FrozenCLIPEncoder, serialize_embedding
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, FramePacket, VideoMeta
from streamtimelens.retrieval.frame_candidates import (
    build_frame_candidates, coarse_prediction, refine_frame_candidate,
)
from streamtimelens.stream.baseline_runner import BaselineIngestConfig, RawBaselineIngestor


def packet(index, timestamp, value=0, *, video_id="v", size=(16, 24)):
    image = Image.fromarray(np.full((size[0], size[1], 3), value, dtype=np.uint8), mode="RGB")
    output = BytesIO()
    image.save(output, format="PNG")
    return FramePacket(timestamp, index, output.getvalue(), size[1], size[0], "fixture", video_id)


def embedding(index, dimensions=4):
    vector = np.zeros(dimensions, dtype=np.float32)
    vector[index % dimensions] = 1.0
    return vector


class FakeClipBackend:
    def encode_images(self, images):
        return np.asarray([[float(np.asarray(image).mean()), 1.0, 2.0] for image in images])

    def encode_texts(self, texts):
        return np.asarray([[len(text), 1.0, 2.0] for text in texts], dtype=np.float32)


class CandidateAndIntegrationTest(unittest.TestCase):
    def test_topk_merge_neighbor_expansion_and_coarse_output(self):
        rows = {}
        vectors = ([1, 0], [.9, .1], [0, 1], [.8, .2])
        for index, (timestamp, vector) in enumerate(zip((0, 2, 10, 12), vectors)):
            rows[f"{index:09d}.jpg"] = {
                "timestamp_s": timestamp, "frame_index": index,
                "clip_embedding": serialize_embedding(vector),
            }
        candidates = build_frame_candidates([1, 0], rows, top_k=2, merge_gap_s=3, expand_neighbors=1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].hit_refs, ("000000000.jpg", "000000001.jpg"))
        self.assertEqual(candidates[0].frame_refs, ("000000000.jpg", "000000001.jpg", "000000002.jpg"))
        prediction = coarse_prediction(candidates)
        self.assertEqual(prediction.status, "fallback")
        self.assertEqual(prediction.candidate_ids, (candidates[0].candidate_id,))

    def test_both_baselines_share_snapshot_and_candidate_query_path(self):
        class FakeEncoder:
            def __init__(self):
                self.encoder = FrozenCLIPEncoder(backend=FakeClipBackend(), batch_size=8)
                self.batch_size = self.encoder.batch_size
            def due(self, timestamp): return self.encoder.due(timestamp)
            def encode_images(self, images): return self.encoder.encode_images(images)
            def encode_text(self, text): return self.encoder.encode_text(text)
            def resource_dicts(self): return self.encoder.resource_dicts()

        meta = VideoMeta("v", 8, 1, 9)
        frames = [packet(index, index, index * 20) for index in range(9)]
        for method in ("uniform_raw", "semantic_reservoir"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                encoder = FakeEncoder()
                ingestor = RawBaselineIngestor(
                    meta, Budget(256 * 1024, 1), BaselineIngestConfig(method, 4),
                    clip_encoder=encoder,
                )
                manifests = ingestor.run(
                    iter(frames), [4, 8], SnapshotWriter(Path(temporary)),
                    config={"method": method}, snapshot_prefix="v",
                )
                self.assertEqual(set(manifests), {4.0, 8.0})
                reader = SnapshotReader(Path(temporary) / "v" / "rho_0.50")
                self.assertEqual(reader.manifest.method, method)
                metadata = reader.read_frame_metadata()
                self.assertTrue(metadata)
                self.assertTrue(all(row["timestamp_s"] <= 4 for row in metadata.values()))
                candidates = build_frame_candidates(encoder.encode_text("door"), metadata)
                self.assertTrue(candidates)

    def test_sparse_local_refiner_uses_snapshot_only_and_traces_typed_fallback(self):
        meta = VideoMeta("v", 4, 1, 5)
        tiny = BytesIO()
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(tiny, format="PNG")
        refs = []
        metadata = {}
        frames = []
        for index in (1, 2, 3):
            name = f"{index:09d}.png"
            refs.append(name)
            frames.append((name, tiny.getvalue()))
            metadata[name] = {
                "timestamp_s": float(index), "frame_index": index, "sha256": None,
                "clip_embedding": serialize_embedding(embedding(index)),
            }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            SnapshotWriter(root).write(
                name="s", t_q=4, meta=meta, budget=Budget(64 * 1024, 1), cards=[],
                raw_frames=frames, raw_metadata=metadata, writer_calls=0, config={}, method="uniform_raw",
            )
            # Omit optional SHA in this hand-built legacy-style metadata.
            manifest_meta = root / "s" / "frame_metadata.json"
            data = json.loads(manifest_meta.read_text())
            for row in data.values():
                row.pop("sha256", None)
            manifest_meta.write_text(json.dumps(data, sort_keys=True))
            # Rebuild through the writer instead of bypassing manifest integrity.
            SnapshotWriter(root).write(
                name="s", t_q=4, meta=meta, budget=Budget(64 * 1024, 1), cards=[],
                raw_frames=frames, raw_metadata=data, writer_calls=0, config={}, method="uniform_raw",
            )
            reader = SnapshotReader(root / "s")
            candidate = build_frame_candidates(embedding(1), reader.read_frame_metadata(), top_k=3)[0]
            class Service:
                last_call_stats = {"generated_tokens": 5}
                def generate(self, messages, videos, max_new_tokens):
                    return "no usable timestamp"
            prediction, trace = refine_frame_candidate("opens", reader, candidate, Service(), max_frames=3)
            self.assertEqual(prediction.status, "fallback")
            self.assertEqual(trace["kind"], "refiner_fallback")
            self.assertEqual(trace["parse_status"], "no_timestamp")


if __name__ == "__main__":
    unittest.main()

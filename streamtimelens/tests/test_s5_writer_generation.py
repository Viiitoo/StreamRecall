import json
import unittest

import numpy as np

from streamtimelens.evaluation.writer_generation import (
    LoadedChunkFrames, WriterChunk, generate_writer_record,
)


class WriterFeasibilityGenerationTest(unittest.TestCase):
    def test_generation_is_query_blind_and_preserves_fixed_record_contract(self):
        chunk = WriterChunk.from_mapping({
            "chunk_id": "chunk-1", "video_id": "video-1", "video_path": "/video.mp4",
            "segment": [0, 2], "sampled_timestamps": [0, 1, 2],
            "gt_spans": [[0.5, 1.5]],
        })

        class Service:
            last_call_stats = {"generated_tokens": 7}
            processor = object()

            def generate(self, messages, videos, max_new_tokens):
                prompt = messages[0]["content"][1]["text"]
                self.prompt = prompt
                return json.dumps({"segment": [0, 2], "events": []})

        service = Service()

        def load_frames(item):
            self.assertEqual(item.chunk_id, "chunk-1")
            return LoadedChunkFrames(
                tuple(np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)),
                (0, 10, 20), 10.0, 21,
            )

        row = generate_writer_record(
            chunk, model_id="checkpoint", model_revision="revision", service=service,
            frame_loader=load_frames,
        )
        self.assertEqual(row["chunk_id"], chunk.chunk_id)
        self.assertEqual(row["sampled_timestamps"], [0.0, 1.0, 2.0])
        self.assertEqual(row["gt_spans"], [[0.5, 1.5]])
        self.assertGreaterEqual(row["gpu_s"], 0)
        self.assertNotIn("opens the refrigerator", service.prompt.lower())
        self.assertNotIn("ground truth", service.prompt.lower())

    def test_invalid_chunk_fails_before_inference(self):
        with self.assertRaisesRegex(ValueError, "sampled timestamps"):
            WriterChunk.from_mapping({
                "chunk_id": "bad", "video_id": "v", "video_path": "/v.mp4",
                "segment": [0, 2], "sampled_timestamps": [0, 3], "gt_spans": [],
            })


if __name__ == "__main__":
    unittest.main()

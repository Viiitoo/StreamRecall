import tempfile
import unittest
from pathlib import Path

import numpy as np

from streamtimelens.refiner.model_service import TimeLensModelService
from streamtimelens.refiner.prompts import (
    OFFICIAL_CROP_VERSION, SPARSE_LOCAL_VERSION, build_grounding_prompt, video_messages,
)
from streamtimelens.refiner.timelens_adapter import SparseFrame, prepare_sparse_video


class SparseTimeLensAdapterTest(unittest.TestCase):
    def test_preserves_global_indices_and_duplicates_for_processor(self):
        frames = [
            SparseFrame(100, 4.0, np.full((2, 2, 3), 100, dtype=np.uint8)),
            SparseFrame(0, 0.0, np.zeros((2, 2, 3), dtype=np.uint8)),
            SparseFrame(25, 1.0, np.full((2, 2, 3), 25, dtype=np.uint8)),
        ]
        result = prepare_sparse_video(frames, original_fps=25, total_num_frames=200, k_frames=3)
        self.assertEqual(result.unique_indices, (0, 25, 100))
        self.assertEqual(result.timestamps_s, (0.0, 1.0, 4.0))
        self.assertEqual(result.metadata["frames_indices"], [0, 0, 25, 25, 100, 100])
        self.assertEqual(result.video.shape, (6, 3, 2, 2))
        np.testing.assert_array_equal(result.video[2], result.video[3])
        self.assertIn("timestamp=4.0s", result.timestamp_audit[-1])

    def test_deduplicates_and_enforces_limit_and_metadata(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        result = prepare_sparse_video(
            [SparseFrame(0, 0, image), SparseFrame(0, 0, image)],
            original_fps=25, total_num_frames=10, k_frames=1,
        )
        self.assertEqual(result.unique_indices, (0,))
        with self.assertRaisesRegex(ValueError, "exceeds K_frames"):
            prepare_sparse_video([SparseFrame(0, 0, image), SparseFrame(1, .04, image)],
                                 original_fps=25, total_num_frames=10, k_frames=1)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            prepare_sparse_video([SparseFrame(1, 1.0, image)], original_fps=25, total_num_frames=10, k_frames=1)


class RefinerPromptTest(unittest.TestCase):
    def test_sparse_prompt_marks_global_non_contiguous_and_candidate_as_prior(self):
        prompt = build_grounding_prompt(
            SPARSE_LOCAL_VERSION, query="person opens a door", candidate=(10, 20),
            card_summary={"summary": "doorway visible"},
        )
        self.assertIn("non-contiguous", prompt.text)
        self.assertIn("global", prompt.text)
        self.assertIn("coarse search prior only", prompt.text)
        self.assertTrue(prompt.text.endswith("The event happens in x - y seconds"))
        self.assertEqual(prompt, build_grounding_prompt(
            SPARSE_LOCAL_VERSION, query="person opens a door", candidate=(10, 20),
            card_summary={"summary": "doorway visible"},
        ))
        self.assertEqual(video_messages(prompt)[0]["content"][0]["type"], "video")

    def test_official_prompt_and_api_have_no_ground_truth_slot(self):
        prompt = build_grounding_prompt(OFFICIAL_CROP_VERSION, query="walks away")
        self.assertIn("walks away", prompt.text)
        with self.assertRaises(TypeError):
            build_grounding_prompt(OFFICIAL_CROP_VERSION, query="walks away", gt_span=(1, 2))  # type: ignore[call-arg]

    def test_checkpoint_hashes_cover_model_config_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in {
                "model.safetensors": b"weights", "config.json": b"{}", "generation_config.json": b"{}",
                "preprocessor_config.json": b"{}", "tokenizer.json": b"tokens",
                "tokenizer_config.json": b"{}", "vocab.json": b"{}", "merges.txt": b"",
            }.items():
                (root / name).write_bytes(content)
            hashes = TimeLensModelService._compute_hashes(root)
            self.assertEqual(set(hashes), {"model_sha256", "config_sha256", "tokenizer_sha256"})
            self.assertTrue(all(len(value) == 64 for value in hashes.values()))


if __name__ == "__main__":
    unittest.main()

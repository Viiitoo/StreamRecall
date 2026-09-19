import unittest

from baas.dataset import GROUNDER_PROMPT_TEXT_TIMESTAMP, build_messages, prepare_inputs
from baas.sampling import SamplingError, VideoMetadata, uniform_plan


class FakePixels:
    shape = (32, 3)


class FakeInputs(dict):
    pass


class FakeImageProcessor:
    merge_size = 2


class FakeProcessor:
    image_processor = FakeImageProcessor()

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "rendered"

    def __call__(self, **_kwargs):
        return FakeInputs(image_grid_thw=[[1, 4, 4], [1, 4, 4]], pixel_values=FakePixels())


class DatasetAdapterTest(unittest.TestCase):
    def test_timestamp_prefixes_are_interleaved_and_token_count_is_measured(self):
        plan = uniform_plan(VideoMetadata(fps=2, frame_count=4), 8, 42, video_id="v", max_frames=2)
        processor = FakeProcessor()
        inputs, stats = prepare_inputs("unused.mp4", "open the door", plan, processor, frames=[object(), object()])
        content = processor.messages[0]["content"]
        self.assertEqual([entry["type"] for entry in content], ["text", "image", "text", "image", "text"])
        self.assertEqual(content[0]["text"], "0.000000 seconds: ")
        self.assertIn("open the door", content[-1]["text"])
        self.assertEqual(stats.actual_visual_tokens, 8)
        self.assertEqual(stats.pixel_value_patches, 32)

    def test_actual_processor_tokens_over_budget_fail_closed(self):
        plan = uniform_plan(VideoMetadata(fps=2, frame_count=4), 4, 42, video_id="v", max_frames=2)
        with self.assertRaisesRegex(SamplingError, "exceed budget"):
            prepare_inputs(
                "unused.mp4",
                "open the door",
                plan,
                FakeProcessor(),
                frames=[object(), object()],
            )


if __name__ == "__main__":
    unittest.main()

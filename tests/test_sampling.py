import unittest

from baas.sampling import SamplingError, VideoMetadata, build_plan, stable_plan_json, uniform_plan, with_actual_visual_tokens


class SamplingPlanTest(unittest.TestCase):
    def test_uniform_endpoints_are_half_open_and_deterministic(self):
        metadata = VideoMetadata(fps=10, frame_count=10, duration_sec=1.0)
        first = uniform_plan(metadata, 100, 42, video_id="v", max_frames=4)
        second = uniform_plan(metadata, 100, 42, video_id="v", max_frames=4)
        self.assertEqual(first, second)
        self.assertEqual(first.frame_indices, (0, 2, 5, 7))
        self.assertEqual(first.timestamps_sec, (0.0, 0.2, 0.5, 0.7))
        self.assertTrue(all(timestamp < 1.0 for timestamp in first.timestamps_sec))
        self.assertEqual(stable_plan_json(first), stable_plan_json(second))

    def test_short_video_deduplicates_and_fills_indices(self):
        metadata = VideoMetadata(fps=30, frame_count=3, duration_sec=0.1)
        plan = uniform_plan(metadata, 10, 7, video_id="short", max_frames=10)
        self.assertEqual(plan.frame_indices, (0, 1, 2))
        self.assertEqual(len(set(plan.frame_indices)), 3)

    def test_vfr_metadata_uses_frame_pts(self):
        metadata = VideoMetadata(
            fps=30, frame_count=4, duration_sec=1.0,
            frame_timestamps_sec=(0.0, 0.05, 0.70, 0.90),
        )
        plan = uniform_plan(metadata, 10, 0, video_id="vfr", max_frames=3)
        self.assertEqual(plan.frame_indices, (0, 2, 3))
        self.assertEqual(plan.timestamps_sec, (0.0, 0.70, 0.90))

    def test_invalid_budget_and_actual_overflow_are_rejected(self):
        metadata = VideoMetadata(fps=1, frame_count=2)
        with self.assertRaises(SamplingError):
            uniform_plan(metadata, 0, 0, video_id="x")
        plan = uniform_plan(metadata, 4, 0, video_id="x", max_frames=1)
        with self.assertRaises(SamplingError):
            with_actual_visual_tokens(plan, 5)

    def test_public_build_plan_interface_is_uniform_in_p0(self):
        metadata = VideoMetadata(fps=2, frame_count=4)
        self.assertEqual(
            build_plan(metadata, 100, 42, video_id="v", max_frames=2),
            uniform_plan(metadata, 100, 42, video_id="v", max_frames=2),
        )


if __name__ == "__main__":
    unittest.main()

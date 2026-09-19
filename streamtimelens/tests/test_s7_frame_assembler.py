import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, VideoMeta
from streamtimelens.refiner.frame_assembler import assemble_refinement_frames
from streamtimelens.retrieval.candidates import RetrievalCandidate


def jpeg(color):
    output = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(output, format="JPEG")
    return output.getvalue()


def candidate(card_id="c"):
    return RetrievalCandidate("candidate", 0, 4, 1, (card_id,), 1, {})


class FrameAssemblerTest(unittest.TestCase):
    def _snapshot(self, root, *, with_raw=True):
        refs = ["000000000.jpg", "000000004.jpg", "000000008.jpg"] if with_raw else []
        card = {
            "id": "c", "t_start": 0, "t_end": 4, "raw_ref_ids": refs,
            "boundary_cache": {
                "left_frame_ids": refs[:1], "right_frame_ids": refs[-1:],
                "internal_frame_ids": refs[1:2],
            },
        }
        frames = [(ref, jpeg((index * 50, 0, 0))) for index, ref in enumerate(refs)]
        metadata = {
            ref: {"timestamp_s": index * 2.0, "frame_index": index * 4, "blob": ref,
                  "novelty": index / 10}
            for index, ref in enumerate(refs)
        }
        SnapshotWriter(root).write(
            name="s", t_q=5, meta=VideoMeta("v", 10, 2, 20),
            budget=Budget(100_000, 1), cards=[card], raw_frames=frames,
            raw_metadata=metadata, writer_calls=0, config={},
        )
        return SnapshotReader(root / "s")

    def test_boundaries_are_retained_and_time_sorted(self):
        with tempfile.TemporaryDirectory() as temporary:
            assembly = assemble_refinement_frames(
                self._snapshot(Path(temporary)), candidate(), max_frames=2,
            )
        self.assertEqual(assembly.status, "ok")
        self.assertEqual(assembly.frame_refs, ("000000000.jpg", "000000008.jpg"))
        self.assertEqual(assembly.prepared.unique_indices, (0, 8))
        self.assertEqual(assembly.prepared.video.shape[0], 4)
        self.assertTrue(assembly.diagnostics["left_boundary_retained"])
        self.assertTrue(assembly.diagnostics["right_boundary_retained"])

    def test_no_visual_evidence_is_typed_and_bad_cap_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            reader = self._snapshot(Path(temporary), with_raw=False)
            assembly = assemble_refinement_frames(reader, candidate(), max_frames=2)
            self.assertEqual(assembly.status, "NO_VISUAL_EVIDENCE")
            self.assertIsNone(assembly.prepared)
            with self.assertRaisesRegex(ValueError, "cap"):
                assemble_refinement_frames(reader, candidate(), max_frames=0)


if __name__ == "__main__":
    unittest.main()

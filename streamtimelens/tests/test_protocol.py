import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

from streamtimelens.protocol.arrival import build_arrival_plan, verify_arrival_plan, write_arrival_plan
from streamtimelens.protocol.audit import SnapshotAccessError, audit_query_reads
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter, directory_bytes
from streamtimelens.protocol.types import Budget, FramePacket, VideoMeta
from streamtimelens.retrieval.ranker import locate_lexically
from streamtimelens.stream.decoder import decord_packets, jsonl_packets
from streamtimelens.stream.runner import StreamingIngestor


FIXTURE = Path(__file__).parent / "fixtures" / "frames.jsonl"


class ArrivalPlanTest(unittest.TestCase):
    def test_natural_and_fixed_cohorts_are_frozen(self):
        rows = build_arrival_plan([{"video_id": "v", "query_id": "q", "query": "open door",
                                    "gt_span": [1, 2], "duration": 10}], [.25, .5])
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[0].eligible)
        self.assertTrue(rows[-1].eligible)
        with tempfile.TemporaryDirectory() as temporary:
            plan = Path(temporary) / "arrival.jsonl"
            digest = write_arrival_plan(rows, plan)
            self.assertEqual(len(digest), 64)
            self.assertEqual(len(plan.read_text().splitlines()), 3)
            self.assertEqual(verify_arrival_plan(plan), rows)
            with self.assertRaises(FileExistsError):
                write_arrival_plan(rows, plan)

    def test_event_end_delay_is_clamped_and_tampering_is_detected(self):
        rows = build_arrival_plan([{"video_id": "v", "query_id": "q", "query": "open door",
                                    "gt_span": [7, 9], "duration": 10}], [.25], [5, 15])
        delayed = [row for row in rows if row.arrival_kind == "delta"]
        self.assertEqual([row.t_q for row in delayed], [10.0, 10.0])
        self.assertTrue(all(row.eligible for row in delayed))
        with tempfile.TemporaryDirectory() as temporary:
            plan = Path(temporary) / "arrival.jsonl"
            write_arrival_plan(rows, plan)
            plan.write_text(plan.read_text() + "{}\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_arrival_plan(plan)

    def test_non_finite_arrival_inputs_are_rejected(self):
        example = {"video_id": "v", "query_id": "q", "query": "open door",
                   "gt_span": [1, 2], "duration": 10}
        with self.assertRaises(ValueError):
            build_arrival_plan([example], [math.nan])
        with self.assertRaises(ValueError):
            build_arrival_plan([example], [.5], [math.inf])


class SnapshotProtocolTest(unittest.TestCase):
    def test_writer_rejects_root_escape_query_leaks_and_non_finite_state(self):
        meta = VideoMeta("v", 4, 1, 4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshots"
            writer = SnapshotWriter(root)
            common = dict(t_q=1, meta=meta, budget=Budget(8192, 1), cards=[],
                          raw_frames=[], writer_calls=0, config={})
            with self.assertRaisesRegex(ValueError, "relative path"):
                writer.write(name="../escaped", **common)
            self.assertFalse((Path(temporary) / "escaped").exists())
            with self.assertRaisesRegex(ValueError, "query"):
                writer.write(name="query-leak", **{**common, "config": {"query": "secret"}})
            with self.assertRaisesRegex(ValueError, "video_path"):
                writer.write(
                    name="metadata-leak", raw_metadata={"frame.jpg": {"video_path": "/secret.mp4"}},
                    **common,
                )
            with self.assertRaisesRegex(ValueError, "finite"):
                writer.write(name="nan", **{**common, "t_q": math.nan})

    def test_runner_rejects_non_finite_snapshot_time(self):
        ingestor = StreamingIngestor(VideoMeta("v", 4, 1, 4), Budget(8192, 1))
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "outside"):
            ingestor.run(
                [FramePacket(0, 0, b"x", 1, 1)], [math.nan], SnapshotWriter(Path(temporary)),
                config={}, snapshot_prefix="v",
            )

    def test_single_decode_multiple_bounded_snapshots_and_no_path_escape(self):
        meta, packets = jsonl_packets(FIXTURE)
        meta = VideoMeta("demo", 12, 1, 13)
        budget = Budget(memory_bytes=8192, writer_calls_per_minute=60)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "snapshots"
            ingestor = StreamingIngestor(meta, budget)
            manifests = ingestor.run(packets, [6, 12], SnapshotWriter(output), config={"test": True}, snapshot_prefix="demo")
            self.assertEqual(set(manifests), {6.0, 12.0})
            self.assertEqual(sum(row["kind"] == "frame_seen" for row in ingestor.trace), 7)
            snapshot = output / "demo" / "rho_0.50"
            reader = SnapshotReader(snapshot)
            self.assertLessEqual(directory_bytes(snapshot), budget.memory_bytes)
            self.assertGreaterEqual(reader.manifest.state_bytes, 1)
            with self.assertRaises(SnapshotAccessError):
                reader.path("../original.mp4")
            with self.assertRaises(SnapshotAccessError):
                reader.path("ingest.trace.jsonl")
            self.assertTrue(reader.read_cards())
            metadata = json.loads(reader.path(reader.manifest.raw_metadata).read_text())
            self.assertIn("000000000.jpg", metadata)
            self.assertIn("000000006.jpg", metadata)  # current now_ring is promoted at snapshot time
            self.assertIn("working_set", ingestor.ledger.components)

    def test_snapshot_excludes_frames_after_query_arrival(self):
        meta = VideoMeta("v", 4, 1, 3)
        frames = [FramePacket(timestamp, index, bytes([index]), 1, 1) for index, timestamp in enumerate((0.0, 2.0, 4.0))]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            StreamingIngestor(meta, Budget(8192, 60)).run(
                frames, [1.0], SnapshotWriter(root), config={}, snapshot_prefix="v"
            )
            reader = SnapshotReader(root / "v" / "rho_0.25")
            metadata = json.loads(reader.path(reader.manifest.raw_metadata).read_text())
            self.assertTrue(all(item["timestamp_s"] <= 1.0 for item in metadata.values()))

    def test_oversized_frame_is_evicted_without_leaving_unaccounted_state(self):
        ingestor = StreamingIngestor(VideoMeta("v", 1, 1, 1), Budget(128, 1))
        ingestor.observe(FramePacket(0, 0, b"x" * 256, 1, 1))
        self.assertEqual(ingestor.raw.byte_size, 0)
        self.assertLessEqual(ingestor.ledger.logical_bytes, ingestor.ledger.limit_bytes)

    def test_reader_rejects_tampered_oversized_snapshot(self):
        meta = VideoMeta("v", 1, 1, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            SnapshotWriter(root).write(name="s", t_q=1, meta=meta, budget=Budget(4096, 1), cards=[], raw_frames=[], writer_calls=0, config={})
            (root / "s" / "unlisted.bin").write_bytes(b"x" * 5000)
            with self.assertRaises(ValueError):
                SnapshotReader(root / "s")

    def test_reader_rejects_content_tampering_and_unlisted_small_file(self):
        meta = VideoMeta("v", 1, 1, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            SnapshotWriter(root).write(name="s", t_q=1, meta=meta, budget=Budget(4096, 1), cards=[],
                                       raw_frames=[("000000000.jpg", b"original")], writer_calls=0, config={})
            (root / "s" / "frames" / "000000000.jpg").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "integrity"):
                SnapshotReader(root / "s")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            SnapshotWriter(root).write(name="s", t_q=1, meta=meta, budget=Budget(4096, 1), cards=[], raw_frames=[], writer_calls=0, config={})
            (root / "s" / "unlisted.txt").write_text("x")
            with self.assertRaisesRegex(ValueError, "unlisted"):
                SnapshotReader(root / "s")

    def test_reader_rejects_same_size_manifest_tampering(self):
        meta = VideoMeta("v", 10, 1, 10)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            SnapshotWriter(root).write(name="s", t_q=1.0, meta=meta, budget=Budget(4096, 1), cards=[],
                                       raw_frames=[], writer_calls=0, config={})
            manifest = root / "s" / "manifest.json"
            manifest.write_text(manifest.read_text().replace('"t_q": 1.0', '"t_q": 9.0'))
            with self.assertRaisesRegex(ValueError, "manifest integrity"):
                SnapshotReader(root / "s")

    def test_reader_rejects_symlinked_payload(self):
        meta = VideoMeta("v", 1, 1, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            SnapshotWriter(root).write(name="s", t_q=1, meta=meta, budget=Budget(4096, 1), cards=[], raw_frames=[], writer_calls=0, config={})
            os.symlink("/etc/hosts", root / "s" / "frames" / "escaped.jpg")
            with self.assertRaisesRegex(ValueError, "symlink"):
                SnapshotReader(root / "s")

    def test_manifest_reports_exact_query_visible_bytes(self):
        meta = VideoMeta("v", 1, 1, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = SnapshotWriter(root).write(
                name="s", t_q=1, meta=meta, budget=Budget(4096, 1), cards=[],
                raw_frames=[("000000000.jpg", b"jpeg")], writer_calls=0, config={},
            )
            self.assertEqual(manifest.state_bytes, directory_bytes(root / "s"))

    def test_decoder_source_is_single_pass_and_auditable(self):
        _, packets = jsonl_packets(FIXTURE)
        decoded = list(packets)
        self.assertEqual(packets.decode_count, len(decoded))
        self.assertEqual(packets.last_frame_index, decoded[-1].frame_index)
        self.assertEqual(packets.seek_count, 0)
        with self.assertRaisesRegex(RuntimeError, "second decode"):
            list(packets)

    def test_jsonl_vfr_timestamps_remain_monotonic(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "vfr.jsonl"
            source.write_text("\n".join(json.dumps({"video_id": "v", "fps": 30, "timestamp_s": timestamp,
                                                       "frame_index": index, "image_b64": ""})
                                        for index, timestamp in enumerate((0.0, .02, .09))) + "\n")
            _, packets = jsonl_packets(source)
            decoded = list(packets)
            self.assertEqual([packet.timestamp_s for packet in decoded], [0.0, .02, .09])
            self.assertEqual(packets.timestamp_mode, "jsonl_timestamp")

    def test_decord_backend_uses_sequential_indices_and_vfr_pts(self):
        import numpy

        class FakeFrame:
            def asnumpy(self):
                return numpy.zeros((2, 3, 3), dtype=numpy.uint8)

        class FakeReader:
            timestamps = ((0.0, .1), (.1, .3), (.3, .35), (.35, .8), (.8, 1.0))
            def __init__(self, *_args, **_kwargs):
                self.cursor = 0
            def get_avg_fps(self):
                return 4
            def __len__(self):
                return 5
            def get_frame_timestamp(self, index):
                return self.timestamps[index]
            def next(self):
                self.cursor += 1
                return FakeFrame()

        fake_decord = types.SimpleNamespace(VideoReader=FakeReader, cpu=lambda _: object())
        original = sys.modules.get("decord")
        sys.modules["decord"] = fake_decord
        try:
            meta, packets = decord_packets(Path("fixture.mp4"), sample_fps=2)
            decoded = list(packets)
        finally:
            if original is None:
                del sys.modules["decord"]
            else:
                sys.modules["decord"] = original
        self.assertEqual(meta.original_fps, 4)
        self.assertEqual([packet.frame_index for packet in decoded], [0, 2, 4])
        self.assertEqual([packet.timestamp_s for packet in decoded], [0.0, .3, .8])
        self.assertEqual(packets.timestamp_mode, "decord_pts")
        self.assertEqual(packets.decode_count, 5)
        self.assertEqual(packets.emitted_packet_count, 3)

    def test_query_read_audit_only_allows_snapshot_manifest_files(self):
        meta = VideoMeta("v", 4, 1, 4)
        cards = [{"id": "card", "t_start": 1, "t_end": 2, "summary": "opens door"}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            SnapshotWriter(root).write(name="s", t_q=4, meta=meta, budget=Budget(4096, 1), cards=cards,
                                       raw_frames=[], writer_calls=0, config={})
            reader = SnapshotReader(root / "s")
            with audit_query_reads(reader.root, set(reader.manifest.allowed_files)) as audit:
                prediction = locate_lexically("opens door", reader)
                with self.assertRaises(SnapshotAccessError):
                    Path(temporary, "original-video.mp4").read_bytes()
            self.assertEqual(prediction.status, "ok")
            self.assertTrue(audit.reads)
            self.assertTrue(set(audit.reads).issubset(audit.allowed_paths))

    def test_snapshot_rejects_video_trace_prediction_and_cleans_failed_temp_dir(self):
        meta = VideoMeta("v", 1, 1, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = SnapshotWriter(root)
            with self.assertRaisesRegex(ValueError, "video_path"):
                writer.write(name="s", t_q=1, meta=meta, budget=Budget(4096, 1), cards=[], raw_frames=[],
                             writer_calls=0, config={"video_path": "/data/video.mp4"})
            with self.assertRaisesRegex(ValueError, "trace"):
                writer.write(name="s", t_q=1, meta=meta, budget=Budget(4096, 1), cards=[], raw_frames=[("trace.txt", b"x")],
                             writer_calls=0, config={})
            self.assertFalse((root / "s").exists())
            self.assertFalse(any(path.name.startswith(".s.") for path in root.iterdir()))

    def test_query_cli_runs_after_original_video_is_removed(self):
        meta = VideoMeta("v", 4, 1, 4)
        cards = [{"id": "card", "t_start": 1, "t_end": 2, "summary": "opens door"}]
        package_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original.mp4"
            original.write_bytes(b"not mounted for query")
            SnapshotWriter(root).write(name="source_snapshot", t_q=4, meta=meta, budget=Budget(4096, 1), cards=cards,
                                       raw_frames=[], writer_calls=0, config={})
            isolated = root / "isolated"
            shutil.copytree(root / "source_snapshot", isolated / "snapshot")
            original.unlink()
            completed = subprocess.run(
                [sys.executable, str(package_root / "scripts" / "answer_snapshots.py"), "--snapshot", "snapshot", "--query", "opens door"],
                cwd=isolated, text=True, capture_output=True, check=True,
            )
            self.assertEqual(json.loads(completed.stdout)["status"], "ok")

    def test_pixel_only_auto_query_does_not_load_clip(self):
        package_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            SnapshotWriter(root).write(
                name="pixel", t_q=4, meta=VideoMeta("v", 4, 1, 4), budget=Budget(4096, 1),
                cards=[], raw_frames=[], writer_calls=0, config={}, method="uniform_raw", pixel_only=True,
            )
            automatic = subprocess.run(
                [sys.executable, str(package_root / "scripts" / "answer_snapshots.py"),
                 "--snapshot", str(root / "pixel"), "--query", "opens door"],
                text=True, capture_output=True,
            )
            self.assertEqual(automatic.returncode, 0, automatic.stderr)
            self.assertEqual(json.loads(automatic.stdout)["status"], "NOT_FOUND")
            explicit = subprocess.run(
                [sys.executable, str(package_root / "scripts" / "answer_snapshots.py"),
                 "--snapshot", str(root / "pixel"), "--query", "opens door", "--retrieval", "frame"],
                text=True, capture_output=True,
            )
            self.assertNotEqual(explicit.returncode, 0)
            self.assertIn("pixel-only", explicit.stderr)

    def test_result_entrypoints_reject_unversioned_external_outputs(self):
        package_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary)
            build = subprocess.run(
                [sys.executable, str(package_root / "scripts" / "build_snapshots.py"),
                 "--frames", str(FIXTURE), "--video-id", "v", "--duration", "12",
                 "--output", str(outside / "snapshots")],
                text=True, capture_output=True,
            )
            self.assertNotEqual(build.returncode, 0)
            self.assertIn("result path must be below", build.stderr)


if __name__ == "__main__":
    unittest.main()

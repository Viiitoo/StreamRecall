import inspect
from io import BytesIO

import numpy as np

from streamtimelens.memory.hierarchical_event import HierarchicalEventMemory, load_event_memory
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, FramePacket, VideoMeta
from streamtimelens.retrieval.event_candidates import (
    EVENT_CANDIDATE_MARGIN_S,
    EVENT_EXPAND_NEIGHBORS,
    EVENT_MERGE_GAP_S,
    EVENT_TOP_K,
    build_event_candidates,
    locate_event_memory,
)
from streamtimelens.stream.decoder import SinglePassPackets
from streamtimelens.stream.hierarchical_event_runner import (
    HierarchicalEventIngestor,
    hem_ingest_config,
)


class FakeEncoder:
    batch_size = 2

    def due(self, timestamp_s):
        return True

    def encode_images(self, images):
        return np.asarray([
            (1.0, 0.0) if int(np.asarray(image)[0, 0, 0]) < 100 else (0.0, 1.0)
            for image in images
        ], dtype=np.float32)


def _jpeg(value):
    from PIL import Image

    output = BytesIO()
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(
        output, format="PNG",
    )
    return output.getvalue()


def _packets():
    holder = {}

    def rows():
        for index in range(5):
            holder["stream"].record_source_decode()
            yield FramePacket(
                float(index), index, _jpeg(0 if index < 3 else 255), 4, 4,
                "synthetic", "video",
            )

    stream = SinglePassPackets(rows)
    holder["stream"] = stream
    return stream


def _events():
    memory = HierarchicalEventMemory()
    for index, vector in enumerate(((1, 0), (1, 0), (0, 1), (0, 1), (1, 0))):
        memory.observe(timestamp_s=float(index * 3), frame_index=index, embedding=vector)
    return memory.events()


def test_event_reader_contract_is_fixed_and_deterministic():
    assert EVENT_TOP_K == 8
    assert EVENT_MERGE_GAP_S == 4.0
    assert EVENT_EXPAND_NEIGHBORS == 1
    assert EVENT_CANDIDATE_MARGIN_S == 0.0
    events = _events()
    left = build_event_candidates((1.0, 0.0), events, upper_bound_s=12.0)
    right = build_event_candidates((1.0, 0.0), tuple(reversed(events)), upper_bound_s=12.0)
    assert left == right
    assert left
    assert left[0].start_s == 0.0
    assert left[0].end_s <= 12.0


def test_ingestor_is_query_blind_single_pass_and_writes_arrival_state(tmp_path):
    meta = VideoMeta("video", 4.0, 1.0, 5)
    budget = Budget(1024 * 1024, 0)
    source = _packets()
    ingestor = HierarchicalEventIngestor(
        meta, budget, clip_encoder=FakeEncoder(), source_revision="a" * 40,
    )
    parameters = set(inspect.signature(HierarchicalEventIngestor.observe).parameters)
    assert parameters == {"self", "packet"}
    writer = SnapshotWriter(tmp_path, source_revision="a" * 40)
    manifests = ingestor.run(
        source, (2.0, 4.0), writer, config=hem_ingest_config(), snapshot_prefix="snapshots",
    )
    assert source.decode_count == source.emitted_packet_count == 5
    assert source.seek_count == 0
    assert set(manifests) == {2.0, 4.0}
    early = SnapshotReader(tmp_path / "snapshots/rho_0.50")
    final = SnapshotReader(tmp_path / "snapshots/rho_1.00")
    assert sum(row.support_count for row in load_event_memory(early)) == 3
    assert sum(row.support_count for row in load_event_memory(final)) == 5
    assert max(row.end_s for row in load_event_memory(early)) <= 2.0
    assert max(row.end_s for row in load_event_memory(final)) <= 4.0


def test_query_reader_uses_only_event_snapshot_and_does_not_mutate_it(tmp_path):
    meta = VideoMeta("video", 4.0, 1.0, 5)
    ingestor = HierarchicalEventIngestor(
        meta, Budget(1024 * 1024, 0), clip_encoder=FakeEncoder(),
        source_revision="b" * 40,
    )
    ingestor.run(
        _packets(), (4.0,), SnapshotWriter(tmp_path, source_revision="b" * 40),
        config=hem_ingest_config(),
    )
    reader = SnapshotReader(tmp_path / "snapshot/rho_1.00")
    before = {
        name: reader.path(name).read_bytes() for name in reader.manifest.allowed_files
    }
    prediction, debug = locate_event_memory((1.0, 0.0), reader)
    after_reader = SnapshotReader(reader.root)
    after = {
        name: after_reader.path(name).read_bytes()
        for name in after_reader.manifest.allowed_files
    }
    assert prediction.start_s is not None and prediction.end_s is not None
    assert debug["candidate_count"] >= 1
    assert before == after
    assert not reader.frame_paths()

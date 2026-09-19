import inspect
import json
import math

import numpy as np
import pytest

from streamtimelens.memory.hierarchical_event import (
    BOUNDARY_COSINE,
    EMBEDDING_PRECISION,
    HEM_SCHEMA_VERSION,
    LEVEL_CAPACITIES,
    MAX_FINE_EVENT_DURATION_S,
    HierarchicalEvent,
    HierarchicalEventMemory,
    event_memory_sha256,
    load_event_memory,
    write_event_snapshot,
)
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter, directory_bytes
from streamtimelens.protocol.types import Budget, VideoMeta


def _vector(index):
    angle = 0.0 if index % 2 == 0 else math.pi / 2
    return np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float32)


def _memory(count=200):
    memory = HierarchicalEventMemory()
    for index in range(count):
        memory.observe(timestamp_s=float(index), frame_index=index, embedding=_vector(index))
    return memory


def test_contract_is_fixed_and_observe_api_is_query_blind():
    assert HEM_SCHEMA_VERSION == "hem_event_memory_v1"
    assert LEVEL_CAPACITIES == (64, 32, 16)
    assert BOUNDARY_COSINE == 0.85
    assert MAX_FINE_EVENT_DURATION_S == 8.0
    assert EMBEDDING_PRECISION == "fp16"
    parameters = set(inspect.signature(HierarchicalEventMemory.observe).parameters)
    assert parameters == {"self", "timestamp_s", "frame_index", "embedding"}
    assert not parameters & {"query", "query_id", "gt", "gt_span"}
    with pytest.raises(ValueError, match="frozen"):
        HierarchicalEventMemory(capacities=(32, 16, 8))


def test_similar_tokens_form_bounded_fine_event_and_visual_change_splits():
    memory = HierarchicalEventMemory()
    for index in range(5):
        reason = memory.observe(
            timestamp_s=float(index), frame_index=index, embedding=(1.0, 0.0),
        )
    assert reason == "merge_similar"
    assert len(memory.events()) == 1
    assert memory.events()[0].support_count == 5
    reason = memory.observe(timestamp_s=5.0, frame_index=5, embedding=(0.0, 1.0))
    assert reason == "boundary_visual_change"
    assert len(memory.events()) == 2
    assert sum(row.support_count for row in memory.events()) == 6


def test_hierarchy_conserves_support_and_keeps_recent_state_fine():
    memory = _memory()
    events = memory.events()
    assert sum(row.support_count for row in events) == memory.seen_count == 200
    assert all(count <= cap for count, cap in zip(memory.level_counts, LEVEL_CAPACITIES))
    assert events[0].scale > 0
    assert events[-1].scale == 0
    assert [row.first_frame_index for row in events] == sorted(
        row.first_frame_index for row in events
    )
    assert all(left.last_frame_index < right.first_frame_index for left, right in zip(events, events[1:]))


def test_hierarchy_is_deterministic_and_records_round_trip():
    left, right = _memory(), _memory()
    assert left.records() == right.records()
    assert event_memory_sha256(left.events()) == event_memory_sha256(right.events())
    restored = tuple(HierarchicalEvent.from_record(row) for row in left.records())
    assert [row.event_id for row in restored] == [row.event_id for row in left.events()]
    assert [row.support_count for row in restored] == [row.support_count for row in left.events()]


def test_event_snapshot_is_self_contained_byte_accounted_and_read_only(tmp_path):
    memory = _memory(100)
    writer = SnapshotWriter(tmp_path, source_revision="hem-test-revision")
    manifest = write_event_snapshot(
        memory, writer, name="snapshot", t_q=100.0,
        meta=VideoMeta("synthetic", 100.0, 1.0, 100),
        budget=Budget(1024 * 1024, 0),
        config={"method": "HEM-01", "query_independent": True},
        source_revision="hem-test-revision",
    )
    reader = SnapshotReader(tmp_path / "snapshot")
    before = tuple(
        (name, reader.path(name).read_bytes()) for name in sorted(reader.manifest.allowed_files)
    )
    events = load_event_memory(reader)
    after_reader = SnapshotReader(tmp_path / "snapshot")
    after = tuple(
        (name, after_reader.path(name).read_bytes())
        for name in sorted(after_reader.manifest.allowed_files)
    )
    assert manifest.event_memory == "event_memory.json"
    assert manifest.method == "hierarchical_event_memory"
    assert manifest.state_bytes == directory_bytes(tmp_path / "snapshot")
    assert manifest.state_bytes <= manifest.budget_bytes
    assert not reader.frame_paths()
    assert sum(row.support_count for row in events) == 100
    assert before == after


def test_snapshot_rejects_query_bearing_config_and_future_state(tmp_path):
    memory = _memory(3)
    writer = SnapshotWriter(tmp_path, source_revision="hem-test-revision")
    with pytest.raises(ValueError, match="query snapshot"):
        write_event_snapshot(
            memory, writer, name="leak", t_q=3.0,
            meta=VideoMeta("synthetic", 3.0, 1.0, 3),
            budget=Budget(1024 * 1024, 0), config={"query": "forbidden"},
            source_revision="hem-test-revision",
        )
    with pytest.raises(ValueError, match="excludes observed"):
        write_event_snapshot(
            memory, writer, name="future", t_q=1.0,
            meta=VideoMeta("synthetic", 3.0, 1.0, 3),
            budget=Budget(1024 * 1024, 0), config={"method": "HEM-01"},
            source_revision="hem-test-revision",
        )


def test_default_snapshot_manifest_does_not_gain_an_event_memory_field(tmp_path):
    writer = SnapshotWriter(tmp_path, source_revision="baseline-revision")
    writer.write(
        name="baseline", t_q=1.0, meta=VideoMeta("v", 1.0, 1.0, 1),
        budget=Budget(1024 * 1024, 0), cards=[], raw_frames=[], raw_metadata={},
        writer_calls=0, config={"method": "baseline"}, method="uniform_raw",
    )
    payload = json.loads((tmp_path / "baseline" / "manifest.json").read_text())
    assert "event_memory" not in payload

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]


def _load_script(name):
    path = PROJECT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


preregister = _load_script("preregister_jq01_data")
run_oof = _load_script("run_jq01_oof")


def _row(prefix, index, *, source="ActivityNet", bucket="short"):
    row = {
        "video_id": f"{prefix}-video-{index}",
        "group_id": f"{prefix}-group-{index}",
        "query_id": f"{prefix}-query-{index}",
        "source": source,
        "duration_bucket": bucket,
        "content_sha256": f"{index + (0 if prefix == 'dev' else 100):064x}",
    }
    if prefix == "dev":
        row["outer_fold"] = index % 4
    return row


def _exclusions():
    return {"video_ids": [], "group_ids": [], "query_ids": [], "content_sha256": []}


def test_old_development_is_kept_reusable_and_d_lock_is_separate():
    development = [_row("dev", index) for index in range(4)]
    d_lock = [_row("lock", index) for index in range(8)]
    selected, audit = preregister.build_data_registry(
        development, d_lock, _exclusions(), folds=4,
        minimum_d_lock_videos=8, seed=20260911,
    )
    dev_rows = [row for row in selected if row["data_role"] == "development/consumed"]
    lock_rows = [row for row in selected if row["data_role"] == "d_lock"]
    assert len(dev_rows) == 4
    assert len(lock_rows) == 8
    assert {row["outer_fold"] for row in dev_rows} == {0, 1, 2, 3}
    assert {row["outer_fold"] for row in lock_rows} == {None}
    assert audit["development_annotation_status"] == "consumed_allowed_for_repeated_development"
    assert not audit["d_lock_annotation_opened"]
    repeated, _ = preregister.build_data_registry(
        list(reversed(development)), list(reversed(d_lock)), _exclusions(), folds=4,
        minimum_d_lock_videos=8, seed=20260911,
    )
    first_folds = {row["group_id"]: row["outer_fold"] for row in selected if row["outer_fold"] is not None}
    repeated_folds = {row["group_id"]: row["outer_fold"] for row in repeated if row["outer_fold"] is not None}
    assert first_folds == repeated_folds


def test_formal_or_content_overlap_is_rejected_and_d_lock_floor_is_enforced():
    development = [_row("dev", index) for index in range(4)]
    d_lock = [_row("lock", index) for index in range(8)]
    exclusions = _exclusions()
    exclusions["video_ids"] = [development[0]["video_id"]]
    with pytest.raises(ValueError, match="entered JQ-01 development"):
        preregister.build_data_registry(
            development, d_lock, exclusions, folds=4, minimum_d_lock_videos=8, seed=1,
        )
    overlapping = [dict(row) for row in d_lock]
    overlapping[0]["content_sha256"] = development[0]["content_sha256"]
    with pytest.raises(ValueError, match="overlaps on content_sha256"):
        preregister.build_data_registry(
            development, overlapping, _exclusions(), folds=4,
            minimum_d_lock_videos=8, seed=1,
        )
    with pytest.raises(RuntimeError, match="requires 8"):
        preregister.build_data_registry(
            development, d_lock[:7], _exclusions(), folds=4,
            minimum_d_lock_videos=8, seed=1,
        )


def test_experiment_registry_extends_without_changing_prior_records(tmp_path):
    summary = {"passed": False, "decision": "iterate", "gate_checks": {"ci": False}}
    first = run_oof.build_experiment_registry(
        [], revision_id="jq-v1-r1", hypothesis="first attempt", code_revision="a" * 40,
        config_sha256="b" * 64, observations_sha256="c" * 64,
        j0_selection_sha256="d" * 64, result_path="/results/r1", summary=summary,
    )
    path = tmp_path / "experiment_registry.json"
    path.write_text(json.dumps(first), encoding="utf-8")
    parent, digest = run_oof._load_parent_registry(path)
    assert digest and parent == first["records"]
    second = run_oof.build_experiment_registry(
        parent, revision_id="jq-v1-r2", hypothesis="second attempt",
        code_revision="e" * 40, config_sha256="f" * 64,
        observations_sha256="c" * 64, j0_selection_sha256="d" * 64,
        result_path="/results/r2", summary={
            "passed": True, "decision": "candidate", "gate_checks": {"ci": True},
        },
    )
    assert second["records"][0] == first["records"][0]
    assert [row["revision_id"] for row in second["records"]] == ["jq-v1-r1", "jq-v1-r2"]
    with pytest.raises(ValueError, match="already exists"):
        run_oof.build_experiment_registry(
            parent, revision_id="jq-v1-r1", hypothesis="duplicate", code_revision="e" * 40,
            config_sha256="f" * 64, observations_sha256="c" * 64,
            j0_selection_sha256="d" * 64, result_path="/results/duplicate", summary=summary,
        )
    with pytest.raises(ValueError, match="changed the fixed development"):
        run_oof.build_experiment_registry(
            parent, revision_id="jq-v1-r3", hypothesis="changed samples",
            code_revision="e" * 40, config_sha256="f" * 64,
            observations_sha256="0" * 64, j0_selection_sha256="d" * 64,
            result_path="/results/r3", summary=summary,
        )


def test_j2_accepts_only_complete_fixed_development_role(tmp_path):
    selection = [
        {
            **_row("dev", index), "data_role": "development/consumed",
        }
        for index in range(2)
    ]
    path = tmp_path / "selection.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in selection), encoding="utf-8",
    )
    observations = [
        SimpleNamespace(
            observation_id=f"obs-{index}", video_id=row["video_id"],
            group_id=row["group_id"], query_id=row["query_id"],
            content_sha256=row["content_sha256"],
        )
        for index, row in enumerate(selection)
    ]
    folds = {row["group_id"]: row["outer_fold"] for row in selection}
    run_oof._validate_j0_rows(path, observations, folds)
    with pytest.raises(ValueError, match="complete fixed development"):
        run_oof._validate_j0_rows(path, observations[:1], folds)
    observations[0].content_sha256 = "f" * 64
    with pytest.raises(ValueError, match="not admitted"):
        run_oof._validate_j0_rows(path, observations, folds)

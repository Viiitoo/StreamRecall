import importlib.util
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def _load_mechanism_script():
    path = PROJECT / "scripts" / "run_jq01_mechanism_gate.py"
    spec = importlib.util.spec_from_file_location("run_jq01_mechanism_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_synthetic_mechanism_audit_persists_all_nine_candidates_without_mutation():
    audit = _load_mechanism_script()._synthetic_candidate_audit()
    assert not audit["effect_data_read"]
    assert audit["candidate_count"] == 9
    assert len(set(audit["candidate_ids"])) == 9
    assert audit["snapshot_unchanged"]
    assert audit["snapshot_state_bytes"] <= audit["snapshot_budget_bytes"]
    assert audit["baseline_bit_exact_if_selected"]
    assert sorted(row["rank"] for row in audit["debug"]["candidates"]) == list(range(1, 10))
    assert sum(row["selected"] for row in audit["debug"]["candidates"]) == 1

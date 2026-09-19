"""Installed command entry points; implementation remains in standalone scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _script_main(name: str) -> int:
    script = Path(__file__).resolve().parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"streamtimelens._{script.stem}", script)
    if spec is None or spec.loader is None:  # pragma: no cover - installation corruption
        raise RuntimeError(f"cannot load CLI script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


def build_arrival_plan_main() -> int:
    return _script_main("build_arrival_plan.py")


def build_snapshots_main() -> int:
    return _script_main("build_snapshots.py")


def answer_snapshots_main() -> int:
    return _script_main("answer_snapshots.py")


def answer_hybrid_v3_main() -> int:
    return _script_main("answer_hybrid_v3.py")


def run_hybrid_v3_matrix_main() -> int:
    return _script_main("run_hybrid_v3_matrix.py")


def evaluate_hybrid_v3_main() -> int:
    return _script_main("evaluate_hybrid_v3.py")


def answer_visual_grid_main() -> int:
    return _script_main("answer_visual_grid.py")


def run_oracle_refiner_main() -> int:
    return _script_main("run_oracle_refiner.py")


def run_writer_feasibility_main() -> int:
    return _script_main("run_writer_feasibility.py")


def generate_writer_feasibility_main() -> int:
    return _script_main("generate_writer_feasibility.py")


def run_offline_timelens_main() -> int:
    return _script_main("run_offline_timelens.py")


def evaluate_main() -> int:
    return _script_main("evaluate.py")


def run_matrix_main() -> int:
    return _script_main("run_matrix.py")


def run_engineering_smoke_main() -> int:
    return _script_main("run_engineering_smoke.py")


def audit_visual_cache_main() -> int:
    return _script_main("audit_visual_cache.py")


def select_visual_dev_main() -> int:
    return _script_main("select_visual_dev.py")


def summarize_visual_dev_main() -> int:
    return _script_main("summarize_visual_dev.py")


def prepare_formal_visual_main() -> int:
    return _script_main("prepare_formal_visual.py")


def summarize_visual_formal_main() -> int:
    return _script_main("summarize_visual_formal.py")


def select_visual_long_configs_main() -> int:
    return _script_main("select_visual_long_configs.py")


def summarize_visual_long_main() -> int:
    return _script_main("summarize_visual_long.py")


def summarize_offline_formal_main() -> int:
    return _script_main("summarize_offline_formal.py")


def run_visual_refiner_gate_main() -> int:
    return _script_main("run_visual_refiner_gate.py")


def freeze_dev_config_main() -> int:
    return _script_main("freeze_dev_config.py")


def freeze_visual_config_main() -> int:
    return _script_main("freeze_visual_config.py")


def freeze_visual_selection_main() -> int:
    return _script_main("freeze_visual_selection.py")


def render_formal_report_main() -> int:
    return _script_main("render_formal_report.py")


def run_golden_regression_main() -> int:
    return _script_main("run_golden_regression.py")


def prepare_long_video_extension_main() -> int:
    return _script_main("prepare_long_video_extension.py")


def run_snag_g0_parity_main() -> int:
    return _script_main("run_snag_g0_parity.py")


def build_snag_snapshots_main() -> int:
    return _script_main("build_snag_snapshots.py")


def prepare_snag_development_main() -> int:
    return _script_main("prepare_snag_development.py")


def train_evaluate_snag_pooled_main() -> int:
    return _script_main("train_evaluate_snag_pooled.py")


def audit_snag_formal_start_main() -> int:
    return _script_main("audit_snag_formal_start.py")


def answer_snag_snapshots_main() -> int:
    return _script_main("answer_snag_snapshots.py")


def prepare_snag_formal_main() -> int:
    return _script_main("prepare_snag_formal.py")


def evaluate_snag_formal_main() -> int:
    return _script_main("evaluate_snag_formal.py")


def audit_snag_g2_diagnostics_main() -> int:
    return _script_main("audit_snag_g2_diagnostics.py")


def audit_snag_g1_upper_bound_main() -> int:
    return _script_main("audit_snag_g1_upper_bound.py")

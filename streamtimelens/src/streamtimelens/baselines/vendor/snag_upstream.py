"""Auditable loader for the pinned upstream SnAG implementation.

The upstream checkout is loaded under a private module namespace.  This avoids
accidentally importing another ``libs`` package and, importantly, does not run
SnAG's top-level ``libs/__init__.py`` (which eagerly imports training-only
dependencies such as TensorBoard and the compiled NMS extension).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .snag_model import UPSTREAM_REVISION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"SnAG upstream path is not a Git checkout: {root}") from exc


def verify_upstream_checkout(root: Path | str) -> Path:
    resolved = Path(root).expanduser().resolve(strict=True)
    revision = _git_revision(resolved)
    if revision != UPSTREAM_REVISION:
        raise ValueError(
            f"SnAG revision mismatch: expected {UPSTREAM_REVISION}, got {revision}"
        )
    required = (
        "libs/core/opt.py",
        "libs/modeling/model.py",
        "libs/worker.py",
    )
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise ValueError(f"incomplete SnAG checkout; missing: {', '.join(missing)}")
    return resolved


def _package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module


def _load_upstream_modules(root: Path) -> tuple[Any, Any]:
    # The namespace includes the revision, so two different checkouts cannot be
    # silently mixed in one process.
    namespace = f"_streamtimelens_snag_{UPSTREAM_REVISION[:12]}"
    _package(namespace, root)
    _package(f"{namespace}.libs", root / "libs")
    _package(f"{namespace}.libs.core", root / "libs/core")
    _package(f"{namespace}.libs.modeling", root / "libs/modeling")
    opt_module = importlib.import_module(f"{namespace}.libs.core.opt")
    model_module = importlib.import_module(f"{namespace}.libs.modeling.model")
    return opt_module, model_module


def load_upstream_nms_extension(upstream_root: Path | str) -> Any:
    """Load SnAG's compiled NMS operator without importing its training stack."""
    root = verify_upstream_checkout(upstream_root)
    nms_root = root / "libs/nms"
    if str(nms_root) not in sys.path:
        sys.path.insert(0, str(nms_root))
    try:
        # Import torch first so its shared libraries are visible to the extension.
        import torch  # noqa: F401
        return importlib.import_module("nms_1d_cpu_vg")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "SnAG NMS extension is unavailable; run `python3 setup_nms.py build_ext --inplace` "
            f"under {nms_root}"
        ) from exc


@dataclass(frozen=True)
class LoadedSnAG:
    model: Any
    option: Mapping[str, Any]
    upstream_root: Path
    upstream_revision: str
    option_path: Path
    option_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_key: str

    def provenance(self) -> dict[str, object]:
        return {
            "upstream_root": str(self.upstream_root),
            "upstream_revision": self.upstream_revision,
            "option_path": str(self.option_path),
            "option_sha256": self.option_sha256,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_key": self.checkpoint_key,
        }


def load_snag_model(
    upstream_root: Path | str,
    option_path: Path | str,
    checkpoint_path: Path | str,
    *,
    device: str = "cpu",
    checkpoint_key: str = "model_ema",
) -> LoadedSnAG:
    """Construct the real pinned PtTransformer and load an official checkpoint."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - experiment dependency
        raise RuntimeError("torch is required to load SnAG") from exc

    root = verify_upstream_checkout(upstream_root)
    option_file = Path(option_path).expanduser().resolve(strict=True)
    checkpoint_file = Path(checkpoint_path).expanduser().resolve(strict=True)
    try:
        option_file.relative_to(root)
    except ValueError:
        # Experiment opt.yaml files legitimately live outside the source tree.
        pass
    opt_module, model_module = _load_upstream_modules(root)
    option = opt_module.load_opt(str(option_file), is_training=False)
    model = model_module.PtTransformer(option["model"])
    checkpoint = torch.load(str(checkpoint_file), map_location="cpu")
    if not isinstance(checkpoint, Mapping) or checkpoint_key not in checkpoint:
        raise ValueError(f"SnAG checkpoint has no {checkpoint_key!r} state")
    model.load_state_dict(checkpoint[checkpoint_key], strict=True)
    model.to(device).eval().requires_grad_(False)
    return LoadedSnAG(
        model=model,
        option=option,
        upstream_root=root,
        upstream_revision=UPSTREAM_REVISION,
        option_path=option_file,
        option_sha256=sha256_file(option_file),
        checkpoint_path=checkpoint_file,
        checkpoint_sha256=sha256_file(checkpoint_file),
        checkpoint_key=checkpoint_key,
    )


def load_precomputed_text_feature(
    path: Path | str, *, normalize: bool = False, max_length: int | None = None,
) -> np.ndarray:
    """Load the exact ``(channels, tokens)`` layout used by SnAG datasets."""
    source = Path(path).expanduser().resolve(strict=True)
    value = np.load(source, allow_pickle=False).astype(np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("SnAG text feature must be a finite [tokens,channels] array")
    value = np.ascontiguousarray(value.T)
    if max_length is not None:
        if max_length < 1:
            raise ValueError("max_length must be positive")
        value = value[:, :max_length]
    if normalize:
        norms = np.linalg.norm(value, axis=0, keepdims=True)
        value = value / np.maximum(norms, np.finfo(np.float32).eps)
    return value


def write_loader_record(path: Path | str, loaded: LoadedSnAG) -> None:
    destination = Path(path)
    destination.write_text(
        json.dumps(loaded.provenance(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

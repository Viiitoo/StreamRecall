"""Frozen TimeLens service, sparse adapter, prompts, and strict parsing."""

from .model_service import TimeLensModelService
from .prompts import OFFICIAL_CROP_VERSION, SPARSE_LOCAL_VERSION, build_grounding_prompt, video_messages
from .timelens_adapter import PreparedSparseVideo, SparseFrame, prepare_sparse_video

__all__ = [
    "OFFICIAL_CROP_VERSION", "SPARSE_LOCAL_VERSION", "PreparedSparseVideo", "SparseFrame",
    "TimeLensModelService", "build_grounding_prompt", "prepare_sparse_video", "video_messages",
]

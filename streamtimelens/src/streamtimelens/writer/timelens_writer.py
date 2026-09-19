"""TimeLens evidence-writer adapter sharing the process-wide model service."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol, Sequence

from streamtimelens.evaluation.cost import ComponentTimer
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.protocol.types import FramePacket, VideoMeta
from streamtimelens.refiner.model_service import TimeLensModelService
from streamtimelens.refiner.timelens_adapter import SparseFrame, prepare_sparse_video

from .card_builder import build_evidence_cards
from .parse import ParseStatus, parse_writer_output
from .prompts import WRITER_MAX_NEW_TOKENS, WriterPromptSpec, build_writer_prompt, writer_video_messages


@dataclass(frozen=True)
class WriterCallResult:
    cards: tuple[EvidenceCard, ...]
    parse_status: ParseStatus
    raw_output: str
    prompt: WriterPromptSpec
    stats: dict[str, Any]
    error: str | None = None


class EvidenceWriter(Protocol):
    """A writer receives one closed segment; its API has no text-request slot."""

    def write(
        self, source_chunk_id: str, frames: Sequence[FramePacket], video_meta: VideoMeta
    ) -> WriterCallResult:
        ...


def _decode_frame(packet: FramePacket) -> Any:
    try:
        from PIL import Image
        import numpy as np

        with Image.open(BytesIO(packet.image)) as image:
            return np.asarray(image.convert("RGB").copy())
    except OSError as exc:
        raise ValueError(f"writer frame {packet.frame_index} is not a decodable image") from exc


class TimeLensEvidenceWriter:
    """Frozen greedy writer; malformed generations are repaired locally once."""

    def __init__(
        self,
        model_path: str,
        *,
        writer_revision: str,
        device_map: str = "auto",
        service: Any | None = None,
        k_frames: int = 32,
    ) -> None:
        if not model_path or not writer_revision or k_frames <= 0:
            raise ValueError("writer checkpoint, revision, and frame cap are required")
        self.model_path = str(model_path)
        self.writer_revision = writer_revision
        self.k_frames = int(k_frames)
        self.service = service or TimeLensModelService.get(model_path, device_map=device_map)

    def write(
        self, source_chunk_id: str, frames: Sequence[FramePacket], video_meta: VideoMeta
    ) -> WriterCallResult:
        ordered = sorted(frames, key=lambda frame: (frame.timestamp_s, frame.frame_index))
        if len(ordered) < 2 or ordered[0].timestamp_s >= ordered[-1].timestamp_s:
            raise ValueError("writer accepts exactly one non-empty closed segment")
        if len(ordered) > self.k_frames:
            raise ValueError("closed segment exceeds configured writer frame cap")
        prepared = prepare_sparse_video(
            [SparseFrame(frame.frame_index, frame.timestamp_s, _decode_frame(frame)) for frame in ordered],
            original_fps=video_meta.original_fps, total_num_frames=video_meta.total_num_frames,
            k_frames=self.k_frames,
        )
        segment = (prepared.timestamps_s[0], prepared.timestamps_s[-1])
        prompt = build_writer_prompt(
            segment=segment, sampled_timestamps=prepared.timestamps_s,
        )
        torch_module = getattr(self.service, "torch", None)
        with ComponentTimer("timelens_evidence_writer", torch_module=torch_module) as timer:
            raw_output = self.service.generate(
                writer_video_messages(prompt), [prepared.processor_video], WRITER_MAX_NEW_TOKENS,
            )
        model_stats = dict(getattr(self.service, "last_call_stats", {}))
        resource = timer.as_dict()
        generated_tokens = int(model_stats.get("generated_tokens", 0))
        parsed = parse_writer_output(
            raw_output, segment=segment, sampled_timestamps=prepared.timestamps_s,
        )
        hashes = dict(getattr(self.service, "hashes", {}))
        cards = build_evidence_cards(
            parsed, source_chunk_id=source_chunk_id, segment=segment, prompt=prompt,
            writer_revision=self.writer_revision,
            writer_provenance={"model_path_name": self.model_path.rsplit("/", 1)[-1], **hashes},
            generated_tokens=generated_tokens,
        )
        return WriterCallResult(
            tuple(cards), parsed.status, str(raw_output), prompt,
            {**model_stats, "resource": resource, "timestamp_audit": prepared.timestamp_audit},
            parsed.error,
        )

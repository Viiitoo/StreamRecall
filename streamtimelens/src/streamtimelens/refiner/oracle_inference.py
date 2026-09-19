"""Concrete three-condition TimeLens inference for the S2 Oracle experiment."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from streamtimelens.protocol.oracle import OracleExample, OraclePrediction

from .model_service import TimeLensModelService
from .parse import parse_refiner_output
from .prompts import OFFICIAL_CROP_VERSION, SPARSE_LOCAL_VERSION, build_grounding_prompt, video_messages
from .timelens_adapter import SparseFrame, prepare_sparse_video


FrameLoader = Callable[[OracleExample, Sequence[int]], Sequence[Any]]
PredictionCallback = Callable[[OraclePrediction], None]


class TimeLensOracleRunner:
    """Run dense-window, sparse-cache, and full-video conditions with one model."""

    def __init__(self, service: TimeLensModelService, frame_loader: FrameLoader, *,
                 dense_fps: float = 2.0, max_new_tokens: int = 128) -> None:
        if dense_fps <= 0 or max_new_tokens <= 0:
            raise ValueError("dense FPS and token limit must be positive")
        self.service = service
        self.frame_loader = frame_loader
        self.dense_fps = dense_fps
        self.max_new_tokens = max_new_tokens

    def _indices(self, example: OracleExample, span: tuple[float, float]) -> tuple[int, ...]:
        step = max(1, round(example.original_fps / self.dense_fps))
        first = max(0, round(span[0] * example.original_fps))
        last = min(example.total_num_frames - 1, round(span[1] * example.original_fps))
        indices = list(range(first, last + 1, step))
        if indices[-1] != last:
            indices.append(last)
        return tuple(indices)

    def _infer(self, example: OracleExample, mode: str, indices: Sequence[int],
               candidate: tuple[float, float]) -> OraclePrediction:
        images = self.frame_loader(example, indices)
        if len(images) != len(indices):
            raise ValueError("frame loader returned the wrong number of images")
        prepared = prepare_sparse_video(
            [SparseFrame(index, index / example.original_fps, image) for index, image in zip(indices, images)],
            original_fps=example.original_fps, total_num_frames=example.total_num_frames, k_frames=len(indices),
        )
        version = SPARSE_LOCAL_VERSION if mode == "sparse_adapter" else OFFICIAL_CROP_VERSION
        prompt = build_grounding_prompt(
            version, query=example.query,
            candidate=candidate if version == SPARSE_LOCAL_VERSION else None,
            card_summary=None,
        )
        answer = self.service.generate(video_messages(prompt), [prepared.processor_video], self.max_new_tokens)
        parsed = parse_refiner_output(answer, t_q=example.video_duration_s, candidate=candidate)
        return OraclePrediction(example.example_id, mode, parsed.span, parsed.status, answer)  # type: ignore[arg-type]

    def run(
        self,
        examples: Sequence[OracleExample],
        *,
        existing_predictions: Sequence[OraclePrediction] = (),
        prediction_callback: PredictionCallback | None = None,
    ):
        from streamtimelens.protocol.oracle import evaluate_oracle_predictions

        example_ids = {example.example_id for example in examples}
        predictions = list(existing_predictions)
        observed: dict[tuple[str, str], OraclePrediction] = {}
        for prediction in predictions:
            key = (prediction.example_id, prediction.mode)
            if prediction.example_id not in example_ids:
                raise ValueError(f"resume prediction references unknown example {prediction.example_id}")
            if key in observed:
                raise ValueError(
                    f"duplicate resume prediction for {prediction.example_id}/{prediction.mode}"
                )
            observed[key] = prediction

        def record(prediction: OraclePrediction) -> None:
            predictions.append(prediction)
            observed[(prediction.example_id, prediction.mode)] = prediction
            if prediction_callback is not None:
                prediction_callback(prediction)

        offline_cache: dict[str, tuple[tuple[float, float] | None, str, str]] = {}
        by_id = {example.example_id: example for example in examples}
        for prediction in predictions:
            if prediction.mode == "offline_timelens":
                query_id = by_id[prediction.example_id].query_id
                value = (prediction.predicted_span, prediction.status, prediction.raw_answer)
                previous = offline_cache.setdefault(query_id, value)
                if previous != value:
                    raise ValueError(f"inconsistent resumed offline predictions for query {query_id}")
        for example in examples:
            if (example.example_id, "dense_crop") not in observed:
                dense_indices = self._indices(example, example.crop_span)
                record(self._infer(example, "dense_crop", dense_indices, example.crop_span))
            if (example.example_id, "sparse_adapter") not in observed:
                record(self._infer(
                    example, "sparse_adapter", example.frame_indices, example.crop_span,
                ))
            if (example.example_id, "offline_timelens") not in observed:
                if example.query_id not in offline_cache:
                    offline = self._infer(
                        example, "offline_timelens",
                        self._indices(example, (0.0, example.video_duration_s)),
                        (0.0, example.video_duration_s),
                    )
                    offline_cache[example.query_id] = (
                        offline.predicted_span, offline.status, offline.raw_answer,
                    )
                span, status, answer = offline_cache[example.query_id]
                record(OraclePrediction(
                    example.example_id, "offline_timelens", span, status, answer,
                ))
        return predictions, evaluate_oracle_predictions(examples, predictions)

import unittest
from types import SimpleNamespace

from streamtimelens.refiner.frame_assembler import FrameAssembly
from streamtimelens.refiner.pipeline import refine_candidates
from streamtimelens.retrieval.candidates import RetrievalCandidate


def candidate(name, start, end, score):
    return RetrievalCandidate(name, start, end, score, (name,), 1, {})


class Snapshot:
    manifest = SimpleNamespace(t_q=20.0)

    def read_cards(self):
        return [
            {"id": "a", "summary": "door"}, {"id": "b", "summary": "walk"},
            {"id": "novis", "summary": "none"},
        ]


class Prepared:
    processor_video = ("pixels", {"fps": 2})


def assembler(snapshot, item, *, max_frames):
    del snapshot, max_frames
    if item.candidate_id == "novis":
        return FrameAssembly("NO_VISUAL_EVIDENCE", (), None, {"reason": "none"})
    return FrameAssembly("ok", (item.candidate_id + ".jpg",), Prepared(), {})


class Service:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.last_call_stats = {"generated_tokens": 5}

    def generate(self, messages, videos, max_new_tokens):
        del messages, videos, max_new_tokens
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return answer


class RefinerPipelineTest(unittest.TestCase):
    def test_budgeted_attempts_select_valid_and_keep_raw_resources(self):
        candidates = [
            candidate("novis", 0, 3, 1.1), candidate("a", 2, 6, 1.0),
            candidate("b", 8, 12, 0.9),
        ]
        result = refine_candidates(
            "opens door", Snapshot(), candidates,
            Service(["no timestamps", "The event happens in 9 - 11 seconds"]),
            max_calls=2, max_frames=8, assembler=assembler,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.span, (9.0, 11.0))
        self.assertEqual(result.model_calls, 2)
        self.assertEqual([attempt.status for attempt in result.attempts], [
            "NO_VISUAL_EVIDENCE", "parse_failure", "ok",
        ])
        self.assertEqual(result.attempts[-1].resource["component"], "timelens_local_refiner")
        self.assertIn("9 - 11", result.attempts[-1].raw_answer)

    def test_failures_fallback_without_claiming_success(self):
        candidates = [candidate("a", 2, 6, 1.0)]
        failed = refine_candidates(
            "opens door", Snapshot(), candidates, Service([RuntimeError("boom")]),
            max_calls=1, max_frames=8, assembler=assembler,
        )
        self.assertEqual(failed.status, "fallback")
        self.assertEqual(failed.span, (2, 6))
        self.assertEqual(failed.attempts[0].status, "model_error")
        no_budget = refine_candidates(
            "opens door", Snapshot(), candidates, None, max_calls=0, max_frames=8,
        )
        self.assertEqual(no_budget.status, "fallback")
        self.assertEqual(no_budget.model_calls, 0)


if __name__ == "__main__":
    unittest.main()

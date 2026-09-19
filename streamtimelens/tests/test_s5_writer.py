import inspect
import json
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from streamtimelens.protocol.types import FramePacket, VideoMeta
from streamtimelens.writer.card_builder import build_evidence_cards
from streamtimelens.writer.parse import parse_writer_output
from streamtimelens.writer.prompts import WRITER_MAX_NEW_TOKENS, build_writer_prompt
from streamtimelens.writer.timelens_writer import TimeLensEvidenceWriter


def event(summary="opens door", span=(1.0, 2.0), support=(1.0, 2.0), phase="complete"):
    return {
        "summary": summary, "actors": ["person"], "actions": ["open"],
        "objects": ["door"], "scene": "hallway", "span": list(span),
        "phase": phase, "visual_support": list(support),
    }


def document(events=None):
    return {"segment": [0.0, 3.0], "events": [event()] if events is None else events}


def jpeg(value):
    image = Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8))
    output = BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


class WriterSchemaParseTest(unittest.TestCase):
    def test_golden_empty_multi_and_ongoing_outputs(self):
        for events in ([], [event()], [event("walking", (0, 1), (0, 1), "ongoing"), event()]):
            result = parse_writer_output(
                json.dumps(document(events)), segment=(0, 3), sampled_timestamps=(0, 1, 2, 3),
            )
            self.assertEqual(result.status, "valid")
            self.assertEqual(len(result.document.events), len(events))

    def test_fence_wrapper_and_trailing_comma_are_repaired_without_semantic_fill(self):
        raw = "prefix ```json\n" + json.dumps(document()).replace("}]}", "},]}") + "\n``` suffix"
        result = parse_writer_output(raw, segment=(0, 3), sampled_timestamps=(0, 1, 2, 3))
        self.assertEqual(result.status, "repaired")
        missing = document()
        del missing["events"][0]["actions"]
        fallback = parse_writer_output(
            json.dumps(missing), segment=(0, 3), sampled_timestamps=(0, 1, 2, 3),
        )
        self.assertEqual(fallback.status, "fallback")
        self.assertEqual(fallback.document.events[0].summary, "unknown_event")

    def test_out_of_segment_and_unsampled_support_fall_back(self):
        for invalid in (
            document([event(span=(-0.1, 2))]),
            document([event(support=(1.25, 2))]),
        ):
            result = parse_writer_output(
                json.dumps(invalid), segment=(0, 3), sampled_timestamps=(0, 1, 2, 3),
            )
            self.assertEqual(result.status, "fallback")
            self.assertEqual(result.document.events[0].span, (0.0, 3.0))
            self.assertEqual(result.document.events[0].visual_support, [0.0, 1.0, 2.0, 3.0])

    def test_prompt_is_frozen_and_has_no_request_or_ground_truth_argument(self):
        prompt = build_writer_prompt(segment=(0, 3), sampled_timestamps=(0, 1, 2, 3))
        self.assertEqual(prompt.max_new_tokens, 256)
        self.assertEqual(prompt.temperature, 0)
        self.assertIn("may be empty", prompt.text)
        self.assertNotIn("ground truth", prompt.text.lower())
        with self.assertRaises(TypeError):
            build_writer_prompt(segment=(0, 3), sampled_timestamps=(0, 1), query="door")


class CardBuilderAndAdapterTest(unittest.TestCase):
    def test_cards_are_stable_sorted_and_store_provenance_and_uncertainty(self):
        parsed = parse_writer_output(
            json.dumps(document([event("later", (2, 3), (2, 3)), event("early", (0, 1), (0, 1))])),
            segment=(0, 3), sampled_timestamps=(0, 1, 2, 3),
        )
        prompt = build_writer_prompt(segment=(0, 3), sampled_timestamps=(0, 1, 2, 3))
        first = build_evidence_cards(
            parsed, source_chunk_id="chunk-1", segment=(0, 3), prompt=prompt,
            writer_revision="rev", generated_tokens=7,
        )
        second = build_evidence_cards(
            parsed, source_chunk_id="chunk-1", segment=(0, 3), prompt=prompt,
            writer_revision="rev", generated_tokens=7,
        )
        self.assertEqual([card.summary for card in first], ["early", "later"])
        self.assertEqual([card.id for card in first], [card.id for card in second])
        self.assertEqual(first[0].prompt_hash, prompt.template_sha256)
        self.assertEqual(first[0].generated_tokens, 7)
        serialized = first[0].serializable()
        self.assertEqual(serialized["byte_size"], len(json.dumps(
            serialized, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")))

    def test_timelens_adapter_shares_service_and_records_raw_tokens_and_time(self):
        class Service:
            hashes = {"model_sha256": "a" * 64}
            last_call_stats = {}

            def generate(inner, messages, videos, max_new_tokens):
                self.assertEqual(max_new_tokens, WRITER_MAX_NEW_TOKENS)
                self.assertEqual(len(videos), 1)
                inner.last_call_stats = {"generated_tokens": 13, "prompt_tokens": 20}
                return json.dumps({"segment": [0, 1], "events": [event(span=(0, 1), support=(0, 1))]})

        service = Service()
        writer = TimeLensEvidenceWriter("/models/TimeLens-7B", writer_revision="rev", service=service)
        frames = [
            FramePacket(0, 0, jpeg(0), 4, 4, video_id="v"),
            FramePacket(1, 1, jpeg(255), 4, 4, video_id="v"),
        ]
        result = writer.write("chunk", frames, VideoMeta("v", 2, 1, 2))
        self.assertEqual(result.parse_status, "valid")
        self.assertEqual(result.cards[0].generated_tokens, 13)
        self.assertIn("resource", result.stats)
        self.assertIn("timestamp_audit", result.stats)
        self.assertNotIn("query", inspect.signature(writer.write).parameters)


if __name__ == "__main__":
    unittest.main()

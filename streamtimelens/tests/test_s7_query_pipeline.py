import tempfile
import unittest
from pathlib import Path

import numpy as np

from streamtimelens.config import BudgetConfig, ProtocolConfig
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, VideoMeta
from streamtimelens.retrieval.embedder import CardTextEmbedder
from streamtimelens.retrieval.query_pipeline import QueryOutput, answer_card_snapshot


class Backend:
    def encode(self, texts):
        rows = []
        for text in texts:
            rows.append([1.0, 0.1] if "door" in text else [0.1, 1.0])
        return np.asarray(rows, dtype=np.float32)


class QueryPipelineTest(unittest.TestCase):
    def test_end_to_end_card_query_has_no_gt_and_explicit_fallback(self):
        embedder = CardTextEmbedder("fixture", revision="rev", backend=Backend())
        cards = [
            EvidenceCard("door", 0, 1, 3, "opens door", normalized_text="opens door"),
            EvidenceCard("walk", 0, 5, 7, "walks away", normalized_text="walks away"),
        ]
        embedder.embed_cards(cards)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            SnapshotWriter(root).write(
                name="s", t_q=8, meta=VideoMeta("v", 10, 2, 20),
                budget=Budget(100_000, 1), cards=[card.serializable() for card in cards],
                raw_frames=[], writer_calls=1, config={}, embedder=vars(embedder.metadata),
            )
            output = answer_card_snapshot(
                query_id="q", query="door", snapshot=SnapshotReader(root / "s"),
                embedder=embedder, protocol=ProtocolConfig(not_found_threshold=0),
                budget=BudgetConfig(100_000, 1, refine_calls_per_query=0),
            )
        payload = output.to_dict()
        self.assertEqual(payload["status"], "fallback")
        self.assertEqual(payload["video_id"], "v")
        self.assertEqual(payload["retrieved_cards"][0], "door")
        self.assertEqual(payload["span"], [1.0, 3.0])
        self.assertNotIn("gt", str(payload).lower())
        self.assertIn("query_embedding", payload["resource"])

    def test_output_rejects_gt_and_not_found_span(self):
        with self.assertRaisesRegex(ValueError, "ground truth"):
            QueryOutput("q", "v", .5, "error", None, 0, (), (), (), {}, {"gt_span": [1, 2]})
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            QueryOutput("q", "v", .5, "not_found", (1, 2), 0, (), (), (), {}, {})


if __name__ == "__main__":
    unittest.main()

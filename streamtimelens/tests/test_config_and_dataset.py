import math
import tempfile
import unittest
from pathlib import Path

from streamtimelens.config import BudgetConfig, ProtocolConfig, resolve_config
from streamtimelens.protocol.datasets.timelens import adapt_timelens_annotations, group_by_video, stable_query_id


ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_resolution_is_hash_stable_across_yaml_key_order(self):
        config = resolve_config(protocol_path=ROOT / "configs/protocol.yaml", budget_path=ROOT / "configs/budgets/1m.yaml",
                                method_path=ROOT / "configs/methods/full.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            alternative = Path(temporary) / "method.yaml"
            alternative.write_text("raw_cache: boundary_and_ring\nretrieval: lexical_mmr\nwriter: deterministic_unknown_event\ntrigger: joint\nmethod: streamtimelens-p0\n")
            reordered = resolve_config(protocol_path=ROOT / "configs/protocol.yaml", budget_path=ROOT / "configs/budgets/1m.yaml",
                                       method_path=alternative)
        self.assertEqual(config.sha256, reordered.sha256)

    def test_unknown_and_query_leaking_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bad = Path(temporary) / "bad.yaml"
            bad.write_text("method: x\ntrigger: y\nwriter: z\nretrieval: q\nraw_cache: r\nquery_path: leaked.jsonl\n")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                resolve_config(protocol_path=ROOT / "configs/protocol.yaml", budget_path=ROOT / "configs/budgets/1m.yaml", method_path=bad)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            resolve_config(protocol_path=ROOT / "configs/protocol.yaml", budget_path=ROOT / "configs/budgets/1m.yaml",
                           method_path=ROOT / "configs/methods/full.yaml", overrides={"method": {"video_path": "leak.mp4"}})

    def test_s3_method_configs_are_strict_and_loadable(self):
        for name in ("uniform_raw", "semantic_reservoir"):
            config = resolve_config(
                protocol_path=ROOT / "configs/protocol.yaml",
                budget_path=ROOT / "configs/budgets/1m.yaml",
                method_path=ROOT / f"configs/methods/{name}.yaml",
                model_path=ROOT / "configs/models.yaml",
            )
            self.assertEqual(config.method.method, name)
            self.assertEqual(config.model.clip_model, "openai/clip-vit-base-patch32")
            self.assertEqual(config.method.writer, "none")

    def test_non_finite_protocol_and_budget_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            ProtocolConfig(decode_fps=math.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            ProtocolConfig(arrival_ratios=(0.25, math.inf))
        with self.assertRaisesRegex(ValueError, "finite"):
            BudgetConfig(memory_bytes=1024, writer_calls_per_minute=math.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            BudgetConfig(memory_bytes=math.inf, writer_calls_per_minute=1)
        with self.assertRaisesRegex(ValueError, "forest"):
            ProtocolConfig(forest_max_roots=0)
        with self.assertRaisesRegex(ValueError, "forest"):
            ProtocolConfig(utility_has_raw_weight=-1)


class TimeLensAdapterTest(unittest.TestCase):
    def test_multiple_spans_and_text_normalization_have_stable_ids(self):
        rows = [{"video_id": "v", "description": " Open   Door ", "duration": 8,
                 "timestamps": [[1, 2], [3, 5]], "dataset": "fixture"}]
        records = adapt_timelens_annotations(rows)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].query_id, stable_query_id("v", "open door", (1, 2)))
        self.assertEqual(list(group_by_video(records)), ["v"])

    def test_invalid_and_duplicate_annotations_fail(self):
        with self.assertRaises(ValueError):
            adapt_timelens_annotations([{"video_id": "v", "query": "x", "duration": 2, "span": [2, 1]}])
        duplicate = {"video_id": "v", "query": "x", "duration": 2, "span": [0, 1]}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            adapt_timelens_annotations([duplicate, duplicate])
        with self.assertRaisesRegex(FileNotFoundError, "unavailable video"):
            adapt_timelens_annotations([duplicate], available_video_ids={"other"})

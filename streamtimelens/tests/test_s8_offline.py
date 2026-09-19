import unittest
import hashlib
import json
import tempfile
from pathlib import Path

from streamtimelens.baselines.offline_timelens import (
    LoadedOfflineVideo, OfflineQuery, OfflineTimeLensWrapper,
)
from streamtimelens.baselines.offline_resume import (
    sha256_file, sha256_model, validate_prediction_rows, validate_resume_source,
)


class Service:
    def generate(self, messages, videos, max_new_tokens):
        del messages, videos, max_new_tokens
        return "The event happens in 1 - 2 seconds"


class Loader:
    def __init__(self):
        self.paths = []

    def __call__(self, path):
        self.paths.append(path)
        return LoadedOfflineVideo(("video", {}), 10, 20, 4096)


class OfflineWrapperTest(unittest.TestCase):
    def test_same_video_is_reloaded_and_costed_for_every_query(self):
        loader = Loader()
        queries = [
            OfflineQuery("q1", "v", "door", Path("v.mp4")),
            OfflineQuery("q2", "v", "walk", Path("v.mp4")),
        ]
        predictions = OfflineTimeLensWrapper(Service(), loader=loader).run(queries)
        self.assertEqual(len(loader.paths), 2)
        self.assertEqual([row.span for row in predictions], [(1.0, 2.0), (1.0, 2.0)])
        self.assertTrue(all(row.resource["full_video_reloads"] == 1 for row in predictions))
        self.assertTrue(all(row.resource["visual_tokens"] == 4096 for row in predictions))
        self.assertTrue(all(row.resource["component"] == "offline_timelens_query" for row in predictions))

    def test_invalid_query_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "fields"):
            OfflineTimeLensWrapper(Service(), loader=Loader()).run([
                OfflineQuery("", "v", "door", Path("v.mp4")),
            ])


class OfflineResumeTest(unittest.TestCase):
    def test_model_directory_hash_is_path_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").write_bytes(b"model")
            first = sha256_model(root)
            (root / "b").write_bytes(b"model")
            self.assertNotEqual(first, sha256_model(root))

    def test_duplicate_unknown_and_wrong_shard_predictions_fail(self):
        queries = [{"query_id": "q1", "video_id": "v1"}]
        shard = int(hashlib.sha256(b"q1").hexdigest(), 16) % 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_prediction_rows([
                {"query_id": "q1", "video_id": "v1"},
                {"query_id": "q1", "video_id": "v1"},
            ], queries, num_shards=2, shard_index=shard)
        with self.assertRaisesRegex(ValueError, "manifest"):
            validate_prediction_rows([
                {"query_id": "unknown", "video_id": "v1"},
            ], queries, num_shards=2, shard_index=shard)
        with self.assertRaisesRegex(ValueError, "shard"):
            validate_prediction_rows([
                {"query_id": "q1", "video_id": "v1"},
            ], queries, num_shards=2, shard_index=1 - shard)

    def test_resume_source_requires_matching_hashes_shard_and_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queries = [{"query_id": "q1", "video_id": "v1"}]
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"query_id": "q1", "video_id": "v1"}) + "\n", encoding="utf-8",
            )
            config = {
                "protocol": "offline_upper_bound_not_streaming",
                "model_sha256": "model", "query_manifest_sha256": "queries",
                "num_shards": 1, "shard_index": 0,
            }
            (root / "config.resolved.json").write_text(json.dumps(config), encoding="utf-8")
            (root / "provenance.json").write_text(
                json.dumps({"code": {"commit": "a" * 40}}), encoding="utf-8",
            )
            rows, metadata = validate_resume_source(
                root, model_sha256="model", query_manifest_sha256="queries", queries=queries,
                num_shards=1, shard_index=0,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(metadata["source_commit"], "a" * 40)
            self.assertEqual(metadata["predictions_sha256"], sha256_file(predictions))
            config["model_sha256"] = "other"
            (root / "config.resolved.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                validate_resume_source(
                    root, model_sha256="model", query_manifest_sha256="queries", queries=queries,
                    num_shards=1, shard_index=0,
                )


if __name__ == "__main__":
    unittest.main()

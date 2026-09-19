import json
import tempfile
import unittest
from pathlib import Path

from baas.runner import _filter_fixed_samples, _select_annos


class RunnerSelectionTest(unittest.TestCase):
    def test_limit_is_global_before_multi_gpu_sharding(self):
        annos = [{"id": index} for index in range(20)]
        chunks = [
            _select_annos(annos, limit=5, chunk=2, index=index)
            for index in range(2)
        ]
        self.assertEqual([[item["id"] for item in chunk] for chunk in chunks], [[0, 2, 4], [1, 3]])
        self.assertEqual(sum(map(len, chunks)), 5)

    def test_fixed_sample_manifest_preserves_requested_order(self):
        annos = [
            {"video_path": "/v/B.mp4", "query": "second", "span": [[3, 4]]},
            {"video_path": "/v/A.mp4", "query": "first", "span": [[1, 2]]},
        ]
        requested = [
            {"video_id": "A", "query": "first", "span": [1, 2]},
            {"video_id": "B", "query": "second", "span": [3, 4]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.json"
            path.write_text(json.dumps(requested), encoding="utf-8")
            selected = _filter_fixed_samples(annos, path)
        self.assertEqual([Path(item["video_path"]).stem for item in selected], ["A", "B"])


if __name__ == "__main__":
    unittest.main()

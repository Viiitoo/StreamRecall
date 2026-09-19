import json
import tempfile
import unittest
from pathlib import Path

from streamtimelens.evaluation.matrix import (
    MatrixDataset, MatrixQuery, MatrixSpec, MatrixVideo, execute_matrix, expand_matrix,
)


class MatrixRunnerTest(unittest.TestCase):
    def _spec(self):
        dataset = MatrixDataset(
            "dev", (MatrixVideo("v1", "/v1.mp4"), MatrixVideo("v2", "/v2.mp4")),
            (MatrixQuery("q1", "v1", "door"), MatrixQuery("q2", "v2", "walk")),
        )
        return MatrixSpec(
            ("full.yaml", "uniform.yaml"), ("256k.yaml", "1m.yaml"), (dataset,),
            (.25, .5, .75, 1.0), ("0", "1", "2", "3"),
            ("ingest", "{video_id}", "{rhos}", "{job_dir}"),
            ("query", "{query_id}", "{rho}", "{snapshot}"),
        )

    def test_expansion_ingests_once_reuses_four_rhos_and_resumes_by_hash(self):
        calls = []

        def runner(command, env):
            calls.append((tuple(command), env["CUDA_VISIBLE_DEVICES"]))
            return 0

        with tempfile.TemporaryDirectory() as temporary:
            jobs = expand_matrix(self._spec(), Path(temporary))
            first = execute_matrix(jobs, runner=runner)
            first_call_count = len(calls)
            second = execute_matrix(jobs, runner=runner)
        self.assertEqual(len(jobs), 8)
        self.assertEqual(first["ingest_executed"], 8)
        self.assertEqual(first["query_executed"], 32)
        self.assertEqual(second["ingest_resumed"], 8)
        self.assertEqual(second["query_resumed"], 32)
        self.assertEqual(len(calls), first_call_count)
        ingest_calls = [call for call, _ in calls if call[0] == "ingest"]
        self.assertTrue(all(call[2] == "0.25,0.5,0.75,1" for call in ingest_calls))
        self.assertLessEqual(len({gpu for _, gpu in calls}), 4)

    def test_conflicting_marker_and_config_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            jobs = expand_matrix(self._spec(), Path(temporary))
            job = jobs[0]
            execute_matrix([job], runner=lambda command, env: 0)
            marker = job.job_dir / ".ingest.complete.json"
            row = json.loads(marker.read_text(encoding="utf-8"))
            row["job_hash"] = "different"
            marker.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash conflict"):
                execute_matrix([job], runner=lambda command, env: 0)

    def test_query_grid_reuses_ingest_and_expands_only_query_jobs(self):
        base = self._spec()
        spec = MatrixSpec(
            base.methods[:1], base.budgets[:1], base.datasets, base.rhos[:1], base.gpu_ids,
            base.ingest_command,
            ("query", "{query_id}", "{top_k}", "{query_variant}"),
            query_grid=({"top_k": 4}, {"top_k": 8}),
        )
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            jobs = expand_matrix(spec, Path(temporary))
            result = execute_matrix(jobs, runner=lambda command, env: calls.append(command) or 0)
        self.assertEqual(result["ingest_executed"], 2)
        self.assertEqual(result["query_executed"], 4)
        self.assertEqual({call[2] for call in calls if call[0] == "query"}, {"4", "8"})

    def test_batch_query_mode_runs_once_per_video_job(self):
        base = self._spec()
        spec = MatrixSpec(
            base.methods[:1], base.budgets[:1], base.datasets, base.rhos, base.gpu_ids,
            base.ingest_command, ("batch", "{queries}", "{query_grid}", "{snapshot_root}"),
            query_grid=({"top_k": 4}, {"top_k": 8}), query_mode="per_video_grid",
        )
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            jobs = expand_matrix(spec, Path(temporary))
            result = execute_matrix(jobs, runner=lambda command, env: calls.append(command) or 0)
        self.assertEqual(result["ingest_executed"], 2)
        self.assertEqual(result["query_executed"], 2)
        self.assertEqual(len([call for call in calls if call[0] == "batch"]), 2)


if __name__ == "__main__":
    unittest.main()

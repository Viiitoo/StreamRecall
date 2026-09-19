import unittest

from streamtimelens.evaluation.resources import ResourceObservation, aggregate_resources


class ResourceAggregationTest(unittest.TestCase):
    def test_phase_component_latency_amortization_and_realtime(self):
        rows = [
            ResourceObservation("v", "ingest", "observer", 2, cpu_s=1, stream_duration_s=10,
                                state_peak_bytes=100, snapshot_bytes=80, max_backlog_s=.2),
            ResourceObservation("v", "ingest", "writer", 3, cuda_s=2, stream_duration_s=10),
            ResourceObservation("v", "query", "embed", 1, query_id="q1", cold_query=True),
            ResourceObservation("v", "query", "refine", 2, query_id="q1", cold_query=True),
            ResourceObservation("v", "query", "embed", .5, query_id="q2", cold_query=False),
        ]
        result = aggregate_resources(rows)
        self.assertEqual(result["components"]["ingest"]["writer"]["cuda_s"], 2)
        self.assertEqual(result["ingest_wall_s"], 5)
        self.assertEqual(result["cold_query_wall_s_mean"], 3)
        self.assertEqual(result["warm_query_wall_s_mean"], .5)
        self.assertEqual(result["amortized"]["1"]["wall_s_per_query"], 8)
        self.assertEqual(result["amortized"]["all"]["wall_s_per_query"], 4.25)
        self.assertEqual(result["realtime_throughput_mean"], 2)
        self.assertEqual(result["state_peak_bytes"], 100)
        self.assertEqual(result["max_backlog_s"], .2)

    def test_empty_and_invalid_query_resource(self):
        self.assertEqual(aggregate_resources([])["query_wall_s"], 0)
        with self.assertRaisesRegex(ValueError, "query ID"):
            ResourceObservation("v", "query", "embed", 1)


if __name__ == "__main__":
    unittest.main()

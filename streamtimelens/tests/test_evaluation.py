import time
import unittest

from streamtimelens.evaluation.cost import ComponentTimer, aggregate_resource_records
from streamtimelens.evaluation.trace import TraceLog


class CostTest(unittest.TestCase):
    def test_cpu_timer_and_nested_timer_complete(self):
        with ComponentTimer("outer") as outer:
            with ComponentTimer("inner") as inner:
                time.sleep(0.001)
        self.assertEqual(outer.record.status, "ok")
        self.assertEqual(inner.record.status, "ok")
        self.assertIsNone(outer.record.cuda_s)
        self.assertGreaterEqual(outer.record.wall_s, inner.record.wall_s)

    def test_timer_records_failure_and_aggregation_counts_it(self):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with ComponentTimer("writer") as timer:
                raise RuntimeError("boom")
        self.assertEqual(timer.record.status, "error")
        totals = aggregate_resource_records([timer.record])
        self.assertEqual(totals["writer"]["calls"], 1)
        self.assertEqual(totals["writer"]["errors"], 1)


class TraceTest(unittest.TestCase):
    def test_sequence_is_monotonic_and_jsonl_is_stable(self):
        trace = TraceLog()
        trace.record(kind="frame_seen", timestamp_s=1, reason="decoder")
        trace.record(kind="writer_called", timestamp_s=2, reason="periodic", object_id="card")
        self.assertEqual([row["sequence"] for row in trace], [0, 1])
        self.assertEqual(len(trace.to_jsonl().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()

import math
import unittest

from streamtimelens.evaluation.types import ResourceRecord
from streamtimelens.protocol.types import FramePacket, Prediction, QueryRecord, VideoMeta


class PersistentTypesTest(unittest.TestCase):
    def test_json_and_msgpack_round_trip(self):
        records = [
            VideoMeta("v", 10, 25, 250, 1920, 1080),
            FramePacket(1, 25, b"jpeg", 224, 224, video_id="v"),
            QueryRecord("q", "v", "open door", (1, 2), 5, .5, "natural"),
            Prediction(1, 2, .8, "ok", ("e",), "q", ("c",)),
            ResourceRecord("writer", .1, .1, None, "ok", 4, 5),
        ]
        for record in records:
            self.assertEqual(type(record).from_json(record.to_json()), record)
            self.assertEqual(type(record).from_msgpack(record.to_msgpack()), record)

    def test_invalid_times_spans_and_confidence_are_rejected(self):
        with self.assertRaises(ValueError):
            VideoMeta("v", math.nan, 1, 1)
        with self.assertRaises(ValueError):
            FramePacket(-1, 0, b"", 1, 1)
        with self.assertRaises(ValueError):
            QueryRecord("q", "v", "x", (2, 1), 1, .5, "natural")
        with self.assertRaises(ValueError):
            Prediction(1, 1, .5, "ok")
        with self.assertRaises(ValueError):
            ResourceRecord("x", 0, 0, None, "ok", -1, 0)


if __name__ == "__main__":
    unittest.main()

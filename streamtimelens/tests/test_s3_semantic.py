import unittest

import numpy as np

from streamtimelens.memory.raw_cache import RawFrameCache
from streamtimelens.memory.semantic_reservoir import SemanticReservoir
from streamtimelens.protocol.types import FramePacket


class SemanticReservoirTest(unittest.TestCase):
    def test_static_then_scene_change_retains_anchor_and_novelty_boundedly(self):
        store = RawFrameCache()
        cache = SemanticReservoir(store, 4, duration_s=20)
        for index in range(200):
            vector = np.array([1, 0, 0]) if index < 100 else np.array([0, 1, 0])
            cache.observe(FramePacket(index / 10, index, str(index).encode(), 1, 1), vector)
            self.assertLessEqual(len(cache), 4)
        self.assertEqual({item.role for item in cache.frames}, {"temporal_anchor", "semantic_novelty"})
        self.assertLessEqual(len(store), 4)

    def test_one_frame_capacity_is_anchor_only(self):
        cache = SemanticReservoir(RawFrameCache(), 1, duration_s=3)
        for index in range(3):
            vector = np.eye(3, dtype=np.float32)[index]
            cache.observe(FramePacket(index, index, bytes([index]), 1, 1), vector)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.frames[0].role, "temporal_anchor")


if __name__ == "__main__":
    unittest.main()

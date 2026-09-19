import unittest

import numpy as np

from streamtimelens.memory.raw_cache import RawFrameCache
from streamtimelens.memory.uniform import UniformRawCache
from streamtimelens.protocol.types import FramePacket


def embedding(index):
    vector = np.zeros(4, dtype=np.float32)
    vector[index % 4] = 1
    return vector


class UniformRawTest(unittest.TestCase):
    def test_known_slots_are_bounded_and_resize_does_not_reread(self):
        store = RawFrameCache()
        cache = UniformRawCache(store, 4, duration_s=100, seed=7)
        for index in range(101):
            cache.observe(FramePacket(index, index, bytes([index % 251]), 1, 1), embedding(index))
        self.assertEqual([round(item.ref.timestamp_s) for item in cache.frames], [12, 37, 62, 87])
        retained = {item.ref.frame_index for item in cache.frames}
        cache.resize(2)
        self.assertTrue({item.ref.frame_index for item in cache.frames}.issubset(retained))

    def test_unknown_reservoir_is_seed_deterministic(self):
        def run():
            cache = UniformRawCache(RawFrameCache(), 5, seed=11, pixel_only=True)
            for index in range(100):
                cache.observe(FramePacket(index, index, str(index).encode(), 1, 1), None)
            return [item.ref.frame_index for item in cache.frames]
        self.assertEqual(run(), run())
        self.assertEqual(len(run()), 5)


if __name__ == "__main__":
    unittest.main()

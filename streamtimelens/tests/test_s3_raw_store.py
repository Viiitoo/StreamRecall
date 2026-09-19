import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from streamtimelens.memory.raw_cache import RawFrameCache, compress_rgb_image
from streamtimelens.protocol.types import FramePacket


def packet(index, timestamp, value=0):
    image = Image.fromarray(np.full((16, 24, 3), value, dtype=np.uint8), mode="RGB")
    output = BytesIO()
    image.save(output, format="PNG")
    return FramePacket(timestamp, index, output.getvalue(), 24, 16, video_id="v")


class SharedRawStoreTest(unittest.TestCase):
    def test_dedup_resize_refcounts_and_verified_read(self):
        store = RawFrameCache()
        first = store.add(packet(0, 0, 50), owner="ring")
        second = store.add(packet(1, 1, 50), owner="persistent")
        self.assertEqual(first.content_id, second.content_id)
        self.assertEqual(len(store.items()), 1)
        with Image.open(BytesIO(store.get(first.frame_id))) as decoded:
            self.assertEqual(min(decoded.size), 224)
        store.retain(first.frame_id, "active_segment")
        store.release(first.frame_id, "ring")
        store.release(first.frame_id, "active_segment")
        self.assertNotIn(first.frame_id, store)
        self.assertEqual(store.content_refcount(second.content_id), 1)

    def test_budget_failure_is_atomic(self):
        payload, _, _ = compress_rgb_image(np.zeros((8, 12, 3), dtype=np.uint8))
        store = RawFrameCache(max_bytes=len(payload) - 1)
        with self.assertRaises(MemoryError):
            store.add(packet(0, 0))
        self.assertEqual((len(store), store.byte_size), (0, 0))


if __name__ == "__main__":
    unittest.main()

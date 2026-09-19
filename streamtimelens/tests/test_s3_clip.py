import unittest

import numpy as np

from streamtimelens.observer.clip_encoder import (
    FrozenCLIPEncoder, deserialize_embedding, serialize_embedding,
)


class Backend:
    def encode_images(self, images):
        return np.asarray([[float(np.asarray(image).mean()), 1.0, 2.0] for image in images])
    def encode_texts(self, texts):
        return np.asarray([[len(text), 1.0, 2.0] for text in texts], dtype=np.float32)


class FrozenClipTest(unittest.TestCase):
    def test_batch_single_normalization_and_persistence(self):
        images = [np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)]
        batch = FrozenCLIPEncoder(backend=Backend(), batch_size=8).encode_images(images)
        singles = np.concatenate([
            FrozenCLIPEncoder(backend=Backend(), batch_size=1).encode_images([image]) for image in images
        ])
        np.testing.assert_allclose(batch, singles, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(batch, axis=1), np.ones(2), atol=1e-6)
        for precision, tolerance in (("fp16", 5e-4), ("int8", 1e-2)):
            restored = deserialize_embedding(serialize_embedding(batch[1], precision))
            np.testing.assert_allclose(restored, batch[1], atol=tolerance)

    def test_visual_half_fps_gate(self):
        encoder = FrozenCLIPEncoder(backend=Backend(), visual_fps=.5)
        self.assertEqual([encoder.due(t) for t in (0, 1, 2, 3.9, 4)], [True, False, True, False, True])
        encoder.reset_visual_clock()
        self.assertTrue(encoder.due(0))


if __name__ == "__main__":
    unittest.main()

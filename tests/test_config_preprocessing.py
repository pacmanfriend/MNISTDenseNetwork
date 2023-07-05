import unittest

import numpy as np

from app.config import TrainingConfig
from app.preprocessing import flatten_and_normalize_images, one_hot_labels, prepare_digit_image


class TrainingConfigTest(unittest.TestCase):
    def test_valid_config_has_no_errors(self):
        config = TrainingConfig(
            train_size=100,
            epochs=2,
            batch_size=16,
            alpha=0.1,
            num_workers=1,
        )

        self.assertEqual(config.validate(max_train_size=1000, max_workers=4), [])

    def test_invalid_config_reports_errors(self):
        config = TrainingConfig(
            train_size=0,
            epochs=0,
            batch_size=0,
            alpha=0.0,
            num_workers=8,
        )

        errors = config.validate(max_train_size=1000, max_workers=4)

        self.assertGreaterEqual(len(errors), 5)


class PreprocessingTest(unittest.TestCase):
    def test_flatten_and_normalize_images(self):
        images = np.full((2, 28, 28), 255, dtype=np.uint8)

        result = flatten_and_normalize_images(images)

        self.assertEqual(result.shape, (2, 784))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.all(result == 1.0))

    def test_one_hot_labels(self):
        result = one_hot_labels(np.array([0, 2]), classes=3)

        np.testing.assert_array_equal(
            result,
            np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32),
        )

    def test_prepare_digit_image(self):
        image = np.full((28, 28), 255, dtype=np.uint8)

        result = prepare_digit_image(image)

        self.assertEqual(result.shape, (1, 784))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.all(result == 1.0))


if __name__ == "__main__":
    unittest.main()

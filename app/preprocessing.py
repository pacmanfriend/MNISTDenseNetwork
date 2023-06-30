from __future__ import annotations

import numpy as np


def flatten_and_normalize_images(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 3:
        return images.reshape(images.shape[0], 28 * 28).astype(np.float32) / 255.0
    if images.ndim == 2 and images.shape[1] == 28 * 28:
        return images.astype(np.float32) / 255.0
    raise ValueError(f"Expected images with shape (N, 28, 28) or (N, 784), got {images.shape}")


def one_hot_labels(labels: np.ndarray, classes: int = 10) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"Expected labels with shape (N,), got {labels.shape}")
    return np.eye(classes, dtype=np.float32)[labels]


def prepare_digit_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.shape != (28, 28):
        raise ValueError(f"Expected image with shape (28, 28), got {image.shape}")
    return image.astype(np.float32).reshape(1, 28 * 28) / 255.0

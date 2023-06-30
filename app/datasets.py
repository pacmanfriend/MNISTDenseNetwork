from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MnistData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def max_train_size(self) -> int:
        return int(self.x_train.shape[0])


def load_mnist() -> MnistData:
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return MnistData(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

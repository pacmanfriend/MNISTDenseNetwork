from __future__ import annotations

import numpy as np

from neural import DenseNetwork


class AppState:
    def __init__(self):
        self.model: DenseNetwork | None = None
        self.x_train: np.ndarray | None = None
        self.x_test:  np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test:  np.ndarray | None = None

    @property
    def data_loaded(self) -> bool:
        return self.x_train is not None

    @property
    def max_train_size(self) -> int:
        return len(self.x_train) if self.x_train is not None else 0

    def load_mnist(self) -> None:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test

from __future__ import annotations

from app.datasets import MnistData
from neural import DenseNetwork


class AppState:
    def __init__(self):
        self.model: DenseNetwork | None = None
        self.mnist: MnistData | None = None

    @property
    def data_loaded(self) -> bool:
        return self.mnist is not None

    @property
    def max_train_size(self) -> int:
        return self.mnist.max_train_size if self.mnist is not None else 0

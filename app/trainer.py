from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from app.state import AppState


class TrainingController:
    def __init__(self, state: AppState):
        self._state = state
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        train_size: int,
        epochs: int,
        batch_size: int,
        alpha: float,
        num_workers: int,
        on_epoch_end: Callable[[int, float, float], None] | None = None,
        on_done: Callable[[float, float], None] | None = None,
    ) -> None:
        if self.is_running:
            return

        state = self._state
        images      = state.x_train[:train_size].reshape(train_size, 28 * 28).astype(np.float32) / 255.0
        labels      = np.eye(10, dtype=np.float32)[state.y_train[:train_size]]
        test_images = state.x_test.reshape(len(state.x_test), 28 * 28).astype(np.float32) / 255.0
        test_labels = np.eye(10, dtype=np.float32)[state.y_test]

        def run():
            t0 = time.monotonic()
            state.model.fit(
                images, labels,
                batch_size=batch_size, epochs=epochs,
                validation_split=0.1, alpha=alpha,
                num_workers=num_workers, on_epoch_end=on_epoch_end,
            )
            elapsed = time.monotonic() - t0
            acc = state.model.evaluate(test_images, test_labels, verbose=True)
            if on_done is not None:
                on_done(acc, elapsed)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

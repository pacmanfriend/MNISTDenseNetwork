from __future__ import annotations

import threading
import time
from typing import Callable

from app.config import TrainingConfig
from app.preprocessing import flatten_and_normalize_images, one_hot_labels
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
        config: TrainingConfig,
        on_epoch_end: Callable[[int, float, float], None] | None = None,
        on_done: Callable[[float, float], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        if self.is_running:
            return

        state = self._state
        if state.model is None:
            raise RuntimeError("Model is not created or loaded.")
        if state.mnist is None:
            raise RuntimeError("MNIST data is not loaded.")

        errors = config.validate(state.max_train_size)
        if errors:
            raise ValueError("\n".join(errors))

        mnist = state.mnist
        images = flatten_and_normalize_images(mnist.x_train[:config.train_size])
        labels = one_hot_labels(mnist.y_train[:config.train_size])
        test_images = flatten_and_normalize_images(mnist.x_test)
        test_labels = one_hot_labels(mnist.y_test)
        model = state.model

        def run():
            try:
                t0 = time.monotonic()
                model.fit(
                    images, labels,
                    batch_size=config.batch_size, epochs=config.epochs,
                    validation_split=config.validation_split, alpha=config.alpha,
                    num_workers=config.num_workers, on_epoch_end=on_epoch_end,
                    verbose=False,
                )
                elapsed = time.monotonic() - t0
                acc = model.evaluate(test_images, test_labels, verbose=False)
                if on_done is not None:
                    on_done(acc, elapsed)
            except Exception as exc:
                if on_error is not None:
                    on_error(exc)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

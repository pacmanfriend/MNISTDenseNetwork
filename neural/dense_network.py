from __future__ import annotations

import multiprocessing
import numpy as np
import sys
import time
from pathlib import Path
from typing import Callable

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

_MIN_SAMPLES_PER_WORKER = 5_000


class DenseNetwork:
    def __init__(self, hidden_size: int = 100):
        self.input_size = 784
        self.hidden_size = hidden_size
        self.output_size = 10

        self.weights_0_1 = None
        self.weights_1_2 = None
        self.weights_2_3 = None

    def init_weights(self):
        self.weights_0_1 = 0.2 * np.random.random((self.input_size, self.hidden_size)) - 0.1
        self.weights_1_2 = 0.2 * np.random.random((self.hidden_size, self.hidden_size)) - 0.1
        self.weights_2_3 = 0.2 * np.random.random((self.hidden_size, self.output_size)) - 0.1

    def _forward(self, x: np.ndarray) -> np.ndarray:
        l1 = tanh(x @ self.weights_0_1)
        l2 = tanh(l1 @ self.weights_1_2)
        return softmax(l2 @ self.weights_2_3)

    def get_info(self) -> dict:
        initialized = all(w is not None for w in [self.weights_0_1, self.weights_1_2, self.weights_2_3])
        info = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'initialized': initialized,
        }
        if initialized:
            info['shapes'] = {
                'weights_0_1': self.weights_0_1.shape,
                'weights_1_2': self.weights_1_2.shape,
                'weights_2_3': self.weights_2_3.shape,
            }
            info['total_params'] = self.weights_0_1.size + self.weights_1_2.size + self.weights_2_3.size
        return info

    def fit(
        self,
        x_train,
        y_train,
        batch_size: int,
        epochs: int,
        validation_split: float = 0.1,
        alpha: float = 0.01,
        num_workers: int | None = None,
        on_epoch_end: Callable[[int, float, float], None] | None = None,
        verbose: bool = True,
    ) -> dict:
        if self.weights_0_1 is None or self.weights_1_2 is None or self.weights_2_3 is None:
            self.init_weights()

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 < validation_split < 1.0):
            raise ValueError("validation_split must be in (0, 1)")

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        if x_train.ndim != 2 or x_train.shape[1] != self.input_size:
            raise ValueError(f"x_train must have shape (N, {self.input_size}), got {x_train.shape}")
        if y_train.ndim != 2 or y_train.shape[1] != self.output_size:
            raise ValueError(f"y_train must be one-hot with shape (N, {self.output_size}), got {y_train.shape}")
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same length")

        validation_size = int(len(x_train) * validation_split)
        train_size = len(x_train) - validation_size

        if num_workers is None:
            num_workers = min(
                multiprocessing.cpu_count(),
                max(1, train_size // _MIN_SAMPLES_PER_WORKER),
            )
        else:
            num_workers = max(1, min(num_workers, multiprocessing.cpu_count()))

        history: dict[str, list[float]] = {'train_acc': [], 'val_acc': []}

        for e in range(epochs):
            perm = np.random.permutation(len(x_train))
            train_images = x_train[perm[:train_size]]
            train_labels = y_train[perm[:train_size]]
            val_images   = x_train[perm[train_size:]]
            val_labels   = y_train[perm[train_size:]]

            start = time.monotonic()

            shards_x = np.array_split(train_images, num_workers)
            shards_y = np.array_split(train_labels, num_workers)
            shard_args = [
                (sx, sy, self.weights_0_1, self.weights_1_2, self.weights_2_3, batch_size)
                for sx, sy in zip(shards_x, shards_y)
            ]

            if num_workers > 1:
                with multiprocessing.Pool(num_workers) as pool:
                    results = pool.map(_train_shard, shard_args)
            else:
                results = [_train_shard(shard_args[0])]

            dw01 = sum(r[0] for r in results)
            dw12 = sum(r[1] for r in results)
            dw23 = sum(r[2] for r in results)
            correct_cnt = sum(r[3] for r in results)

            total_batches = sum(int(np.ceil(sx.shape[0] / batch_size)) for sx in shards_x)
            self.weights_0_1 += alpha * dw01 / total_batches
            self.weights_1_2 += alpha * dw12 / total_batches
            self.weights_2_3 += alpha * dw23 / total_batches

            train_acc = correct_cnt / float(train_size)

            val_acc = 0.0
            if validation_size > 0:
                val_probs = self._forward(val_images)
                val_acc = float((val_probs.argmax(axis=1) == val_labels.argmax(axis=1)).mean())

            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            elapsed = time.monotonic() - start
            if verbose:
                sys.stdout.write(
                    f"I:{e} || Val-Acc:{val_acc:.4f} || Train-Acc:{train_acc:.4f} || {elapsed:.2f}s\n"
                )

            if on_epoch_end is not None:
                on_epoch_end(e, train_acc, val_acc)

        return history

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"x must have shape (N, {self.input_size}) or ({self.input_size},), got {x.shape}")
        if self.weights_0_1 is None or self.weights_1_2 is None or self.weights_2_3 is None:
            raise RuntimeError("Model is not initialized. Call fit() or init_weights() first.")
        return self._forward(x).argmax(axis=1)

    def evaluate(self, x_test, y_test, verbose: bool = True) -> float:
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        if x_test.ndim != 2 or x_test.shape[1] != self.input_size:
            raise ValueError(f"x_test must have shape (N, {self.input_size}), got {x_test.shape}")
        if y_test.ndim != 2 or y_test.shape[1] != self.output_size:
            raise ValueError(f"y_test must be one-hot with shape (N, {self.output_size}), got {y_test.shape}")
        if x_test.shape[0] != y_test.shape[0]:
            raise ValueError("x_test and y_test must have the same length")
        if self.weights_0_1 is None or self.weights_1_2 is None or self.weights_2_3 is None:
            raise RuntimeError("Model is not initialized. Call fit() or init_weights() first.")

        probs = self._forward(x_test)
        acc = float((probs.argmax(axis=1) == y_test.argmax(axis=1)).mean())
        if verbose:
            sys.stdout.write(f"Test-Acc:{acc:.4f}\n")
        return acc

    def save_model(self, path: str | Path) -> None:
        if self.weights_0_1 is None or self.weights_1_2 is None or self.weights_2_3 is None:
            raise RuntimeError("Model is not initialized. Call fit() or init_weights() first.")
        if h5py is None:
            raise ModuleNotFoundError("h5py is required for HDF5 save/load. Install it via requirements.txt.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f:
            arch = f.create_group("arch")
            arch.attrs["input_size"] = int(self.input_size)
            arch.attrs["hidden_size"] = int(self.hidden_size)
            arch.attrs["output_size"] = int(self.output_size)

            w = f.create_group("weights")
            w.create_dataset("weights_0_1", data=self.weights_0_1)
            w.create_dataset("weights_1_2", data=self.weights_1_2)
            w.create_dataset("weights_2_3", data=self.weights_2_3)

    def load_model(self, path: str | Path) -> None:
        if h5py is None:
            raise ModuleNotFoundError("h5py is required for HDF5 save/load. Install it via requirements.txt.")

        path = Path(path)
        with h5py.File(path, "r") as f:
            arch = f["arch"].attrs
            self.input_size  = int(arch["input_size"])
            self.hidden_size = int(arch["hidden_size"])
            self.output_size = int(arch["output_size"])

            w = f["weights"]
            self.weights_0_1 = np.asarray(w["weights_0_1"], dtype=np.float64)
            self.weights_1_2 = np.asarray(w["weights_1_2"], dtype=np.float64)
            self.weights_2_3 = np.asarray(w["weights_2_3"], dtype=np.float64)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh2deriv(x: np.ndarray) -> np.ndarray:
    return 1 - (x ** 2)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def _train_shard(args):
    x_shard, y_shard, w01, w12, w23, batch_size = args
    shard_size = x_shard.shape[0]

    dw01 = np.zeros_like(w01)
    dw12 = np.zeros_like(w12)
    dw23 = np.zeros_like(w23)
    correct_cnt = 0

    num_batches = int(np.ceil(shard_size / float(batch_size)))
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, shard_size)
        bs = batch_end - batch_start

        layer_0 = x_shard[batch_start:batch_end]
        layer_1 = tanh(layer_0 @ w01)
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1_dropout = layer_1 * dropout_mask * 2
        layer_2 = tanh(layer_1_dropout @ w12)
        layer_3 = softmax(layer_2 @ w23)

        correct_cnt += int(
            (layer_3.argmax(axis=1) == y_shard[batch_start:batch_end].argmax(axis=1)).sum()
        )

        layer_3_delta = (y_shard[batch_start:batch_end] - layer_3) / float(bs)
        layer_2_delta = layer_3_delta @ w23.T * tanh2deriv(layer_2)
        layer_1_delta = layer_2_delta @ w12.T * tanh2deriv(layer_1) * dropout_mask * 2

        dw23 += layer_2.T @ layer_3_delta
        dw12 += layer_1_dropout.T @ layer_2_delta
        dw01 += layer_0.T @ layer_1_delta

    return dw01, dw12, dw23, correct_cnt

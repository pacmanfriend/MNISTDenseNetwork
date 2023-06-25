from __future__ import annotations

import multiprocessing
import numpy as np
import sys
import time
from pathlib import Path

_MIN_SAMPLES_PER_WORKER = 5_000

class DenseNetwork:
    def __init__(self):
        self.input_size = 784
        self.hidden_size = 100
        self.output_size = 10

        self.weights_0_1 = None
        self.weights_1_2 = None
        self.weights_2_3 = None

    def init_weights(self):
        self.weights_0_1 = 0.2 * np.random.random((self.input_size, self.hidden_size)) - 0.1
        self.weights_1_2 = 0.2 * np.random.random((self.hidden_size, self.hidden_size)) - 0.1
        self.weights_2_3 = 0.2 * np.random.random((self.hidden_size, self.output_size)) - 0.1

    def fit(self, x_train, y_train, batch_size, epochs, validation_split=0.1, alpha=0.01, num_workers=None):
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
        train_size = int(len(x_train) - validation_size)

        if num_workers is None:
            num_workers = min(
                multiprocessing.cpu_count(),
                max(1, train_size // _MIN_SAMPLES_PER_WORKER),
            )
        else:
            num_workers = max(1, min(num_workers, multiprocessing.cpu_count()))

        for e in range(epochs):
            correct_cnt = 0
            test_correct_cnt = 0

            perm = np.random.permutation(len(x_train))
            x_shuf = x_train[perm]
            y_shuf = y_train[perm]

            train_images = x_shuf[:train_size]
            train_labels = y_shuf[:train_size]

            validation_images = x_shuf[train_size:]
            validation_labels = y_shuf[train_size:]

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

            if validation_size > 0:
                num_val_batches = int(np.ceil(validation_size / float(batch_size)))
                for i in range(num_val_batches):
                    batch_start, batch_end = (i * batch_size), min((i + 1) * batch_size, validation_size)
                    layer_0 = validation_images[batch_start:batch_end]
                    layer_1 = tanh(np.dot(layer_0, self.weights_0_1))
                    layer_2 = tanh(np.dot(layer_1, self.weights_1_2))
                    layer_3 = softmax(np.dot(layer_2, self.weights_2_3))

                    test_correct_cnt += int(
                        (layer_3.argmax(axis=1) == validation_labels[batch_start:batch_end].argmax(axis=1)).sum()
                    )

            end = time.monotonic() - start

            sys.stdout.write(
                f"I:{e} || Test-Acc:{test_correct_cnt / float(max(1, validation_size))} "
                f"|| Train-Acc:{correct_cnt / float(train_size)} "
                f"|| {end}s\n")

    def predict(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"x must have shape (N, {self.input_size}) or ({self.input_size},), got {x.shape}")
        if self.weights_0_1 is None or self.weights_1_2 is None or self.weights_2_3 is None:
            raise RuntimeError("Model is not initialized. Call fit() or init_weights() first.")

        layer_1 = tanh(np.dot(x, self.weights_0_1))
        layer_2 = tanh(np.dot(layer_1, self.weights_1_2))
        probs = softmax(np.dot(layer_2, self.weights_2_3))
        return probs.argmax(axis=1)

    def evaluate(self, x_test, y_test, batch_size: int = 1024, verbose: bool = True) -> float:
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if x_test.ndim != 2 or x_test.shape[1] != self.input_size:
            raise ValueError(f"x_test must have shape (N, {self.input_size}), got {x_test.shape}")
        if y_test.ndim != 2 or y_test.shape[1] != self.output_size:
            raise ValueError(f"y_test must be one-hot with shape (N, {self.output_size}), got {y_test.shape}")
        if x_test.shape[0] != y_test.shape[0]:
            raise ValueError("x_test and y_test must have the same length")
        if self.weights_0_1 is None or self.weights_1_2 is None or self.weights_2_3 is None:
            raise RuntimeError("Model is not initialized. Call fit() or init_weights() first.")

        correct_cnt = 0
        n = int(x_test.shape[0])
        num_batches = int(np.ceil(n / float(batch_size)))
        for i in range(num_batches):
            batch_start, batch_end = (i * batch_size), min((i + 1) * batch_size, n)
            xb = x_test[batch_start:batch_end]
            yb = y_test[batch_start:batch_end]

            layer_1 = tanh(np.dot(xb, self.weights_0_1))
            layer_2 = tanh(np.dot(layer_1, self.weights_1_2))
            probs = softmax(np.dot(layer_2, self.weights_2_3))

            correct_cnt += int((probs.argmax(axis=1) == yb.argmax(axis=1)).sum())

        acc = correct_cnt / float(max(1, n))
        if verbose:
            sys.stdout.write(f"Test-Acc:{acc}\n")
        return acc

    def save_model(self, path: str | Path) -> None:
        if self.weights_0_1 is None or self.weights_1_2 is None or self.weights_2_3 is None:
            raise RuntimeError("Model is not initialized. Call fit() or init_weights() first.")

        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("h5py is required for HDF5 save/load. Install it via requirements.txt.") from e

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
        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("h5py is required for HDF5 save/load. Install it via requirements.txt.") from e

        path = Path(path)
        with h5py.File(path, "r") as f:
            arch = f["arch"].attrs
            input_size = int(arch["input_size"])
            hidden_size = int(arch["hidden_size"])
            output_size = int(arch["output_size"])

            if (input_size, hidden_size, output_size) != (self.input_size, self.hidden_size, self.output_size):
                raise ValueError(
                    "Model architecture mismatch: "
                    f"file has ({input_size}, {hidden_size}, {output_size}), "
                    f"but instance is ({self.input_size}, {self.hidden_size}, {self.output_size})"
                )

            w = f["weights"]
            self.weights_0_1 = np.asarray(w["weights_0_1"], dtype=np.float64)
            self.weights_1_2 = np.asarray(w["weights_1_2"], dtype=np.float64)
            self.weights_2_3 = np.asarray(w["weights_2_3"], dtype=np.float64)

def tanh(x):
    return np.tanh(x)


def tanh2deriv(x):
    return 1 - (x ** 2)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


def _train_shard(args):
    """Compute accumulated gradient updates for one data shard (runs in a worker process)."""
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
        layer_1 = tanh(np.dot(layer_0, w01))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1_dropout = layer_1 * dropout_mask * 2
        layer_2 = tanh(np.dot(layer_1_dropout, w12))
        layer_3 = softmax(np.dot(layer_2, w23))

        correct_cnt += int(
            (layer_3.argmax(axis=1) == y_shard[batch_start:batch_end].argmax(axis=1)).sum()
        )

        layer_3_delta = (y_shard[batch_start:batch_end] - layer_3) / float(bs)
        layer_2_delta = layer_3_delta.dot(w23.T) * tanh2deriv(layer_2)
        layer_1_delta = layer_2_delta.dot(w12.T) * tanh2deriv(layer_1)
        layer_1_delta *= dropout_mask * 2

        dw23 += layer_2.T.dot(layer_3_delta)
        dw12 += layer_1_dropout.T.dot(layer_2_delta)
        dw01 += layer_0.T.dot(layer_1_delta)

    return dw01, dw12, dw23, correct_cnt

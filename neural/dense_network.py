from __future__ import annotations

import numpy as np
import sys
import time


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

    def fit(self, x_train, y_train, batch_size, epochs, validation_split=0.1, alpha=0.01):
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

            num_batches = int(np.ceil(train_size / float(batch_size)))
            for i in range(num_batches):
                batch_start, batch_end = (i * batch_size), min((i + 1) * batch_size, train_size)
                bs = batch_end - batch_start

                layer_0 = train_images[batch_start:batch_end]
                layer_1 = tanh(np.dot(layer_0, self.weights_0_1))
                dropout_mask = np.random.randint(2, size=layer_1.shape)
                layer_1_dropout = layer_1 * dropout_mask * 2
                layer_2 = tanh(np.dot(layer_1_dropout, self.weights_1_2))
                layer_3 = softmax(np.dot(layer_2, self.weights_2_3))

                correct_cnt += int((layer_3.argmax(axis=1) == train_labels[batch_start:batch_end].argmax(axis=1)).sum())

                layer_3_delta = (train_labels[batch_start:batch_end] - layer_3) / float(bs)
                layer_2_delta = layer_3_delta.dot(self.weights_2_3.T) * tanh2deriv(layer_2)
                layer_1_delta = layer_2_delta.dot(self.weights_1_2.T) * tanh2deriv(layer_1)
                layer_1_delta *= dropout_mask * 2

                self.weights_2_3 += alpha * layer_2.T.dot(layer_3_delta)
                self.weights_1_2 += alpha * layer_1_dropout.T.dot(layer_2_delta)
                self.weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

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

    def load_model(self):
        pass

    def save_model(self):
        pass


relu = lambda x: (x >= 0) * x
relu2deriv = lambda x: x >= 0


def tanh(x):
    return np.tanh(x)


def tanh2deriv(x):
    return 1 - (x ** 2)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

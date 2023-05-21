from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class NumpyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (N, 28, 28), got {x.shape}")
        if y.ndim != 1:
            raise ValueError(f"Expected y with shape (N,), got {y.shape}")

        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx])  # (28, 28), float32
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


class MNISTDenseNet(nn.Module):
    def __init__(self, hidden: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    epochs: int = 5
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 2


def _resolve_device() -> torch.device:
    return torch.device("cpu")


def load_mnist_from_keras():
    # Lazy import so the module can be imported without TF installed.
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    return (x_train, y_train), (x_test, y_test)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(1, total)


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = _resolve_device()

    (x_train, y_train), (x_test, y_test) = load_mnist_from_keras()
    train_ds = NumpyDataset(x_train, y_train)
    test_ds = NumpyDataset(x_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = MNISTDenseNet(hidden=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = int(yb.numel())
            running_loss += float(loss.item()) * bs
            n_samples += bs

        train_loss = running_loss / max(1, n_samples)
        acc = evaluate(model, test_loader, device)
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} test_acc={acc:.4f}")


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="MNIST dense net in PyTorch (data from keras).")
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    args = p.parse_args()
    return TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    train(_parse_args())

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import cpu_count


@dataclass(frozen=True)
class TrainingConfig:
    train_size: int
    epochs: int
    batch_size: int
    alpha: float
    num_workers: int
    validation_split: float = 0.1

    def validate(self, max_train_size: int, max_workers: int | None = None) -> list[str]:
        if max_workers is None:
            max_workers = cpu_count()

        errors: list[str] = []
        if self.train_size <= 0 or self.train_size > max_train_size:
            errors.append(f"Размер данных: от 1 до {max_train_size}")
        if self.epochs <= 0:
            errors.append("Количество эпох: положительное целое число")
        if self.batch_size <= 0:
            errors.append("Размер батча: положительное целое число")
        if self.alpha <= 0:
            errors.append("Alpha: положительное число")
        if not (1 <= self.num_workers <= max_workers):
            errors.append(f"Количество потоков: от 1 до {max_workers}")
        if not (0.0 < self.validation_split < 1.0):
            errors.append("validation_split: число от 0 до 1")
        return errors

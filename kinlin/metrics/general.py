from typing import List, Any, Union
import torch
from core import Metric

class Epoch(Metric):
    def reset(self) -> None:
        self.value: int = 0

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        self.value += 1

class Accuracy(Metric):
    def reset(self) -> None:
        self.value = None

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        accuracy = y_pred.eq(y_true).sum().item() / y_true.nelement()
        self.update(accuracy * 100)

class Loss(Metric):
    def reset(self) -> None:
        self.value: float = 0.0

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:


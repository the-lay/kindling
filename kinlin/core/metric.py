from typing import List, Any, Union
import torch

### Base classes
class Metric:
    def __init__(self):
        self.name = self.__class__.__name__
        self.value = None

    def reset(self) -> None:
        self.value = None

    def set(self, value) -> None:
        self.value = value

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_epoch_finish(self, epoch: int) -> None:
        pass

    def on_batch_start(self, batch: torch.Tensor, batch_id: int, epoch: int) -> None:
        pass

    def on_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int, loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        pass

class RunningMetric(Metric):
    def set(self, value) -> None:
        if self.value is None:
            self.value = value
        else:
            self.value = (self.value + value) / 2

### General metrics
class Epoch(Metric):
    def reset(self) -> None:
        self.value: int = 0

    def on_epoch_finish(self, epoch: int) -> None:
        self.set(self.value + 1)

class Accuracy(RunningMetric):
    def reset(self) -> None:
        self.value: float = 0.0

    def on_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int, loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        accuracy = y_pred.eq(y_true).sum().item() / y_true.nelement()
        self.set(accuracy)

class Loss(RunningMetric):
    def reset(self) -> None:
        self.value: float = 0.0

    def on_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int, loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        self.set(loss)

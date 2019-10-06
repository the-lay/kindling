from typing import List, Any, Union
import torch
from enum import Enum
import time

### Base classes
class Metric:
    def __init__(self):
        self.name = self.__class__.__name__
        self.value = None

        self.setup()

    def setup(self) -> None:
        pass

    def set(self, value) -> None:
        self.value = value

    def print(self) -> str:
        try:
            return f'{self.value:.4f}'
        except TypeError:
            return ''

    def on_start(self) -> None:
        pass

    def on_finish(self) -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_epoch_finish(self, epoch: int) -> None:
        pass

    def on_batch_start(self, batch: torch.Tensor, batch_id: int, epoch: int) -> None:
        pass

    def on_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int,
                        loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        pass

    def on_training_epoch_start(self, epoch: int) -> None:
        pass

    def on_training_epoch_finish(self, epoch: int) -> None:
        pass

    def on_validation_epoch_start(self, epoch: int) -> None:
        pass

    def on_validation_epoch_finish(self, epoch: int) -> None:
        pass

    def on_testing_start(self) -> None:
        pass

    def on_testing_finish(self) -> None:
        pass

    def on_training_batch_start(self, batch: torch.Tensor, batch_id: int, epoch: int) -> None:
        pass

    def on_training_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int,
                                 loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        pass

    def on_validation_batch_start(self, batch: torch.Tensor, batch_id: int, epoch: int) -> None:
        pass

    def on_validation_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int,
                                   loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        pass

    def on_testing_batch_start(self, batch: torch.Tensor, batch_id: int) -> None:
        pass

    def on_testing_batch_finish(self, batch: torch.Tensor, batch_id: int,
                                loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        pass

class RunningEpochMetric(Metric):
    def __init__(self):
        self.started: bool = False
        super(RunningEpochMetric, self).__init__()

    def on_epoch_start(self, epoch: int) -> None:
        self.started: bool = False
        self.value = None

    def set(self, value) -> None:
        if self.started:
            self.value = (self.value + value) / 2
        else:
            self.value = value
            self.started = True

### General metrics
class Epoch(Metric):
    def setup(self) -> None:
        self.value: int = 0

    def on_training_epoch_finish(self, epoch: int) -> None:
        self.set(self.value + 1)

class Accuracy(RunningEpochMetric):
    def on_epoch_start(self, epoch: int) -> None:
        super(RunningEpochMetric, self).on_epoch_start(epoch)
        self.value: float = 0.0

    def on_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int,
                        loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:

        accuracy = y_pred.eq(y_true).sum().item() / y_true.nelement()
        self.set(accuracy)

class Loss(RunningEpochMetric):
    def on_epoch_start(self, epoch: int) -> None:
        super(RunningEpochMetric, self).on_epoch_start(epoch)
        self.value: float = 0.0

    def on_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int,
                        loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        self.set(loss)

class EpochTimer(Metric):
    def setup(self) -> None:
        self.name = 'Time'

    def on_epoch_start(self, epoch: int) -> None:
        self.value = time.time()

    def on_epoch_finish(self, epoch: int) -> None:
        self.value = self.value - time.time()

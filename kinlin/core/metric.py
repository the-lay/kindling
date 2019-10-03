from typing import List, Any, Union
import torch

class Metric:
    def __init__(self):
        self.name = self.__class__.__name__
        self.value = None

    def on_epoch_start(self, epoch: int):
        pass

    def on_epoch_finish(self, epoch: int):
        pass

    def on_batch_start(self, batch: torch.Tensor, batch_id: int, epoch: int):
        pass

    def on_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int, y_pred: torch.Tensor, y_true: torch.Tensor):
        pass

class Epoch(Metric):
    def __init__(self):
        super(Epoch, self).__init__()
        self.value = 0

    def on_epoch_finish(self, epoch: int):
        self.value += 1









    # represents a running metric
    def __init__(self, name: str = None):
        self.name = name if name or self.__class__.__name__
        self.value = None
        self.reset()

    def reset(self) -> None:
        # reset method is called in the beginning of every epoch
        raise NotImplementedError

    def update(self, value) -> None:
        if self.value is None:
            self.value = value
        else:
            self.value = (self.value + value) / 2

    def set(self, value) -> None:
        self.value = value

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        raise NotImplementedError

class AutoMetric(Metric):
    pass

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


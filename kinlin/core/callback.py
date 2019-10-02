import torch
from typing import Union, List

class Callback:
    def __init__(self):
        pass

### Events
    def on_epoch_start(self, epoch: int):
        pass

    def on_epoch_finish(self, epoch: int):
        pass

    def on_training_epoch_start(self, epoch: int):
        pass

    def on_training_epoch_finish(self, epoch: int):
        pass

    def on_validation_epoch_start(self, epoch: int):
        pass

    def on_validation_epoch_finish(self, epoch: int):
        pass

    def on_training_batch_start(self, batch: torch.Tensor, batch_id: int, epoch: int, validation: bool):
        pass

    def on_training_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int, validation: bool):
        pass

    def on_validation_batch_start(self, batch: torch.Tensor, batch_id: int, epoch: int, validation: bool):
        pass

    def on_validation_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int, validation: bool):
        pass

import torch

class Callback:
    def __init__(self):
        pass

### Events
    def on_start(self) -> None:
        pass

    def on_finish(self) -> None:
        pass

    def on_epoch_start(self, epoch: int, model: 'Model') -> None:
        pass

    def on_epoch_finish(self, epoch: int, model: 'Model') -> None:
        pass

    def on_batch_start(self, batch, batch_id: int, epoch: int) -> None:
        pass

    def on_batch_finish(self, batch, batch_id: int, epoch: int,
                        loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        pass

    def on_training_epoch_start(self, epoch: int, model: 'Model') -> None:
        self.on_epoch_start(epoch, model)

    def on_training_epoch_finish(self, epoch: int, model: 'Model') -> None:
        self.on_epoch_finish(epoch, model)

    def on_validation_epoch_start(self, epoch: int, model: 'Model') -> None:
        self.on_epoch_start(epoch, model)

    def on_validation_epoch_finish(self, epoch: int, model: 'Model') -> None:
        self.on_epoch_finish(epoch, model)

    def on_testing_start(self) -> None:
        self.on_epoch_start(-1, None)

    def on_testing_finish(self) -> None:
        self.on_epoch_finish(-1, None)

    def on_training_batch_start(self, batch, batch_id: int, epoch: int) -> None:
        self.on_batch_start(batch, batch_id, epoch)

    def on_training_batch_finish(self, batch, batch_id: int, epoch: int,
                                 loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        self.on_batch_finish(batch, batch_id, epoch, loss, y_pred, y_true)

    def on_validation_batch_start(self, batch, batch_id: int, epoch: int) -> None:
        self.on_batch_start(batch, batch_id, epoch)

    def on_validation_batch_finish(self, batch, batch_id: int, epoch: int,
                                   loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        self.on_batch_finish(batch, batch_id, epoch, loss, y_pred, y_true)

    def on_testing_batch_start(self, batch, batch_id: int) -> None:
        self.on_batch_start(batch, batch_id, -1)

    def on_testing_batch_finish(self, batch, batch_id: int,
                                loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        self.on_batch_finish(batch, batch_id, -1, loss, y_pred, y_true)

import torch
import torch.nn as nn
import torch.optim as optim
import kinlin as k


class Dataset(k.Dataset):
    pass

class Unet:
    pass

class Baseline(k.Model):
    def training_fn(self, batch: torch.Tensor, batch_id: int, epoch: int) -> torch.Tensor:
        # your custom training here, this function being run for every
        pass

    def validation_fn(self, batch: torch.Tensor, batch_id: int, epoch: int):
        pass

    def on_validation_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int,
                                   loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        # plot one batch for visualization
        if batch_id == 0:
            pass
        super(Baseline, self).on_validation_batch_finish(batch, batch_id, epoch, loss, y_pred, y_true)

# Training parameters
loss = [nn.MSELoss(), nn.MSELoss()]
loss_weights = [1e4, 1]

# Metrics and callbacks
metrics = []
callbacks = []

# Training
network = Unet()
optimizer = optim.Adam(network.parameters(), lr=0.01)
model = Baseline(network, 'Baseline Unet', metrics)
training = k.SupervisedTraining(model, dataset, optimizer, callbacks=callbacks)

training(n_epochs=10)
training.test()

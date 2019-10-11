import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from typing import Union, List, Callable, Tuple
from pathlib import Path
import kinlin as k

class Baseline(k.Model):
    def __init__(self, network: torch.nn.Module, name: str = None, metrics: List[k.Metric] = None):
        super(Baseline, self).__init__(network, name, metrics)
        self.loss = k.losses.TverskyLoss(classes=self.network.n_classes)

    def training_fn(self, batch, batch_id: int, epoch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction = batch['reconstruction'].to(device='cuda')
        classmask = batch['classmask'].to(device='cuda')
        net_output = self.network(reconstruction)
        loss = self.loss(net_output, classmask)
        return loss, net_output, classmask

    def validation_fn(self, batch, batch_id: int, epoch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction = batch['reconstruction'].to(device='cuda')
        classmask = batch['classmask'].to(device='cuda')
        net_output = self.network(reconstruction)
        loss = self.loss(net_output, classmask)
        return loss, net_output, classmask

    def on_validation_batch_finish(self, batch, batch_id: int, epoch: int,
                                   loss: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        super(Baseline, self).on_validation_batch_finish(batch, batch_id, epoch, loss, y_pred, y_true)
        # plot one batch for visualization
        if batch_id == 0:
            # TODO plot a batch
            pass

# Network setup
n_classes = 13
unet_filters = [16, 32, 64, 128]

# Training
network = k.networks.UNet(unet_filters, in_channels=1, n_classes=n_classes)
optimizer = optim.Adam(network.parameters(), lr=0.01)
model = Baseline(network, 'Baseline Unet', metrics=[k.metrics.ClasswiseDiceCoefficient(n_classes),
                                                    k.metrics.ClasswiseJaccardIndex(n_classes),
                                                    k.metrics.DiceCoefficient(n_classes),
                                                    k.metrics.JaccardIndex(n_classes)])
dataset = k.datasets.SHREC(Path(r'C:\Users\ilja-work-laptop\Desktop\data\shrec'), subtomo_size=64, augmentation=True)
training = k.SupervisedTraining(model, dataset, optimizer, callbacks=[k.callbacks.TensorboardCallback(log_dir='tensorboard'),
                                                                      k.callbacks.SpreadsheetCallback(Path('.')),
                                                                      ])

training(n_epochs=10)
training.test()

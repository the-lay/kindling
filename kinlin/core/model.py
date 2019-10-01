import torch
from typing import Union

class Model:

    def __init__(self, network: torch.nn.Module, name: str = 'Untitled model', ):
        # network
        self.network: torch.nn.Module = network
        self.name: str = name
        # training_fn
        # validation_fn
        # losses and weights
        # metrics
        # hyper parameters

        pass

    def get_param_count(self, readable_str: bool = False):
        params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        if readable_str:
            for unit in ['', 'K', 'M', 'B']:
                if params < 1000:
                    break
                params /= 1000
            return f'{params:.3f}{unit}'
        else:
            return params

    def on_epoch_start(self, epoch: int):
        pass

    def on_training_start(self, epoch: int):
        pass

    def on_validation_start(self, epoch: int):
        pass

    def on_batch_start(self, batch: torch.Tensor, batch_id: int, epoch: int, validation: bool):
        pass

    # must return loss to backprop
    def training_fn(self, batch: torch.Tensor, batch_id: int, epoch: int) -> torch.Tensor:
        raise NotImplementedError

    def backprop_fn(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def validation_fn(self, batch: torch.Tensor, batch_id: int, epoch: int):
        raise NotImplementedError

    def on_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int, validation: bool):
        pass

    def on_training_finish(self, epoch:int):
        pass

    def on_validation_finish(self, epoch: int):
        pass

    def on_epoch_finish(self, epoch: int):
        # TODO callbacks
        # TODO metrics
        pass

    def __str__(self):
        return 'TODO'  # TODO

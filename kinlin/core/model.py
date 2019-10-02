import torch
from typing import Union, List, Callable
from .utils import generate_random_id
from .metric import Metric

class Model:

    def __init__(self, network: torch.nn.Module, metrics: List[Metric], name: str = None, ):
        # properties
        self.id: str = generate_random_id()
        self.network: torch.nn.Module = network
        self.metrics: List[Metric] = metrics
        self.name: str = name if name else self.network.__class__.__name__

        # # Support one or multiple loss functions
        # self.loss_fn: Union[List[Callable], Callable]
        # self.loss_weights: Union[List[float], float]

### Methods called by training runners

    def training_fn(self, batch: torch.Tensor, batch_id: int, epoch: int) -> torch.Tensor:
        # must return loss to backprop
        raise NotImplementedError

    def backprop_fn(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def validation_fn(self, batch: torch.Tensor, batch_id: int, epoch: int):
        raise NotImplementedError

### General methods
    def nparams(self, readable_units: bool = False) -> Union[int, str]:
        params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        if readable_units:
            for unit in ['', 'K', 'M', 'B']:
                if params < 1000:
                    break
                params /= 1000
            return f'{params:.3f}{unit}'
        else:
            return params

    def __repr__(self):
        summary = f'Model "{self.name}":\n' \
                  f'\tParameters: {self.nparams(readable_units=True)}\n' \
                  f'\tTODO\n'

        # metrics
        summary += '\tMetrics:\n'
        for m in self.metrics:
            summary += f'\t\t{m.name}: \n' # TODO

        return summary

    def print_summary(self):
        print(repr(self))

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

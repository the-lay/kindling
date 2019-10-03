import torch
from typing import Union, List, Callable, Tuple
from .utils import generate_random_id
from .metric import Metric, Epoch, Accuracy, Loss

class Model:

    def __init__(self, network: torch.nn.Module, epoch_metrics: List[Metric] = None,  training_metrics: List[Metric] = None,
                 validation_metrics: List[Metric] = None, name: str = None):
        # properties
        self.id: str = generate_random_id()
        self.network: torch.nn.Module = network
        self.metrics = {
            'general': epoch_metrics if epoch_metrics else [],
            'training': training_metrics if training_metrics else [],
            'validation': validation_metrics if validation_metrics else []
        }
        self.name: str = name if name else self.network.__class__.__name__

        # set default metrics
        self.reset_metrics()

### Methods called by training strategies
    def training_fn(self, batch: torch.Tensor, batch_id: int, epoch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method should describe training process on one batch and return loss, y_pred and y_true
        :param batch:
        :param batch_id:
        :param epoch:
        :return: loss, y_pred, y_true
        """
        # must return loss to backprop
        raise NotImplementedError

    def backprop_fn(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def validation_fn(self, batch: torch.Tensor, batch_id: int, epoch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param batch:
        :param batch_id:
        :param epoch:
        :return: loss, y_pred, y_true
        """
        raise NotImplementedError

    def testing_fn(self, batch: torch.Tensor, batch_id: int, epoch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

### General methods - not really meant to be overriden
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
                  f'\tParameters: {self.nparams(readable_units=True)}\n'

        # metrics
        summary += '\tGeneral metrics:\n'
        for m in self.metrics['general']:
            summary += f'\t\t{m.name}: {m.value}\n'

        if len(self.metrics['training']) > 0:
            summary += '\tTraining metrics:\n'
            for m in self.metrics['training']:
                summary += f'\t\t{m.name}: {m.value}\n'

        if len(self.metrics['validation']) > 0:
            summary += '\tValidation metrics:\n'
            for m in self.metrics['validation']:
                summary += f'\t\t{m.name}: {m.value}\n'

        return summary

    def print_summary(self) -> None:
        print(repr(self))

### Metrics handling - not really meant to be overriden
    def reset_metrics(self, stage: str = 'all'):
        if stage not in ['all'] or stage not in self.metrics:
            raise ValueError('No such stage')

        if stage == 'all':
            for s in self.metrics:
                self.reset_metrics(s)
        else:
            for m in self.metrics[stage]:
                m.reset()

        # default metrics
        self.metrics['general'].append(Epoch())
        self.metrics['training'].append(Loss())
        self.metrics['training'].append(Accuracy())
        self.metrics['validation'].append(Loss())
        self.metrics['validation'].append(Accuracy())

    def update_validation_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, loss: torch.Tensor):
        for m in self.metrics['validation']:
            m(y_pred, y_true)

    def update_training_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, loss: torch.Tensor):
        for m in self.metrics['training']:
            m(y_pred, y_true)

    def update_general_metrics(self, epoch: int):
        for m in self.metrics['general']:
            m(epoch)

### Events - fully meant to be overriden
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

    def on_training_batch_start(self, batch: torch.Tensor, batch_id: int, epoch: int):
        pass

    def on_training_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int, y_pred: torch.Tensor, y_true: torch.Tensor):
        pass

    def on_validation_batch_start(self, batch: torch.Tensor, batch_id: int, epoch: int):
        pass

    def on_validation_batch_finish(self, batch: torch.Tensor, batch_id: int, epoch: int, y_pred: torch.Tensor, y_true: torch.Tensor):
        pass

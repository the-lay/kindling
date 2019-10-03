import torch
import torch.nn as nn

from enum import Enum
from typing import List, Callable, Any
from tqdm import tqdm

from .model import Model
from .dataset import Dataset
from .experiment import Experiment
from .callback import Callback

class TrainingEvents(Enum):
    EPOCH_START = 'on_epoch_start',
    EPOCH_FINISH = 'on_epoch_finish',
    TRAINING_EPOCH_START = 'on_training_epoch_start',
    TRAINING_EPOCH_FINISH = 'on_training_epoch_finish',
    TRAINING_BATCH_START = 'on_training_batch_start',
    TRAINING_BATCH_FINISH = 'on_training_batch_finish',
    VALIDATION_EPOCH_START = 'on_validation_batch_start',
    VALIDATION_EPOCH_FINISH = 'on_validation_batch_finish',
    VALIDATION_BATCH_START = 'on_validation_batch_start',
    VALIDATION_BATCH_FINISH = 'on_validation_batch_finish'

# TODO later make a TrainingStrategy parent class to extend easily
class SupervisedTraining:
    def __init__(self, model: Model, dataset: Dataset, optimizer: torch.optim.Optimizer, experiment: Experiment, callbacks: List[Callback] = None):

        # properties
        self.model: Model = model
        self.dataset: Dataset = dataset
        self.optimizer: torch.optim.Optimizer = optimizer
        self.experiment: Experiment = experiment
        self.callbacks: List[Callback] = callbacks

        # parallelize network depending on experiment settings
        if len(self.experiment.devices) > 1:
            self.network = nn.DataParallel(self.model.network, device_ids=self.experiment.devices)
        else:
            self.network = self.model.network

        # event handlers
        self.handlers = {k: [] for k in TrainingEvents}

        # register model events
        for event in TrainingEvents:
            self.on_event(event, getattr(self.model, event.value))

        # register callback events
        for callback in self.callbacks:
            self.register_callback(callback)

        # register metric events
        # TODO there must be a prettier way
        for metric in self.model.metrics['general']:
            self.on_event(TrainingEvents.EPOCH_START, getattr(metric, TrainingEvents.EPOCH_START.value))
            self.on_event(TrainingEvents.EPOCH_FINISH, getattr(metric, TrainingEvents.EPOCH_FINISH.value))

        for metric in self.model.metrics['training']:
            self.on_event(TrainingEvents.EPOCH_START, getattr(metric, TrainingEvents.EPOCH_START.value))
            self.on_event(TrainingEvents.EPOCH_FINISH, getattr(metric, TrainingEvents.EPOCH_FINISH.value))
            self.on_event(TrainingEvents.TRAINING_BATCH_START, getattr(metric, 'on_batch_start'))
            self.on_event(TrainingEvents.TRAINING_BATCH_FINISH, getattr(metric, 'on_batch_finish'))

        for metric in self.model.metrics['validation']:
            self.on_event(TrainingEvents.EPOCH_START, getattr(metric, TrainingEvents.EPOCH_START.value))
            self.on_event(TrainingEvents.EPOCH_FINISH, getattr(metric, TrainingEvents.EPOCH_FINISH.value))
            self.on_event(TrainingEvents.VALIDATION_BATCH_START, getattr(metric, 'on_batch_start'))
            self.on_event(TrainingEvents.VALIDATION_BATCH_FINISH, getattr(metric, 'on_batch_finish'))

    def on_event(self, event: TrainingEvents, handler: Callable):
        self.handlers[event].append(handler)

    def register_callback(self, callback: Callback):
        for event in TrainingEvents:
            self.on_event(event, getattr(callback, event.value))

    def emit(self, event: TrainingEvents, *args, **kwargs):
        for handler in self.handlers[event]:
            handler(args, kwargs)

    # TODO more event methods, for example unregister

    def __call__(self, n_epochs: int = 1, validation: bool = True, verbose: bool = True):

        if verbose:
            print(f'Training{" and validating" if validation else ""} for {n_epochs} {"epochs" if n_epochs > 1 else "epoch"}')
            self.model.print_summary()
            self.dataset.print_summary()
            print(f'Optimizer "{self.optimizer.__class__.__name__}":\n'
                  f'\tLearning rate: {self.optimizer.param_groups[0]["lr"]}')
            print(f'Callbacks: {[callback.__class__.__name__ for callback in self.callbacks]}')

        for epoch in range(n_epochs):
            print(f'\nEpoch {epoch}:')
            self.emit(TrainingEvents.EPOCH_START, epoch)

            # Training
            self.model.network.train()
            self.emit(TrainingEvents.TRAINING_EPOCH_START, epoch)

            with tqdm(self.dataset.training_dataloader(), desc='Training', unit='batch') as t:
                for batch_id, batch in enumerate(t):
                    self.emit(TrainingEvents.TRAINING_BATCH_START, batch, batch_id, epoch)
                    loss, y_pred, y_true = self.model.training_fn(batch, batch_id, epoch)
                    self.model.backprop_fn(loss, self.optimizer)
                    self.emit(TrainingEvents.TRAINING_BATCH_FINISH, batch, batch_id, epoch, loss, y_pred, y_true)

                    # update progress bar
                    t.set_postfix({k.__class__.__name__: k.value for k in self.model.metrics['training']})

            self.emit(TrainingEvents.TRAINING_EPOCH_FINISH, epoch)

            # Validation
            if validation:
                self.model.network.eval()
                self.model.network.train(False)
                self.emit(TrainingEvents.VALIDATION_EPOCH_START, epoch)

                with torch.no_grad():
                    with tqdm(self.dataset.validation_dataloader(), desc='Validation', unit='batch') as t:
                        for batch_id, batch in enumerate(t):
                            self.emit(TrainingEvents.VALIDATION_BATCH_START, batch, batch_id, epoch)
                            loss, y_pred, y_true = self.model.validation_fn(batch, batch_id, epoch)
                            self.emit(TrainingEvents.VALIDATION_BATCH_FINISH, batch, batch_id, epoch, loss, y_pred, y_true)

                            # update progress bar
                            t.set_postfix({k.__class__.__name__: k.value for k in self.model['validation']})

                self.emit(TrainingEvents.VALIDATION_EPOCH_FINISH, epoch)

            self.emit(TrainingEvents.EPOCH_FINISH, epoch)

    # TODO
    def test(self):
        pass
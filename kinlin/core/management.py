from typing import List, Union
from pathlib import Path
from tqdm import tqdm
from .utils import *
from .model import Model


class Project:
    def __init__(self, name: str = 'Untitled Project', notes: str = '', path: Union[Path, str] = None):

        # general properties
        self.name: str = name
        self.notes: str = notes
        if isinstance(path, str):
            path = Path(path)
        self.path: Path = path

        # experiments
        self.experiments: List[Experiment] = []

    def new_experiment(self):

        new_exp = Experiment()
        # setup experiment?
        self.experiments += new_exp
        pass

    def continue_experiment(self):
        pass

    def open_experiment_folder(self):
        open_folder(self.path)


class Experiment:

    def __init__(self, project: Project, name: str = 'Untitled Experiment', notes: str = '', seed: int = None,
                 devices: List[int, str, torch.device] = None, tensorboard: bool = True):

        # general properties
        self.id: str = generate_random_id()
        self.project: Project = project
        self.name: str = name
        self.notes: str = notes
        self.path: Path = project.path / 'experiments' / str(self.id)
        self.path.mkdir(parents=True)  # TODO: this should never happen, but it can raise FileExistError

        # save git commit id at the moment of running the experiment
        # TODO bonus points: make .patch file to see what was changed since last commit!
        # TODO bonus bonus points: method to open github website at that revision?
        self.code = get_current_git_head()

        # determinism
        # cudnn_deterministic is slower, but gives almaximum possible determinism
        if seed is None:
            self.seed: int = generate_random_seed()
        else:
            self.seed: int = seed
        set_seed(self.seed, cudnn_deterministic=False)

        # devices
        if devices is None:
            self.devices = [0]
        else:
            self.devices = devices
        # during model initialization torch spills some memory to the GPU0 (not necessarily GPU that we will use)
        torch.cuda.set_device(self.devices[0])

        # tensorboard
        self.tensorboard = tensorboard
        # TODO figure out how (where) to integrate tensorboard - maybe a separate new process to run tensorboard + trainer

        # history
        self.history = []
        # TODO history saving
        # some kind of history rows, to be able to store actions and their results
        # and to store metrics history?

    def new_trainer(self, ):
        pass

    # TODO decorator for functions that should log entry and exit?
    def log(self, action: str,):
        pass


class Trainer:
    # Trainer connects Dataset, Model, Loss, pytorch optimizer, Callbacks (like scheduler, checkpointer etc.) and Metrics
    # training and validation
    def __init__(self, model: Model, dataset: Dataset, optimizer: Any[torch.optim], experiment: Experiment):
        self.model: Model = model
        self.dataset: Dataset = dataset
        self.optimizer: Any[torch.optim] = optimizer
        self.experiment: Experiment = experiment

    def fit(self, n_epochs: int = 1, validation: bool = True, callbacks: List[Callback] = None, verbose: bool = True):

        if verbose:
            print(f'Training{" and validating" if validation else ""} for {n_epochs} {"epochs" if n_epochs > 1 else "epoch"}')
            print(f'Network: {self.model.name}, {self.model.get_param_count(readable_str=True)} parameters')
            print(f'Loss: TODO')
            print(f'Metrics: TODO')
            print(f'Dataset: TODO')
            print(f'Optimizer: {self.optimizer.__class__.__name__}, lr: {self.optimizer.param_groups[0]["lr"]}')
            print(f'Callbacks: TODO')

        for epoch in range(n_epochs):
            print(f'\nEpoch {epoch}:')
            self.model.on_epoch_start(epoch)

            # Training
            self.model.network.train()
            self.model.on_training_start(epoch)

            with tqdm(self.dataset.training_dataloader(), desc='Training', unit='batch') as t:
                for batch_id, batch in enumerate(t):
                    self.model.on_batch_start(batch, batch_id, epoch, validation=False)
                    loss = self.model.training_fn(batch, batch_id, epoch)
                    self.model.backprop_fn(loss, self.optimizer)
                    self.model.on_batch_finish(batch, batch_id, epoch, validation=False)

                    if verbose:
                        # TODO put metrics here
                        t.set_postfix()

            self.model.on_training_finish(epoch)

            if validation:
                # Validation
                self.model.network.eval()
                self.model.network.train(False)

                with torch.no_grad():
                    self.model.on_validation_start(epoch)

                    with tqdm(self.dataset.validation_dataloader(), desc='Validation', unit='batch') as t:
                        for batch_id, batch in enumerate(t):
                            self.model.on_batch_start(batch, batch_id, epoch, validation=True)
                            self.model.validation_fn(batch, batch_id, epoch)
                            self.model.on_batch_finish(batch, batch_id, epoch, validation=True)

                            if verbose:
                                # TODO put metrics here
                                t.set_postfix()
                self.model.on_validation_finish(epoch)

            self.model.on_epoch_finish(epoch)




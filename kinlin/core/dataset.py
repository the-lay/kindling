# generic dataset classes
# MrcDataset
# EmDataset ?
# ImageDataset
from typing import List
from torch.utils.data import DataLoader

class Dataset:

    def __init__(self, split: List[float], batch_sizes: List[int], augmentation: List):
        self.split: List[float] = split
        self.batch_sizes: List[int] = batch_sizes
        pass

    def __aug__(self, batch):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError




    def training_dataloader(self) -> DataLoader:
        pass

    def validation_dataloader(self) -> DataLoader:
        pass

    def test_dataloader(self) -> DataLoader:
        pass

    def __repr__(self):
        return f'Dataset "{self.__class__.__name__}":\n' \
               f'\tSplit: {self.split}\n' \
               f'\tBatch sizes: {self.batch_sizes}\n' \
               f'\tTraining size: {len(self.training_dataloader())} ({len(self.training_dataloader()) * self.batch_sizes[0]})\n' \
               f'\tValidation size: {len(self.validation_dataloader())} ({len(self.validation_dataloader()) * self.batch_sizes[1]})\n' \
               f'\tTest size: {len(self.validation_dataloader())} ({len(self.validation_dataloader()) * self.batch_sizes[1]})\n'

    def print_summary(self):
        print(repr(self))

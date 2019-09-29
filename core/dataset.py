# generic dataset classes
# MrcDataset
# EmDataset ?
# ImageDataset
from typing import List

class Dataset:

    def __init__(self, split: List[float], augmentation: List):
        pass

    def __aug__(self, batch):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError




    def training_dataloader(self):
        pass

    def validation_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def __str__(self):
        return 'TODO'  # TODO

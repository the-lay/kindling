import random
import sys
import string
import subprocess
import platform
from typing import Union
from pathlib import Path
import numpy as np
import torch


def generate_random_id() -> str:
    # https://stackoverflow.com/questions/13484726/safe-enough-8-character-short-unique-random-string
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choices(alphabet, k=8))

def generate_random_seed() -> int:
    return random.randint(-sys.maxsize - 1, sys.maxsize)

def set_seed(seed: int, cudnn_deterministic: bool = False) -> None:
    # https://pytorch.org/docs/master/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cudnn_deterministic and torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def fast_histogram(true: torch.Tensor, pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    # https://github.com/kevinzakka/pytorch-goodies/blob/master/metrics.py

    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist

def open_folder(path: Union[Path, str]):
    if platform.system() == 'Windows':
        if isinstance(path, Path):
            subprocess.Popen(f'explorer "{str(path.absolute())}"')
        elif isinstance(path, str):
            subprocess.Popen(f'explorer "{path}"')
    else:
        raise NotImplementedError  # TODO

def open_file(path: Union[Path, str]):
    if platform.system() == 'Windows':
        if isinstance(path, Path):
            subprocess.Popen(f'explorer /select,"{str(path.absolute())}"')
        elif isinstance(path, str):
            subprocess.Popen(f'explorer /select,"{path}"')
    else:
        raise NotImplementedError  # TODO

def get_current_git_head():
    # https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

def has_nan(x: torch.Tensor) -> bool:
    return torch.isnan(x).any().item()

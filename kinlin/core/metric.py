from typing import List, Any
import torch

class Metric:
    def __init__(self):
        self.value = 0
        self.reset()

    def reset(self) -> None:
        pass

    def update(self, output):
        pass

class Metric:
    def __init__(self):
        self.history = []

class AverageMetric:
    def __init__(self):
        self.average = 0


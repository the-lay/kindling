from ..core import Callback
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

class History(Callback):
    def __init__(self):
        super(History, self).__init__()
        self.history = []

    def on_epoch_finish(self, epoch: int, model: 'Model') -> None:
        super(History, self).on_epoch_finish(epoch, model)
        epoch_metrics = {m.name: m.value for m in model.metrics}
        self.history.append(epoch_metrics)


class TensorboardCallback(Callback):
    def __init__(self, log_dir: str = None, flush_secs: int = 30):
        super(TensorboardCallback, self).__init__()
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ValueError('No Tensorboard installed, can\'t use TensorboardCallback')

        self.tb = SummaryWriter(flush_secs=flush_secs, log_dir=log_dir)

    def on_validation_epoch_finish(self, epoch: int, model: 'Model') -> None:
        super(TensorboardCallback, self).on_validation_epoch_finish(epoch, model)
        for m in model.metrics:
            if isinstance(m.value, list) or isinstance(m.value, np.ndarray):
                self.tb.add_histogram('val/' + m.name, np.array(m.value), epoch)
            elif isinstance(m.value, str):
                continue
            else:
                self.tb.add_scalar(m.name, m.value, epoch)
        self.tb.flush()

    def on_training_epoch_finish(self, epoch: int, model: 'Model') -> None:
        super(TensorboardCallback, self).on_training_epoch_finish(epoch, model)
        for m in model.metrics:
            if isinstance(m.value, list) or isinstance(m.value, np.ndarray):
                self.tb.add_histogram('train/' + m.name, np.array(m.value), epoch)
            elif isinstance(m.value, str):
                continue
            else:
                self.tb.add_scalar(m.name, m.value, epoch)
        self.tb.flush()


class ExcelMetricsLogger(Callback):
    def __init__(self, path: Path, file_name: str = 'history.xlsx', float_format: str = '%.3f',
                 freeze_panes: Tuple = (1, 1), sheet: str = 'Sheet1'):
        super(ExcelMetricsLogger, self).__init__()

        # dataframe to store metrics
        self.history = pd.DataFrame()

        # save parameters
        self.sheet_name = sheet
        self.float_format = float_format
        self.freeze_panes = freeze_panes
        self.file_name = file_name
        self.path = path

    def on_epoch_finish(self, epoch: int, model: 'Model') -> None:
        super(ExcelMetricsLogger, self).on_epoch_finish(epoch, model)

        epoch_metrics = {m.name: m.value for m in model.metrics}
        self.history = self.history.append(epoch_metrics,  ignore_index=True)

        self.history.to_excel(self.path / self.file_name, sheet_name=self.sheet_name, float_format=self.float_format,
                              freeze_panes=self.freeze_panes)

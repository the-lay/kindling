from ..core import Callback
import pandas as pd
import numpy as np

class TensorboardCallback(Callback):
    def __init__(self):
        super(TensorboardCallback, self).__init__()
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ValueError('No Tensorboard installed, can\'t use TensorboardCallback')

        self.tb = SummaryWriter(flush_secs=30)

    def on_epoch_finish(self, epoch: int, model: 'Model') -> None:
        super(TensorboardCallback, self).on_epoch_finish(epoch, model)
        for m in model.metrics:
            if isinstance(m.value, list):
                self.tb.add_histogram(m.name, np.array(m.value), epoch)
            else:
                self.tb.add_scalar(m.name, m.value, epoch)
        self.tb.flush()

# class SpreadsheetCallback(Callback):
#     pd.DataFrame(self.history).to_excel(self.checkpoint_folder / 'history.xlsx', float_format='%.8f')

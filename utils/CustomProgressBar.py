import os
import math
from tqdm import tqdm
from typing import Any, Dict, Optional, Union
from pytorch_lightning.callbacks import RichProgressBar, TQDMProgressBar

def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    """The tqdm doesn't support inf/nan values.
    We have to convert it to None.
    """
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return x

class LitProgressBar(TQDMProgressBar):
    
    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        total_train_batches = self.total_train_batches
        self.main_progress_bar.total = convert_inf(total_train_batches)
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch}")
        
class CustomRichProgressBar(RichProgressBar):
    
    # def on_train_epoch_start(self, trainer, pl_module):
    #     total_train_batches = self.total_train_batches
    #     total_val_batches = self.total_val_batches
    #     if total_train_batches != float("inf"):
    #         # val can be checked multiple times per epoch
    #         val_checks_per_epoch = total_train_batches // trainer.val_check_batch
    #         total_val_batches = total_val_batches * val_checks_per_epoch

    #     # total_batches = total_train_batches + total_val_batches

    #     train_description = self._get_train_description(trainer.current_epoch)
    #     if self.main_progress_bar_id is not None and self._leave:
    #         self._stop_progress()
    #         self._init_progress(trainer)
    #     if self.main_progress_bar_id is None:
    #         self.main_progress_bar_id = self._add_task(total_train_batches, train_description)
    #     elif self.progress is not None:
    #         self.progress.reset(
    #             self.main_progress_bar_id, total=total_train_batches, description=train_description, visible=True
    #         )
    #     self.refresh()
        
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
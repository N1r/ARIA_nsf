import os
import pathlib
from typing import Any, Optional, Sequence

import torchaudio
from lightning import LightningModule, Trainer
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.callbacks import BasePredictionWriter, Callback

from .autoencoder import VoiceAutoEncoder
from .progress import TrainingProgressPrinter

__all__ = ["MyConfigCallback", "MyPredictionWriter", "TrainingProgressPrinter"]


class MyPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir):
        super().__init__("batch")
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=False, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: VoiceAutoEncoder,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        *_, rel_path = batch
        pred, _ = prediction
        sr = pl_module.sample_rate
        out_path = self.output_dir / rel_path[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            out_path,
            pred.as_tensor().cpu(),
            sample_rate=sr,
            encoding="PCM_S",
            bits_per_sample=16,
        )


class MyConfigCallback(Callback):
    def __init__(
        self,
        parser,
        config,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
        self.multifile = multifile
        self.already_saved = False

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.already_saved:
            return
        if trainer.is_global_zero:
            if trainer.logger is not None:
                trainer.logger.log_hyperparams(self.config.as_dict())
            self.already_saved = True
        self.already_saved = trainer.strategy.broadcast(self.already_saved)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.already_saved:
            return

        log_dir = pathlib.Path(trainer.checkpoint_callback.dirpath).parent
        config_path = os.path.join(str(log_dir), self.config_filename)
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        if trainer.is_global_zero:
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )
            self.already_saved = True
            if trainer.logger is not None:
                trainer.logger.log_hyperparams(self.config.as_dict())
        self.already_saved = trainer.strategy.broadcast(self.already_saved)

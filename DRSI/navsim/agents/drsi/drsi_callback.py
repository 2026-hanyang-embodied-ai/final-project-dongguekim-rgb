from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch

from navsim.agents.drsi.drsi_config import DRSIConfig


class DRSICallback(pl.Callback):
    """Callback for DRSI during training."""

    def __init__(
        self,
        config: DRSIConfig,
    ) -> None:

        self._config = config


    def on_validation_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_validation_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_test_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_test_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_train_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_train_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        from navsim.agents.drsi.drsi_agent import DRSIAgent

        assert isinstance(lightning_module.agent, DRSIAgent), f"Expected DRSIAgent, got {type(lightning_module.agent)}"
        if self._config.build_vocab_cache:
            if lightning_module.current_epoch % self._config.vocab_cache_build_freq == 0:
                lightning_module.agent.model.build_vocab_cache()

"""Trains DESCENT and evaluates the best checkpoint on the test split.

usage: python scripts/train.py data=kbos
"""

import os
from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Lightning 2.2 reloads its checkpoints (e.g. for the test after training) with torch.load, which
# defaults to weights_only=True since torch 2.6 and then fails on the stored hyperparameters.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Adds the repository root to PYTHONPATH and sets PROJECT_ROOT (used in configs/paths).
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from descent import utils  # noqa: E402

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model and, if cfg.test is set, evaluates the best checkpoint on the test split."""
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")
    test_metrics = trainer.callback_metrics

    return {**train_metrics, **test_metrics}, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Hydra entry point; returns cfg.optimized_metric for hyperparameter searches."""
    utils.extras(cfg)
    metric_dict, _ = train(cfg)
    return utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()

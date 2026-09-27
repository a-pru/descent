"""Evaluates a DESCENT checkpoint on the test split.

usage: python scripts/eval.py data=kbos [ckpt_path=...] [data.dataset.config.random_ego_agent=false]
"""

from typing import List, Tuple

import hydra
import pyrootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Adds the repository root to PYTHONPATH and sets PROJECT_ROOT (used in configs/paths).
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from descent import utils  # noqa: E402

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates cfg.ckpt_path on the test split of cfg.data."""
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Both the released checkpoints and Lightning checkpoints store the weights in 'state_dict'.
    log.info(f"Loading weights from {cfg.ckpt_path}")
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule)
    return trainer.callback_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    utils.extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()

"""Hyperparameter logging for Lightning loggers."""

from lightning.pytorch.utilities import rank_zero_only

from descent.utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Sends the run config and the number of model parameters to all Lightning loggers.

    Args:
        object_dict: Dict with the Hydra config ('cfg'), the 'model' and the 'trainer'.
    """
    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams = {
        "model": cfg["model"],
        "model/params/total": sum(p.numel() for p in model.parameters()),
        "model/params/trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model/params/non_trainable": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
        "data": cfg["data"],
        "trainer": cfg["trainer"],
        "callbacks": cfg.get("callbacks"),
        "extras": cfg.get("extras"),
        "task_name": cfg.get("task_name"),
        "tags": cfg.get("tags"),
        "ckpt_path": cfg.get("ckpt_path"),
        "seed": cfg.get("seed"),
    }
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

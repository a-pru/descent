"""Task helpers for the train/eval scripts and ego-agent selection."""

import warnings
from importlib.util import find_spec
from typing import Callable

import torch
from omegaconf import DictConfig

from descent.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies the optional 'extras' config: ignore warnings, enforce tags, print the config."""
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Decorator that logs exceptions and the output directory and always closes wandb."""

    def wrap(cfg: DictConfig):
        try:
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")
            raise ex
        finally:
            log.info(f"Output dir: {cfg.paths.output_dir}")
            if find_spec("wandb"):
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()
        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Returns a logged metric by name, or None if no name is given."""
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")
    return metric_value


def separate_ego_agent(value: torch.Tensor, ego_list) -> torch.Tensor:
    """Selects the ego agent of each scene: (B, A, ...) -> (B, 1, ...)."""
    B = value.shape[0]
    ego_value = value[range(B), ego_list].unsqueeze(1)
    assert ego_value.shape[0] == B and ego_value.shape[1] == 1
    return ego_value

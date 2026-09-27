"""Lightning module for training and evaluating DESCENT."""

import itertools
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from lightning import LightningModule
from torchmetrics import MeanMetric

from descent.utils import global_masks as G
from descent.utils.losses import marginal_loss
from descent.utils.metrics import marginal_ade, marginal_fde
from descent.utils.utils import separate_ego_agent


class TrajPred(LightningModule):
    """Trains and evaluates a marginal trajectory predictor.

    Reports minADE and minFDE of the ego agent (in meters) at every horizon in net.pred_lens.
    """

    def __init__(
        self,
        optimizer: EasyDict,
        scheduler: torch.optim.lr_scheduler,
        net: torch.nn.Module,
        extra_params: EasyDict,
    ):
        """Sets up the network and the loss/metric trackers.

        Args:
            optimizer: AdamW parameters (lr, weight_decay, beta1, beta2).
            scheduler: Partially initialized learning-rate scheduler, or None.
            net: Prediction network.
            extra_params: 'seen_airports' and 'unseen_airports', for per-airport metrics.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["net"], logger=False)

        self.net = net
        self.hist_len = self.net.hist_len
        self.pred_lens = self.net.pred_lens
        self.max_pred_len = max(self.pred_lens)

        self.eparams = extra_params
        self.seen_airports = self.eparams.seen_airports
        self.unseen_airports = self.eparams.unseen_airports

        self.train_loss, self.val_loss, self.test_loss = MeanMetric(), MeanMetric(), MeanMetric()

        self.val_ade, self.test_ade, self.val_fde, self.test_fde = {}, {}, {}, {}
        for t in self.pred_lens:
            key = self.horizon_key(t)
            self.val_ade[key], self.test_ade[key] = MeanMetric(), MeanMetric()
            self.val_fde[key], self.test_fde[key] = MeanMetric(), MeanMetric()
        self.val_ade, self.test_ade = nn.ModuleDict(self.val_ade), nn.ModuleDict(self.test_ade)
        self.val_fde, self.test_fde = nn.ModuleDict(self.val_fde), nn.ModuleDict(self.test_fde)

        # Per-airport metrics.
        self.val_seen_ade, self.val_seen_fde = self.airport_metrics(self.seen_airports)
        self.test_seen_ade, self.test_seen_fde = self.airport_metrics(self.seen_airports)
        self.test_unseen_ade, self.test_unseen_fde = self.airport_metrics(self.unseen_airports)

    def horizon_key(self, t: int) -> str:
        """Metric key of a horizon; the longest one is 't=max'."""
        return "t=max" if t == self.max_pred_len else f"t={t}"

    def airport_metrics(self, airports: list):
        """minADE/minFDE metrics per airport and horizon."""
        ade, fde = {}, {}
        for t, airport in itertools.product(self.pred_lens, airports):
            ade[f"{airport}_t={t}"], fde[f"{airport}_t={t}"] = MeanMetric(), MeanMetric()
        return nn.ModuleDict(ade), nn.ModuleDict(fde)

    def on_train_start(self):
        """Discards the loss of Lightning's validation sanity check."""
        self.val_loss.reset()

    def model_step(self, batch: Any):
        """Runs the network on the observed history and computes the loss.

        Returns:
            loss: Scalar loss.
            pred_scores: Mode scores (B, A, K).
            mu: Predicted positions (B, A, T, K, 3).
            sigma: Predicted variances (B, A, T, K, 3).
            Y_fut: Ground-truth future positions (B, A, T_fut, 3).
        """
        scene = batch["scene_dict"]
        Y = scene["rel_sequences"]  # (B, A, T, D)
        X = torch.zeros_like(Y).type(torch.float)
        X[:, :, : self.hist_len] = Y[:, :, : self.hist_len]

        D = Y.shape[-1]
        Y = Y[..., G.REL_XYZ[:D]]

        out = self.net(X, scene)
        pred_scores, mu, sigma = out["pred_scores"], out["mu"], out["sigma"]

        loss = marginal_loss(
            pred_scores,
            mu,
            sigma,
            Y,
            ego_agent=scene["ego_agent_id"],
            agent_mask=scene["agent_masks"],
            data=scene,
        )
        return loss, pred_scores, mu, sigma, Y[:, :, self.hist_len :]

    def on_before_backward(self, loss):
        """Allows the non-deterministic grid_sample backward of the SDF loss."""
        torch.use_deterministic_algorithms(False)

    def on_after_backward(self):
        """Re-enables deterministic algorithms after the backward pass."""
        torch.use_deterministic_algorithms(True, warn_only=True)

    def training_step(self, batch: Any, batch_idx: int):
        """Logs and returns the training loss."""
        loss, _, _, _, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("losses/train", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def evaluation_step(self, batch: Any, split: str):
        """Shared validation/test step: logs the loss, minADE/minFDE and per-airport metrics."""
        loss, _, mu, _, fut_rel = self.model_step(batch)
        ego_agent = batch["scene_dict"]["ego_agent_id"]
        ego_mu = separate_ego_agent(mu, ego_agent)
        ego_fut = separate_ego_agent(fut_rel, ego_agent)
        mask = separate_ego_agent(batch["scene_dict"]["agent_masks"], ego_agent)

        loss_metric = self.val_loss if split == "val" else self.test_loss
        loss_metric(loss)
        self.log(
            f"losses/{split}", loss_metric, on_step=False, on_epoch=True, prog_bar=split == "val"
        )

        ade, fde = (
            (self.val_ade, self.val_fde) if split == "val" else (self.test_ade, self.test_fde)
        )
        for t in self.pred_lens:
            key = self.horizon_key(t)
            # Validation metrics are logged as 't=max'/'t=20', test metrics as '50'/'20'.
            name = key if split == "val" else t
            mu_t = ego_mu[:, :, self.hist_len : self.hist_len + t]
            mask_t = mask[:, :, self.hist_len : self.hist_len + t]
            fut_t = ego_fut[:, :, :t]

            ade[key](marginal_ade(mu_t, fut_t, mask=mask_t))
            self.log(
                f"{split}_ade/{name}",
                ade[key],
                on_step=False,
                on_epoch=True,
                prog_bar=split == "val",
            )
            fde[key](marginal_fde(mu_t, fut_t, mask=mask_t))
            self.log(
                f"{split}_fde/{name}",
                fde[key],
                on_step=False,
                on_epoch=True,
                prog_bar=split == "val",
            )

        if split == "val":
            groups = (
                [("val_seen", self.seen_airports, self.val_seen_ade, self.val_seen_fde)]
                if len(self.seen_airports) > 1
                else []
            )
        else:
            groups = [
                ("test_seen", self.seen_airports, self.test_seen_ade, self.test_seen_fde),
                ("test_unseen", self.unseen_airports, self.test_unseen_ade, self.test_unseen_fde),
            ]

        airport_ids = batch["scene_dict"]["airport_id"]
        for prefix, airports, ap_ade, ap_fde in groups:
            for airport in airports:
                airport_idx = np.where(airport_ids == airport)[0]
                if len(airport_idx) == 0:
                    continue
                airport_mu, airport_fut = ego_mu[airport_idx], ego_fut[airport_idx]
                airport_mask = mask[airport_idx]
                for t in self.pred_lens:
                    mu_t = airport_mu[:, :, self.hist_len : self.hist_len + t]
                    mask_t = airport_mask[:, :, self.hist_len : self.hist_len + t]
                    fut_t = airport_fut[:, :, :t]

                    key = f"{airport}_t={t}"
                    ap_ade[key](marginal_ade(mu_t, fut_t, mask=mask_t))
                    self.log(f"{prefix}_ade/{key}", ap_ade[key], on_step=False, on_epoch=True)
                    ap_fde[key](marginal_fde(mu_t, fut_t, mask=mask_t))
                    self.log(f"{prefix}_fde/{key}", ap_fde[key], on_step=False, on_epoch=True)

    def validation_step(self, batch: Any, batch_idx: int):
        """Logs validation loss and metrics."""
        self.evaluation_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        """Logs test loss and metrics."""
        self.evaluation_step(batch, "test")

    def configure_optimizers(self):
        """AdamW with weight decay only on Linear/Conv weights (minGPT style).

        See https://github.com/karpathy/minGPT/pull/24#issuecomment-679316025.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Conv1d)
        blacklist_weight_modules = (
            torch.nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.MultiheadAttention,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # The out_proj of MultiheadAttention is an nn.Linear and thus in 'decay'; drop it from
        # 'no_decay'.
        no_decay = set(p for p in no_decay if "attn.out_proj.weight" not in p)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (
            str(inter_params),
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.hparams.optimizer.weight_decay,
            },
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.optimizer.lr,
            betas=(self.hparams.optimizer.beta1, self.hparams.optimizer.beta2),
        )

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "losses/val",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

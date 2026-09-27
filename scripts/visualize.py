"""Plots DESCENT predictions for test scenes in the ego frame over the airport map.

usage: python scripts/visualize.py data=kbos [ckpt_path=...] [vis.num_scenes=16] [vis.out_dir=...]

Each figure shows the ego history (black), the other agents' histories (gray), the ground-truth
future (wide gray band), the context lane segments (thin lines) and the predicted modes whose
score is at least vis.min_score (colored, legend shows the softmax over the mode scores).
"""

import json
import os

import hydra
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pyrootutils
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
from matplotlib.lines import Line2D
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from descent.utils.utils import separate_ego_agent  # noqa: E402

MODE_COLORS = ["#D55E00", "#009E73", "royalblue", "crimson", "orange", "purple"]


def load_background(assets_dir: str, airport: str):
    """Returns the airport raster and its extent in km relative to the airport reference point."""
    image = plt.imread(os.path.join(assets_dir, airport, "bkg_map.png"))
    with open(os.path.join(assets_dir, airport, "limits.json"), "r") as fp:
        ref = json.load(fp)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(ref["ref_lat"]))
    ll = ref["espg_4326"]
    extent = [
        (ll["west"] - ref["ref_lon"]) * km_per_deg_lon,
        (ll["east"] - ref["ref_lon"]) * km_per_deg_lon,
        (ll["south"] - ref["ref_lat"]) * km_per_deg_lat,
        (ll["north"] - ref["ref_lat"]) * km_per_deg_lat,
    ]
    return image, extent


def plot_scene(
    scene: dict,
    b: int,
    mu: torch.Tensor,
    scores: torch.Tensor,
    background,
    hist_len: int,
    min_score: float,
    out_file: str,
) -> None:
    """Plots scene b of a batch; ego-frame points are drawn as (y, x) so that the map is upright."""
    rel = scene["rel_sequences"][b].cpu().numpy()  # (A, T, 5)
    masks = scene["agent_masks"][b].cpu().numpy()  # (A, T)
    lanes = scene["custom_context"][b].cpu().numpy()  # (M, P, 2)
    origin = scene["origin"][b].cpu().numpy()
    theta = scene["theta"][b, 0].item()
    mu = mu[b, 0].cpu().numpy()  # (T, K, 3)
    scores = scores[b, 0].cpu()  # (K,)

    fig, ax = plt.subplots(figsize=(10, 10))
    image, extent = background
    img = ax.imshow(image, extent=extent, zorder=0, alpha=0.4, cmap="gray_r")
    # Move the map into the ego frame: translate to the ego position, then rotate by theta.
    img.set_transform(
        transforms.Affine2D().translate(-origin[1], -origin[0]).rotate(theta) + ax.transData
    )

    for lane in lanes:
        ax.plot(lane[:, 1], lane[:, 0], color="dimgray", linewidth=1, zorder=1)

    for a in range(1, rel.shape[0]):
        valid = masks[a, :hist_len]
        ax.scatter(
            rel[a, :hist_len][valid, 1], rel[a, :hist_len][valid, 0], color="gray", s=15, zorder=5
        )
    valid = masks[0, :hist_len]
    ax.scatter(
        rel[0, :hist_len][valid, 1], rel[0, :hist_len][valid, 0], color="black", s=25, zorder=10
    )
    fut = rel[0, hist_len:][masks[0, hist_len:]]
    ax.plot(fut[:, 1], fut[:, 0], color="black", alpha=0.35, linewidth=8, zorder=6)

    probs = torch.softmax(scores, dim=0)
    legend = [
        Line2D([0], [0], color="black", marker="o", linewidth=0, label="Ego history"),
        Line2D([0], [0], color="black", alpha=0.35, linewidth=6, label="Ground-truth future"),
    ]
    points = [rel[0, :hist_len, :2], fut[:, :2]]
    for k in range(mu.shape[1]):
        if scores[k] < min_score:
            continue
        pred = mu[hist_len:, k]
        ax.plot(
            pred[:, 1], pred[:, 0], color=MODE_COLORS[k % len(MODE_COLORS)], linewidth=3, zorder=20
        )
        legend.append(
            Line2D(
                [0],
                [0],
                color=MODE_COLORS[k % len(MODE_COLORS)],
                linewidth=3,
                label=f"Mode {k} (p={probs[k].item():.2f})",
            )
        )
        points.append(pred[:, :2])

    points = np.concatenate(points, axis=0)
    margin = 0.1
    ax.set_xlim(points[:, 1].min() - margin, points[:, 1].max() + margin)
    ax.set_ylim(points[:, 0].min() - margin, points[:, 0].max() + margin)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.legend(handles=legend, loc="lower left", framealpha=0.85)
    ax.set_title(f"{scene['airport_id'][b]} scenario {scene['scenario_id'][b]}")
    fig.savefig(out_file, bbox_inches="tight", dpi=150)
    plt.close(fig)


@hydra.main(version_base="1.3", config_path="../configs", config_name="visualize.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: plots the first vis.num_scenes selected test scenes."""
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    datamodule.prepare_data()
    datamodule.setup("test")
    out_dir = cfg.vis.out_dir
    os.makedirs(out_dir, exist_ok=True)

    backgrounds, count = {}, 0
    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            batch = move_data_to_device(batch, device)
            scene = batch["scene_dict"]
            _, scores, mu, _, _ = model.model_step(batch)
            ego = scene["ego_agent_id"]
            mu, scores = separate_ego_agent(mu, ego), separate_ego_agent(scores, ego)
            for b in range(0, len(ego), cfg.vis.every_n):
                airport = scene["airport_id"][b]
                if airport not in backgrounds:
                    backgrounds[airport] = load_background(cfg.paths.assets_dir, airport)
                out_file = os.path.join(out_dir, f"{airport}_{scene['scenario_id'][b]}.png")
                plot_scene(
                    scene,
                    b,
                    mu,
                    scores,
                    backgrounds[airport],
                    model.hist_len,
                    cfg.vis.min_score,
                    out_file,
                )
                count += 1
                if count >= cfg.vis.num_scenes:
                    print(f"Saved {count} figures to {out_dir}")
                    return


if __name__ == "__main__":
    main()

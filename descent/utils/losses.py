"""Training loss: winner-takes-all NLL with an SDF lane-adherence term."""

import torch
from torch.nn import functional as F

from descent.utils.utils import separate_ego_agent


def marginal_loss(
    pred_scores: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    ego_agent: torch.Tensor,
    agent_mask: torch.Tensor,
    data: dict,
    sdf_weight: float = 10.0,
) -> torch.Tensor:
    """Winner-takes-all loss on the ego agent.

    Cross-entropy on the mode scores, Gaussian NLL on the mode closest to the ground truth and an
    SDF lane-adherence term on that mode.

    Args:
        pred_scores: Mode scores (B, A, H).
        mu: Predicted positions (B, A, T, H, D), ego frame in km.
        sigma: Predicted variances (B, A, T, H, D).
        target: Ground-truth trajectory (B, A, T, D).
        ego_agent: Index of the ego agent in each scene (B,).
        agent_mask: Valid timesteps (B, A, T).
        data: Batch scene dict; needs 'origin', 'theta' and 'sdf'.
        sdf_weight: Weight of the lane-adherence term.

    Returns:
        Scalar loss.
    """
    B, A, T, N, D = mu.size()
    A = 1
    assert ego_agent.shape[0] == mu.shape[0]
    mu = separate_ego_agent(mu, ego_agent)
    sigma = separate_ego_agent(sigma, ego_agent)
    pred_scores = separate_ego_agent(pred_scores, ego_agent)
    target = separate_ego_agent(target, ego_agent)
    agent_mask = separate_ego_agent(agent_mask, ego_agent)

    # Average the distance of each mode over the valid timesteps.
    distance = (mu - target[..., None, :]).norm(dim=-1)
    agent_mask = agent_mask.view(-1, T)
    distance = distance.view(-1, T, N)
    BA, _, _ = distance.shape
    agg_distance = torch.zeros(BA, N).to(agent_mask.device)
    for ba in range(BA):
        agg_distance[ba] = distance[ba, agent_mask[ba]].mean(dim=0)
    agg_distance = agg_distance.view(B, A, N)

    # Only the best mode and its valid timesteps enter the regression loss.
    gt_idx = agg_distance.argmin(dim=-1)
    amask = agent_mask.view(B, A, T, 1, 1).repeat(1, 1, 1, N, D)
    mask = F.one_hot(gt_idx, num_classes=N)[..., None, :, None].repeat(1, 1, T, 1, D) * amask
    mu = mu * mask
    sigma = sigma * mask
    target = target[..., None, :].repeat(1, 1, 1, N, 1) * mask

    loss_cls = F.cross_entropy(
        input=pred_scores.flatten(0, 1), target=gt_idx.flatten(), reduction="mean", ignore_index=N
    )
    loss_reg = F.gaussian_nll_loss(mu, target, sigma)
    loss = loss_cls + loss_reg

    # Lane adherence of the best mode. Masked timesteps are zero and lie at the ego position.
    best_mode = mu[:, 0, :, :, :2].permute(0, 2, 1, 3)
    best_mode = best_mode[torch.arange(B), gt_idx[:, 0]].unsqueeze(1)
    loss += sdf_weight * compute_global_sdf_loss(
        best_mode, data["origin"], data["theta"][:, 0], data
    )
    return loss


def compute_global_sdf_loss(
    pred_trajs_local: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor, data: dict
) -> torch.Tensor:
    """Mean squared distance of predicted points to the nearest lane segment.

    The distances are read from the airport's precomputed lane distance field (SDF, unsigned).

    Args:
        pred_trajs_local: Predicted points (B, K, T, 2) in the ego frame.
        origin: Global position of the ego agent (B, 3).
        theta: Global heading of the ego frame (B,).
        data: Contains 'sdf' = {'sdf_grid': (H, W), 'metadata': {'min_coords', 'max_coords', ...}}.

    Returns:
        Scalar loss.
    """
    sdf_grid = data["sdf"]["sdf_grid"]
    metadata = data["sdf"]["metadata"]

    B, K, T, _ = pred_trajs_local.shape

    sdf_grid = sdf_grid.to(pred_trajs_local.device)
    sdf_grid = sdf_grid.transpose(0, 1).float()

    # Rotate by theta and translate by origin to get from the ego frame to the global frame.
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.stack(
        [torch.stack([cos_t, -sin_t], dim=-1), torch.stack([sin_t, cos_t], dim=-1)], dim=1
    )
    local_xy = pred_trajs_local[..., :2].reshape(B, -1, 2)
    global_xy = torch.bmm(local_xy.float(), R.transpose(1, 2))
    global_xy = global_xy + origin[:, :2].unsqueeze(1)
    global_xy = global_xy.view(B, K, T, 2)

    # grid_sample expects coordinates in [-1, 1] over the SDF bounds.
    min_c = torch.tensor(metadata["min_coords"], device=global_xy.device).flip(0)
    max_c = torch.tensor(metadata["max_coords"], device=global_xy.device).flip(0)
    norm_xy = 2.0 * (global_xy - min_c) / (max_c - min_c) - 1.0

    dist_samples = F.grid_sample(
        sdf_grid.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1),
        norm_xy.view(B, 1, -1, 2).float(),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return torch.mean(dist_samples**2)

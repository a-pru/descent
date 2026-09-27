"""Marginal minADE and minFDE of the predicted modes."""

import torch


def marginal_ade(
    Y_hat: torch.Tensor, Y: torch.Tensor, mask: torch.Tensor = None, scale: float = 1000.0
) -> torch.Tensor:
    """Minimum average displacement error over the H modes, per agent.

    Args:
        Y_hat: Predicted modes (B, A, T', H, D); the last T steps are evaluated.
        Y: Ground-truth future (B, A, T, D) in km.
        mask: Valid timesteps (B, A, T'); only these are averaged.
        scale: Unit conversion of the error (km to m).

    Returns:
        Error of the best mode (B, A), in meters.
    """
    B, A, T, D = Y.size()
    error = (Y_hat[..., -T:, :, :] - Y[..., None, :]).norm(dim=-1)  # (B, A, T, H)
    if mask is None:
        error = error.mean(dim=2)
    else:
        mask = mask[:, :, -T:]
        error = error.view(B * A, T, -1)
        BA, T, H = error.shape
        mask = mask.view(BA, T)
        error_masked = torch.zeros(BA, H).to(error.device)
        for ba in range(BA):
            error_masked[ba] = error[ba, mask[ba]].mean(dim=0)
        error = error_masked.view(B, A, H)
    error = error.min(dim=-1)[0]
    return scale * error


def marginal_fde(
    Y_hat: torch.Tensor, Y: torch.Tensor, mask: torch.Tensor = None, scale: float = 1000.0
) -> torch.Tensor:
    """Minimum final displacement error over the H modes, per agent.

    With a mask, the final step is the last valid timestep of each agent.

    Args:
        Y_hat: Predicted modes (B, A, T', H, D); the last T steps are evaluated.
        Y: Ground-truth future (B, A, T, D) in km.
        mask: Valid timesteps (B, A, T').
        scale: Unit conversion of the error (km to m).

    Returns:
        Error of the best mode (B, A), in meters.
    """
    B, A, T, D = Y.size()
    if mask is None:
        error = (Y_hat[..., -1, :, :] - Y[..., -1, None, :]).norm(dim=-1)
    else:
        mask, Y_hat = mask[:, :, -T:], Y_hat[:, :, -T:]
        t = (mask != 0).cumsum(-1).argmax(-1)  # Last valid index
        x, y = torch.meshgrid(torch.arange(0, B), torch.arange(0, A), indexing="ij")
        Y_T = Y[x, y, t]  # (B, A, D)
        Y_hat_T = Y_hat[x, y, t]  # (B, A, H, D)
        error = (Y_hat_T - Y_T[..., None, :]).norm(dim=-1)
    error = error.min(dim=-1)[0]
    return scale * error

"""DESCENT network."""

from typing import Any, Dict

import torch
import torch.nn as nn

from descent.models.layers.decoder import MultimodalDecoder
from descent.models.layers.lane_embedding import LaneEmbeddingLayer
from descent.models.layers.transformer_blocks import Block


class Descent(nn.Module):
    """Predicts K trajectory modes for the ego agent from agent histories and PRS lane segments.

    The dataset selects the lane segments (see DescentDataset.transform_custom_map). All inputs are
    in the ego frame at the current timestep, with positions in km.
    """

    def __init__(
        self,
        hist_len: int = 10,
        pred_lens: list = [20, 50],
        embed_dim: int = 128,
        agent_enc_depth: int = 4,
        scene_enc_depth: int = 4,
        num_heads: int = 8,
        future_steps: int = 60,
        num_modes: int = 4,
        num_lane_types: int = 6,
    ) -> None:
        """Builds the encoders and the decoder.

        Args:
            hist_len: Number of observed timesteps.
            pred_lens: Evaluated prediction horizons in timesteps.
            embed_dim: Token dimension.
            agent_enc_depth: Number of self-attention blocks of the agent history encoder.
            scene_enc_depth: Number of cross-attention blocks of the scene encoder.
            num_heads: Attention heads per block.
            future_steps: Number of predicted timesteps.
            num_modes: Number of predicted modes K.
            num_lane_types: Number of lane type embeddings.
        """
        super().__init__()
        self.hist_len = hist_len
        self.pred_lens = pred_lens

        agent_dpr = [x.item() for x in torch.linspace(0, 0.2, agent_enc_depth)]
        scene_dpr = [x.item() for x in torch.linspace(0, 0.2, scene_enc_depth)]

        # Agent encoder: embeds every history step (position, speed, heading, step index) and
        # applies self-attention over time.
        self.h_proj = nn.Linear(6, embed_dim)
        self.h_embed = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=False,
                drop_path=agent_dpr[i],
                cross_attn=False,
            )
            for i in range(agent_enc_depth)
        )

        # Lane encoder: embeds each segment from its points (relative to the segment center).
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        # Positional embedding of every token from its position and heading.
        self.pos_embed = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Not used in the forward pass. Kept so that the released checkpoints load with strict=True.
        self.glob_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Scene encoder: the ego-agent token attends to all agent and lane tokens.
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=False,
                drop_path=scene_dpr[i],
                cross_attn=True,
            )
            for i in range(scene_enc_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.lane_type_embed = nn.Embedding(num_lane_types, embed_dim)

        self.decoder = MultimodalDecoder(embed_dim, future_steps, k=num_modes)

        self.apply(self._init_weights)

    def _init_weights(self, module: Any) -> None:
        """Initializes Linear and Embedding weights with N(0, 0.02) and zero biases."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, data: Dict) -> Dict:
        """Predicts K modes for the ego agent.

        Args:
            x: Agent sequences (B, A, T, 5) with x, y, z, speed, heading; only the first hist_len
                steps are read. Agent 0 is the ego agent. Modified in place.
            data: Batch scene dict with 'custom_context' (B, M, P, 2), 'lane_types' (B, M),
                'lane_padding_mask' (B, M, P), 'x_key_padding_mask' (B, A) and
                'lane_key_padding_mask' (B, M).

        Returns:
            Dict with 'pred_scores' (B, A, K), 'mu' (B, A, T, K, 3) and 'sigma' (B, A, T, K, 3).
            The ego prediction is repeated for all A agents.
        """
        device = x.device
        B, A, T, D = x.size()
        lane_positions = data["custom_context"]
        lane_padding_mask = data["lane_padding_mask"]
        lane_types = data["lane_types"]
        x_key_padding_mask = data["x_key_padding_mask"]
        lane_key_padding_mask = data["lane_key_padding_mask"]

        # Make each agent's history relative to its last observed position. This modifies x in
        # place, so the last position becomes zero and the agents' positional embedding below only
        # encodes heading, as during training. The reference is cloned first, because an in-place
        # update that reads its own output is a race on the GPU.
        actor_feat = x[:, :, : self.hist_len]
        actor_feat[..., :3] -= actor_feat[:, :, -1, :3].unsqueeze(-2).clone()

        hist_key_valid_mask = ~x_key_padding_mask
        actor_feat = actor_feat[hist_key_valid_mask]

        ts = (
            torch.arange(actor_feat.shape[-2])
            .view(1, -1, 1)
            .repeat(actor_feat.shape[0], 1, 1)
            .to(device)
            .float()
        )
        actor_feat = torch.cat([actor_feat, ts], dim=-1)  # Append the step index

        actor_feat = self.h_proj(actor_feat)
        for blk in self.h_embed:
            actor_feat = blk(actor_feat)
        actor_feat = torch.max(actor_feat, axis=1).values  # One token per agent

        # Scatter the valid agents back into a dense (B, A, C) tensor.
        actor_feat_tmp = torch.zeros(B * A, actor_feat.shape[-1], device=actor_feat.device)
        mask_indices = torch.nonzero(hist_key_valid_mask.view(B * A), as_tuple=True)
        actor_feat_tmp.scatter_(
            0, mask_indices[0].unsqueeze(1).expand(-1, actor_feat.shape[-1]), actor_feat
        )
        actor_feat = actor_feat_tmp.view(B, A, actor_feat.shape[-1])

        # Encode each lane segment relative to its center. Center and heading come from the two
        # middle points of the 20 samples.
        lane_centers = lane_positions[:, :, 9:11].mean(-2, keepdim=True)
        lanes_angles = torch.atan2(
            lane_positions[:, :, 10, 1] - lane_positions[:, :, 9, 1],
            lane_positions[:, :, 10, 0] - lane_positions[:, :, 9, 0],
        )
        lane_normalized = lane_positions - lane_centers
        lane_normalized = torch.cat(
            [lane_normalized.float(), (lane_padding_mask[..., None]).float()], dim=-1
        ).contiguous()
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D))
        lane_feat = lane_feat.view(B, M, -1)
        lane_feat += self.lane_type_embed.weight[lane_types]

        # Positional embedding of the agent and lane tokens (lanes have z = 0).
        lane_centers = torch.cat(
            [
                lane_centers[:, :, 0],
                torch.zeros((lane_centers.shape[0], lane_centers.shape[1], 1), device=device),
            ],
            dim=-1,
        )
        x_centers = torch.cat([x[:, :, self.hist_len - 1, :3], lane_centers], dim=1)
        angles = torch.cat([x[:, :, self.hist_len - 1, -1], lanes_angles], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        pos_feat = torch.cat([x_centers, x_angles], dim=-1).float()
        pos_embed = self.pos_embed(pos_feat)

        # The ego-agent token attends to all agent and lane tokens.
        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat([x_key_padding_mask, lane_key_padding_mask], dim=1)
        x_encoder = x_encoder + pos_embed

        x_agent = actor_feat[:, 0].unsqueeze(1)
        for blk in self.blocks:
            x_agent = blk(x_agent, k=x_encoder, v=x_encoder, key_padding_mask=key_padding_mask)
        x_agent = self.norm(x_agent)

        # Decode the modes and repeat them for every agent (marginal output format).
        out = self.decoder(x_agent, x_encoder, key_padding_mask, A)
        out["pred_scores"] = out["pred_scores"].unsqueeze(1).repeat(1, A, 1)
        out["mu"] = out["mu"].unsqueeze(1).repeat(1, A, 1, 1, 1)
        out["sigma"] = out["sigma"].unsqueeze(1).repeat(1, A, 1, 1, 1)
        return out

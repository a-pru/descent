"""Mode-query trajectory decoder."""

import torch
import torch.nn as nn

from descent.models.layers.transformer_blocks import Block


class MultimodalDecoder(nn.Module):
    """Decodes k trajectory modes for the ego agent.

    Learned mode queries alternately attend to the ego-agent token and to the lane tokens. MLP heads
    then predict each mode's trajectory, variance and score.
    """

    def __init__(self, embed_dim, future_steps, k=6) -> None:
        """Builds the mode queries, the attention blocks and the prediction heads.

        Args:
            embed_dim: Token dimension.
            future_steps: Number of predicted timesteps.
            k: Number of modes.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.k = k

        self.attn_depth = 3
        dpr = [x.item() for x in torch.linspace(0, 0.2, self.attn_depth)]
        self.lane_blks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=8,
                mlp_ratio=4.0,
                qkv_bias=False,
                drop_path=dpr[i],
                cross_attn=True,
                kdim=embed_dim,
                vdim=embed_dim,
            )
            for i in range(self.attn_depth)
        )
        self.agent_blks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=8,
                mlp_ratio=4.0,
                qkv_bias=False,
                drop_path=dpr[i],
                cross_attn=True,
                kdim=embed_dim,
                vdim=embed_dim,
            )
            for i in range(self.attn_depth)
        )

        self.sigma = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, future_steps * 3),
        )
        self.loc = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, future_steps * 3),
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, 1),
            nn.Sigmoid(),
        )

        self.mode_embed = nn.Embedding(self.k, embedding_dim=embed_dim)

        nn.init.normal_(self.mode_embed.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initializes Linear layers with Xavier-uniform and LayerNorms with ones and zeros."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_agent, x_encoder, key_padding_mask, num_agents):
        """Predicts the modes from the encoded scene.

        Args:
            x_agent: Encoded ego-agent token (B, 1, C).
            x_encoder: Scene tokens (B, A + M, C), A agents followed by M lanes.
            key_padding_mask: True for padded tokens (B, A + M).
            num_agents: A, the number of agent tokens in x_encoder.

        Returns:
            Dict with 'pred_scores' (B, k), 'mu' (B, T, k, 3) and 'sigma' (B, T, k, 3).
        """
        B = x_agent.shape[0]
        kv_lane = x_encoder[:, num_agents:]
        mask_lane = key_padding_mask[:, num_agents:]

        mode_query = self.mode_embed.weight.view(1, self.k, self.embed_dim).repeat(B, 1, 1)
        for i in range(self.attn_depth):
            mode_query = self.agent_blks[i](mode_query, k=x_agent, v=x_agent)
            mode_query = self.lane_blks[i](
                mode_query, k=kv_lane, v=kv_lane, key_padding_mask=mask_lane
            )

        loc = self.loc(mode_query).view(-1, self.k, self.future_steps, 3).permute(0, 2, 1, 3)
        sigma = (
            torch.exp(self.sigma(mode_query))
            .view(-1, self.k, self.future_steps, 3)
            .permute(0, 2, 1, 3)
        )
        pi = self.pi(mode_query).squeeze(-1)
        return {"pred_scores": pi, "mu": loc, "sigma": sigma}

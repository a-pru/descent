"""PointNet-style lane segment encoder."""

import torch
import torch.nn as nn


class LaneEmbeddingLayer(nn.Module):
    """Embeds each lane segment (a polyline of points) into one token."""

    def __init__(self, feat_channel, encoder_channel):
        """Builds the two point-wise convolution stages.

        Args:
            feat_channel: Features per point.
            encoder_channel: Output token dimension.
        """
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(feat_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, self.encoder_channel, 1),
        )

    def forward(self, x):
        """Maps segment points (N, P, C_in) to segment tokens (N, C)."""
        bs, n, _ = x.shape

        feature = self.first_conv(x.transpose(2, 1))  # (N, 256, P)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (N, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # (N, 512, P)
        feature = self.second_conv(feature)  # (N, C, P)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (N, C)
        return feature_global

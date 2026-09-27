"""Transformer building blocks: stochastic depth, MLP and pre-norm self-/cross-attention block."""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drops whole residual paths per sample (stochastic depth), as in timm."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Module wrapper of drop_path."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """Stores the drop probability and whether to rescale kept paths."""
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class Mlp(nn.Module):
    """Two-layer MLP with activation and dropout."""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        """Builds fc1, activation and fc2; hidden and output sizes default to in_features."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    """Pre-norm transformer block.

    With cross_attn=True the queries attend to separately normalized keys and values
    (see forward_cross).
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        cross_attn=False,
        kdim=None,
        vdim=None,
    ):
        """Builds the attention and MLP sub-blocks.

        Args:
            dim: Token dimension.
            num_heads: Attention heads.
            mlp_ratio: Hidden size of the MLP relative to dim.
            qkv_bias: Whether to add learned bias tokens to keys and values.
            drop: Dropout of the MLP.
            attn_drop: Dropout of the attention weights.
            drop_path: Stochastic depth probability.
            act_layer: Activation of the MLP.
            norm_layer: Normalization layer.
            cross_attn: Whether queries attend to separate keys/values.
            kdim: Key dimension (default dim).
            vdim: Value dimension (default dim).
        """
        super().__init__()
        self.cross_attn = cross_attn

        if kdim is None:
            kdim = dim
        if vdim is None:
            vdim = dim
        if self.cross_attn:
            self.normkv = norm_layer(vdim)
            self.normk = norm_layer(kdim)
        self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
            kdim=kdim,
            vdim=vdim,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_self(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        """Self-attention over src."""
        src2 = self.norm1(src)
        src2 = self.attn(
            query=src2,
            key=src2,
            value=src2,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src

    def forward_cross(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        kv: Optional[Tensor] = None,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
    ):
        """Attention of src to (k, v), to kv, or to itself if neither is given."""
        assert (k is None and v is None) or (k is not None and v is not None)
        if k is not None:
            q = self.norm1(src)
            k = self.normk(k)
            v = self.normkv(v)
        elif kv is not None:
            q = self.norm1(src)
            k = v = self.normkv(kv)
        else:
            q = k = v = self.norm1(src)

        attn_output = self.attn(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        # The residual connection starts from the normalized query, not from the block input.
        # The released checkpoints were trained this way.
        src = q + self.drop_path1(attn_output)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        kv: Optional[Tensor] = None,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
    ):
        if self.cross_attn:
            return self.forward_cross(
                src=src, kv=kv, mask=mask, k=k, v=v, key_padding_mask=key_padding_mask
            )
        return self.forward_self(src=src, mask=mask, key_padding_mask=key_padding_mask)

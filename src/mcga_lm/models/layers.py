
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Position-wise FFN used in Eq. (4) of the paper."""

    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """``Z^{(l+1)} = CrossAttn(Q = Z^{(l)}, K, V = X_t)`` -- paper Eq. (3).

    Pre-norm residual form; the paper does not specify norm placement, so the
    standard pre-norm Perceiver IO arrangement is used (ASSUMPTION).
    """

    def __init__(self, latent_dim: int, input_dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_latent = nn.LayerNorm(latent_dim)
        self.norm_input = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=heads,
            kdim=input_dim,
            vdim=input_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = FeedForward(latent_dim, dropout=dropout)
        self.norm_ff = nn.LayerNorm(latent_dim)

    def forward(
        self, latents: torch.Tensor, inputs: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q = self.norm_latent(latents)
        kv = self.norm_input(inputs)
        attended, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        latents = latents + attended
        return latents + self.ff(self.norm_ff(latents))


class SelfAttentionBlock(nn.Module):
    """``Z^{(l+2)} = SelfAttn(Z^{(l+1)}) + FFN(Z^{(l+1)})`` -- paper Eq. (4)."""

    def __init__(self, dim: int, heads: int, ff_mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        x = x + attended
        return x + self.ff(self.norm_ff(x))


class GatedLinearUnit(nn.Module):
    """GLU used inside the TFT's gating machinery (Lim et al., 2021)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """GRN of the Temporal Fusion Transformer.

    Paper Sec. 3.3 states the variable-selection weights ``v_t^{(d)}`` come from
    "a gated residual network"; the GRN definition follows Lim et al. (2021).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_dim)
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.out = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.context is not None and context is not None:
            h = h + self.context(context)
        h = F.elu(h)
        h = self.dropout(self.fc2(h))
        h = self.out(self.glu(h))
        return self.norm(self.proj(x) + h)

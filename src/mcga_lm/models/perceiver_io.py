
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from ..config import InputDims, PerceiverConfig
from .layers import CrossAttentionBlock, SelfAttentionBlock

MODALITIES = ("phys", "beh", "env", "ling")


@dataclass
class MultimodalBatch:
    """The raw input set ``X_t`` of Eq. (2).

    Shapes (B = batch):
        phys: (B, T_p, d_p)   4-ch EEG + HRV(SDNN, RMSSD, LF/HF) + EDA(tonic, phasic)
        beh:  (B, T_b, d_b)   gaze x, gaze y, pupil diameter, blink rate
        env:  (B, d_e)        location one-hot | partner one-hot | noise, sin/cos(time)
        ling: (B, L, d_w)     last L = 10 dialogue tokens
    """

    phys: torch.Tensor
    beh: torch.Tensor
    env: torch.Tensor
    ling: torch.Tensor

    def to(self, device) -> "MultimodalBatch":
        return MultimodalBatch(
            phys=self.phys.to(device),
            beh=self.beh.to(device),
            env=self.env.to(device),
            ling=self.ling.to(device),
        )

    @property
    def batch_size(self) -> int:
        return self.phys.shape[0]

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {"phys": self.phys, "beh": self.beh, "env": self.env, "ling": self.ling}


def fourier_position_features(length: int, bands: int, device, dtype) -> torch.Tensor:
    """Fourier position encoding for the input array (Jaegle et al., 2022).

    The paper does not specify a positional scheme (ASSUMPTION); Perceiver IO's
    standard Fourier features are used so that position survives the bottleneck.
    """
    # Built on CPU then moved: ``logspace`` has no MPS kernel as of torch 2.9,
    # and this is a tiny constant tensor either way.
    pos = torch.linspace(-1.0, 1.0, steps=max(length, 1), dtype=dtype).unsqueeze(-1).to(device)
    freqs = torch.logspace(0.0, math.log10(max(length, 2) / 2), steps=bands, dtype=dtype).to(device)
    scaled = pos * freqs.unsqueeze(0) * math.pi
    return torch.cat([pos, scaled.sin(), scaled.cos()], dim=-1)  # (length, 1 + 2*bands)


def pool_to_tokens(x: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Summarise a long window into ``n_tokens`` mean/std feature tokens.

    Sec. 3.2 describes ``x_phys`` as a raw resampled window, while Sec. 4.1
    describes the physiological streams being reduced to features (log band
    power per 2-s window, HRV indices over 60-s windows, SCL/SCR). Handing the
    Perceiver one token per raw sample follows the former and buries the single
    environmental token under ~128 noisy ones; pooling into segment
    mean-and-spread features follows the latter. See ASSUMPTION A-34.
    """
    b, t, d = x.shape
    if n_tokens <= 0 or n_tokens >= t:
        return x
    per = t // n_tokens
    trimmed = x[:, : per * n_tokens, :].reshape(b, n_tokens, per, d)
    return torch.cat([trimmed.mean(dim=2), trimmed.std(dim=2)], dim=-1)


class ModalityTokeniser(nn.Module):
    """Projects each modality of Eq. (2) into a common input array ``X_t``.

    ``x_beh`` and ``x_ling`` keep their resolution and ``x_env`` is a single
    static token, as Sec. 3.2 describes. ``x_phys`` is pooled into
    ``InputDims.phys_tokens`` mean/spread feature tokens (A-34); set
    ``phys_tokens = 0`` to feed the raw window instead.
    """

    def __init__(self, dims: InputDims, token_dim: int = 128, fourier_bands: int = 8) -> None:
        super().__init__()
        self.dims = dims
        self.token_dim = token_dim
        self.fourier_bands = fourier_bands
        pos_dim = 1 + 2 * fourier_bands
        phys_feat = dims.phys_dim * 2 if dims.phys_tokens else dims.phys_dim
        self.proj = nn.ModuleDict(
            {
                "phys": nn.Linear(phys_feat + pos_dim, token_dim),
                "beh": nn.Linear(dims.beh_features + pos_dim, token_dim),
                "env": nn.Linear(dims.env_dim + pos_dim, token_dim),
                "ling": nn.Linear(dims.ling_dim + pos_dim, token_dim),
            }
        )
        # Learned modality embedding so the bottleneck can tell streams apart and
        # "implicitly weight informative modalities" (Sec. 3.2).
        self.modality_embedding = nn.Parameter(torch.randn(len(MODALITIES), token_dim) * 0.02)

    def forward(self, batch: MultimodalBatch) -> torch.Tensor:
        tensors = {
            "phys": pool_to_tokens(batch.phys, self.dims.phys_tokens),
            "beh": batch.beh,
            "env": batch.env.unsqueeze(1),  # static token
            "ling": batch.ling,
        }
        tokens = []
        for idx, name in enumerate(MODALITIES):
            x = tensors[name]
            b, t, _ = x.shape
            pos = fourier_position_features(t, self.fourier_bands, x.device, x.dtype)
            pos = pos.unsqueeze(0).expand(b, -1, -1)
            projected = self.proj[name](torch.cat([x, pos], dim=-1))
            tokens.append(projected + self.modality_embedding[idx])
        return torch.cat(tokens, dim=1)  # (B, N_total, token_dim)

    @property
    def num_tokens(self) -> int:
        d = self.dims
        phys = d.phys_tokens if 0 < d.phys_tokens < d.phys_window else d.phys_window
        return phys + d.beh_window + 1 + d.ling_tokens


class PerceiverIOEncoder(nn.Module):
    """Perceiver IO cross-attention bottleneck (paper Sec. 3.2, Eqs. 3-4)."""

    def __init__(self, dims: InputDims, cfg: PerceiverConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dims = dims
        self.tokeniser = ModalityTokeniser(dims)
        token_dim = self.tokeniser.token_dim

        if cfg.use_cross_attention:
            # Z in R^{M x D}, Table 3.
            self.latents = nn.Parameter(torch.randn(cfg.num_latents, cfg.latent_dim) * 0.02)
            self.cross_blocks = nn.ModuleList(
                [
                    CrossAttentionBlock(cfg.latent_dim, token_dim, cfg.cross_heads, cfg.dropout)
                    for _ in range(cfg.depth)
                ]
            )
            self.self_blocks = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            SelfAttentionBlock(cfg.latent_dim, cfg.self_heads, cfg.ff_mult, cfg.dropout)
                            for _ in range(cfg.self_attn_per_block)
                        ]
                    )
                    for _ in range(cfg.depth)
                ]
            )
        else:
            # Sec. 4.4 "\ cross-attention": early concatenation + linear projection.
            self.early_proj = nn.Sequential(
                nn.Linear(token_dim * self.tokeniser.num_tokens, cfg.latent_dim),
                nn.GELU(),
                nn.Linear(cfg.latent_dim, cfg.latent_dim),
            )
        self.out_norm = nn.LayerNorm(cfg.latent_dim)

    # ------------------------------------------------------------------ #
    def encode_latents(self, batch: MultimodalBatch) -> torch.Tensor:
        """Return the full latent array ``Z`` (B, M, D); needed by L_recon (Eq. 12)."""
        x = self.tokeniser(batch)
        if not self.cfg.use_cross_attention:
            flat = x.reshape(x.shape[0], -1)
            return self.early_proj(flat).unsqueeze(1)  # degenerate M = 1 latent array
        z = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        for cross, selves in zip(self.cross_blocks, self.self_blocks):
            z = cross(z, x)  # Eq. (3)
            for block in selves:
                z = block(z)  # Eq. (4)
        return z

    def forward(self, batch: MultimodalBatch) -> torch.Tensor:
        """Mean-pool the latent array to ``z_ctx,t in R^D`` (Sec. 3.2)."""
        z = self.encode_latents(batch)
        return self.out_norm(z.mean(dim=1))

    @property
    def output_dim(self) -> int:
        return self.cfg.latent_dim


class LateFusionEncoder(nn.Module):
    """Ablation "\\ Perceiver IO" (Sec. 4.4): per-modality MLP, concatenated.

    Provides the same ``forward``/``encode_latents`` interface so the rest of the
    pipeline is unchanged.
    """

    def __init__(self, dims: InputDims, cfg: PerceiverConfig, hidden: int = 128) -> None:
        super().__init__()
        self.cfg = cfg
        self.dims = dims
        self.encoders = nn.ModuleDict(
            {
                "phys": nn.Sequential(nn.Linear(dims.phys_dim, hidden), nn.GELU()),
                "beh": nn.Sequential(nn.Linear(dims.beh_features, hidden), nn.GELU()),
                "env": nn.Sequential(nn.Linear(dims.env_dim, hidden), nn.GELU()),
                "ling": nn.Sequential(nn.Linear(dims.ling_dim, hidden), nn.GELU()),
            }
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * len(MODALITIES), cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.out_norm = nn.LayerNorm(cfg.latent_dim)

    def encode_latents(self, batch: MultimodalBatch) -> torch.Tensor:
        return self.forward(batch).unsqueeze(1)

    def forward(self, batch: MultimodalBatch) -> torch.Tensor:
        parts = [
            self.encoders["phys"](batch.phys).mean(dim=1),
            self.encoders["beh"](batch.beh).mean(dim=1),
            self.encoders["env"](batch.env),
            self.encoders["ling"](batch.ling).mean(dim=1),
        ]
        return self.out_norm(self.head(torch.cat(parts, dim=-1)))

    @property
    def output_dim(self) -> int:
        return self.cfg.latent_dim


def build_context_encoder(dims: InputDims, cfg: PerceiverConfig, ablate_perceiver: bool = False):
    """Factory honouring the Sec. 4.4 ablation switches."""
    if ablate_perceiver:
        return LateFusionEncoder(dims, cfg)
    return PerceiverIOEncoder(dims, cfg)

"""Perceiver IO encoder (paper Sec. 3.2, Eqs. 2-4)."""

from __future__ import annotations

import pytest


def _batch(cfg, torch, n=3):
    from mcga_lm.models.perceiver_io import MultimodalBatch

    d = cfg.inputs
    return MultimodalBatch(
        phys=torch.randn(n, d.phys_window, d.phys_dim),
        beh=torch.randn(n, d.beh_window, d.beh_features),
        env=torch.randn(n, d.env_dim),
        ling=torch.randn(n, d.ling_tokens, d.ling_dim),
    )


def test_latent_array_has_the_shape_of_table_3(torch_mod, small_cfg) -> None:
    from mcga_lm.models.perceiver_io import PerceiverIOEncoder

    enc = PerceiverIOEncoder(small_cfg.inputs, small_cfg.perceiver)
    latents = enc.encode_latents(_batch(small_cfg, torch_mod))
    assert latents.shape == (3, small_cfg.perceiver.num_latents, small_cfg.perceiver.latent_dim)


def test_context_embedding_is_mean_pooled_to_D(torch_mod, small_cfg) -> None:
    """Sec. 3.2: "the final latent array is mean-pooled to obtain z_ctx,t in R^D"."""
    from mcga_lm.models.perceiver_io import PerceiverIOEncoder

    enc = PerceiverIOEncoder(small_cfg.inputs, small_cfg.perceiver).eval()
    z = enc(_batch(small_cfg, torch_mod))
    assert z.shape == (3, small_cfg.perceiver.latent_dim)


def test_bottleneck_size_is_independent_of_input_length(torch_mod, small_cfg) -> None:
    """The point of the bottleneck (Sec. 3.2, Sec. 4.9: cost independent of N)."""
    from mcga_lm.models.perceiver_io import PerceiverIOEncoder

    enc = PerceiverIOEncoder(small_cfg.inputs, small_cfg.perceiver).eval()
    short = enc.encode_latents(_batch(small_cfg, torch_mod))
    small_cfg.inputs.phys_window *= 2
    enc.tokeniser.dims = small_cfg.inputs
    long = enc.encode_latents(_batch(small_cfg, torch_mod))
    assert short.shape == long.shape  # identical latent array despite 2x input


def test_cross_attention_ablation_changes_the_path_but_not_the_interface(torch_mod, small_cfg) -> None:
    """Sec. 4.4 "\\ cross-attention": early concatenation + linear projection."""
    from mcga_lm.models.perceiver_io import PerceiverIOEncoder

    small_cfg.perceiver.use_cross_attention = False
    enc = PerceiverIOEncoder(small_cfg.inputs, small_cfg.perceiver).eval()
    assert not hasattr(enc, "cross_blocks")
    z = enc(_batch(small_cfg, torch_mod))
    assert z.shape == (3, small_cfg.perceiver.latent_dim)


def test_late_fusion_ablation_matches_the_encoder_interface(torch_mod, small_cfg) -> None:
    """Sec. 4.4 "\\ Perceiver IO": per-modality MLP, concatenated."""
    from mcga_lm.models.perceiver_io import LateFusionEncoder, build_context_encoder

    enc = build_context_encoder(small_cfg.inputs, small_cfg.perceiver, ablate_perceiver=True)
    assert isinstance(enc, LateFusionEncoder)
    z = enc(_batch(small_cfg, torch_mod))
    assert z.shape == (3, small_cfg.perceiver.latent_dim)
    assert enc.encode_latents(_batch(small_cfg, torch_mod)).dim() == 3


def test_encoder_responds_to_every_modality(torch_mod, small_cfg) -> None:
    """A change in any stream of Eq. (2) must move z_ctx."""
    from mcga_lm.models.perceiver_io import PerceiverIOEncoder

    torch = torch_mod
    torch.manual_seed(0)
    enc = PerceiverIOEncoder(small_cfg.inputs, small_cfg.perceiver).eval()
    base = _batch(small_cfg, torch, n=1)
    z0 = enc(base)
    for field in ("phys", "beh", "env", "ling"):
        kwargs = base.as_dict().copy()
        kwargs[field] = torch.randn_like(kwargs[field])
        from mcga_lm.models.perceiver_io import MultimodalBatch

        z1 = enc(MultimodalBatch(**kwargs))
        assert not torch.allclose(z0, z1, atol=1e-5), f"z_ctx ignored the {field} stream"


def test_gradients_flow_to_the_latent_array(torch_mod, small_cfg) -> None:
    from mcga_lm.models.perceiver_io import PerceiverIOEncoder

    enc = PerceiverIOEncoder(small_cfg.inputs, small_cfg.perceiver)
    enc(_batch(small_cfg, torch_mod)).sum().backward()
    assert enc.latents.grad is not None and torch_mod.isfinite(enc.latents.grad).all()

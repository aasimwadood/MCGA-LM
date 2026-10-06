"""Stages 1-2: multimodal encoder and TFT pre-training (paper Sec. 3.7, 4.8).

Stage 1: "Perceiver IO and TFT are jointly pre-trained on the aggregated public
datasets ... using MSE for fatigue regression plus a contrastive loss aligning
contextual and subsequent-utterance embeddings". The reconstruction term of
Eq. (12) is included, as part of Eq. (9). The contrastive term needs an
utterance per record; MAMEM/CLAS/WESAD carry none, so it is active only for
records that have a ``transcript`` -- the "public corpus of ambulatory
physiological recordings and linguistic transcripts (released with our code)",
which has not been released (ERRATA.md, E-9).

Stage 2: "TFT pre-training: pre-trained on the same aggregated physiological
datasets". It repeats stage 1's data with no stated difference (E-9); here it is
a second pass that trains the TFT alone with the encoder frozen.

Sec. 4.8: 100 epochs, batch size 128, AdamW at 1e-4 with cosine decay.

When MAMEM/CLAS/WESAD are absent this falls back to
``synthetic_pretraining_corpus`` and says so in the report's ``source`` field --
a fatigue head fitted that way carries no physiological grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..data.physiology import PhysiologySynthesiser
from ..data.public_datasets import PhysiologyRecord, load_public_corpora, synthetic_pretraining_corpus
from ..losses import contrastive_loss, fatigue_loss, reconstruction_loss
from ..models.perceiver_io import MultimodalBatch
from ..pipeline import MCGALM
from ..seed import set_seed
from ..utils import get_logger, resolve_device

logger = get_logger(__name__)


@dataclass
class PretrainBatch:
    batch: MultimodalBatch
    fatigue: torch.Tensor


def adapt_channels(signals: np.ndarray, target_steps: int, target_channels: int) -> np.ndarray:
    """Crop/pad an arbitrary recording to the ``x_phys`` layout of Sec. 3.2.

    LIMITATION: the public corpora have different channel sets from each other
    and from the paper's ``x_phys`` layout; the paper says they are "harmonised"
    but not how. This linear crop/pad is a placeholder (ASSUMPTION A-27) and is
    why real-corpus pre-training here is marked UNVERIFIED.
    """
    x = np.asarray(signals, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] >= target_steps:
        start = (x.shape[0] - target_steps) // 2
        x = x[start : start + target_steps]
    else:
        x = np.pad(x, ((0, target_steps - x.shape[0]), (0, 0)), mode="edge")
    if x.shape[1] >= target_channels:
        x = x[:, :target_channels]
    else:
        x = np.pad(x, ((0, 0), (0, target_channels - x.shape[1])), mode="constant")
    mu, sd = x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True) + 1e-6
    return (x - mu) / sd


def records_to_batches(
    records: Sequence[PhysiologyRecord], cfg: Config, seed: int = 0
) -> Tuple[MultimodalBatch, torch.Tensor]:
    """Assemble ``X_t`` (Eq. 2) from physiological records.

    Behavioural channels are synthesised from the same fatigue level; the
    environmental and linguistic streams are zero, because the pre-training
    corpora carry neither (they inform ``s_cog`` not at all).
    """
    rng = np.random.default_rng(seed)
    synth = PhysiologySynthesiser(cfg.inputs)
    d = cfg.inputs
    phys = np.stack([adapt_channels(r.signals, d.phys_window, d.phys_dim) for r in records])
    beh = np.stack([synth.behaviour(r.fatigue, min(r.fatigue + 0.1, 1.0), rng) for r in records])
    env = np.zeros((len(records), d.env_dim), dtype=np.float32)
    ling = np.zeros((len(records), d.ling_tokens, d.ling_dim), dtype=np.float32)
    batch = MultimodalBatch(
        phys=torch.from_numpy(phys),
        beh=torch.from_numpy(beh),
        env=torch.from_numpy(env),
        ling=torch.from_numpy(ling),
    )
    fatigue = torch.tensor([r.fatigue for r in records], dtype=torch.float32)
    return batch, fatigue


def pretrain(
    cfg: Config,
    data_root: str = "data/raw",
    epochs: Optional[int] = None,
    out_dir: Optional[str] = None,
    allow_synthetic: bool = True,
    backend_factory: Optional[Callable[[], object]] = None,
) -> Dict[str, object]:
    """Run stages 1 and 2 and checkpoint the encoder.

    ``epochs`` overrides both stages' epoch counts. ``backend_factory`` builds
    the language backend that embeds transcripts for the stage-1 contrastive
    term; it is called only if some record has a transcript.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.training.device)

    records = load_public_corpora(data_root)
    source = "public(MAMEM/CLAS/WESAD)"
    if not records:
        if not allow_synthetic:
            raise FileNotFoundError(
                f"no public corpora under {data_root}; see data/README.md for the DOIs"
            )
        logger.warning(
            "No public corpora found under %s -- falling back to SYNTHETIC pre-training data. "
            "Results are not pre-trained on real physiology.",
            data_root,
        )
        records = synthetic_pretraining_corpus(cfg.inputs, seed=cfg.seed)
        source = "synthetic"

    backend = None
    n_transcripts = sum(1 for r in records if r.transcript)
    if n_transcripts and backend_factory is not None:
        backend = backend_factory()
    elif n_transcripts:
        logger.warning("%d records carry transcripts but no backend was given; "
                       "the stage-1 contrastive term is skipped", n_transcripts)
    else:
        logger.info("no record carries a transcript, so the stage-1 contrastive term is inactive")

    model = MCGALM(cfg).to(device)

    # Stage 1: encoder, TFT and reconstruction decoder, jointly.
    stage1_params = (
        list(model.context_encoder.parameters()) + list(model.tft.parameters()) + list(model.decoder.parameters())
    )
    stage1 = _run_stage(
        "stage 1", model, stage1_params, records, cfg, device,
        epochs or cfg.training.epochs, backend=backend, encoder_trainable=True,
    )
    # Stage 2: the TFT alone, on the same data.
    stage2 = _run_stage(
        "stage 2", model, list(model.tft.parameters()), records, cfg, device,
        epochs or cfg.training.tft_pretrain_epochs, backend=None, encoder_trainable=False,
    )

    history = stage1 + stage2
    result: Dict[str, object] = {
        "source": source,
        "n_records": len(records),
        "n_transcripts": n_transcripts,
        "contrastive_active": backend is not None,
        "epochs": len(stage1),
        "tft_epochs": len(stage2),
        "history": stage1,
        "stage2_history": stage2,
        "final_loss": stage1[-1]["loss"] if stage1 else float("nan"),
        "final_fatigue_mae": history[-1]["fatigue_mae"] if history else float("nan"),
        "final_recon_mse": stage1[-1]["recon_mse"] if stage1 else float("nan"),
    }
    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "context_encoder": model.context_encoder.state_dict(),
                "tft": model.tft.state_dict(),
                "decoder": model.decoder.state_dict(),
                "config": cfg.to_dict(),
                "pretraining_source": source,
            },
            path / "encoder_pretrained.pt",
        )
        result["checkpoint"] = str(path / "encoder_pretrained.pt")
    return result


def _run_stage(
    name: str,
    model: MCGALM,
    params: List[torch.nn.Parameter],
    records: Sequence[PhysiologyRecord],
    cfg: Config,
    device: torch.device,
    n_epochs: int,
    backend=None,
    encoder_trainable: bool = True,
) -> List[Dict[str, float]]:
    """One pre-training stage. With ``encoder_trainable=False`` the encoder is
    run without gradients and only the TFT's fatigue regression is optimised."""
    optimiser = torch.optim.AdamW(params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(n_epochs, 1))
    indices = np.arange(len(records))
    rng = np.random.default_rng(cfg.seed)
    history: List[Dict[str, float]] = []

    for epoch in range(n_epochs):
        rng.shuffle(indices)
        model.train()
        if not encoder_trainable:
            model.context_encoder.eval()
        epoch_losses: List[Dict[str, float]] = []
        for start in range(0, len(indices), cfg.training.batch_size):
            chunk = [records[i] for i in indices[start : start + cfg.training.batch_size]]
            if len(chunk) < 2:
                continue
            batch, fatigue = records_to_batches(chunk, cfg, seed=cfg.seed + epoch)
            batch, fatigue = batch.to(device), fatigue.to(device)
            if encoder_trainable:
                z_ctx, recon = model.encode_context(batch, with_reconstruction=True)
            else:
                with torch.no_grad():
                    z_ctx, recon = model.encode_context(batch, with_reconstruction=True)
            state = model.cognitive_state(z_ctx)
            l_fatigue = fatigue_loss(state.fatigue, fatigue)  # Eq. (10)
            l_recon = reconstruction_loss(batch.phys, recon)  # Eq. (12)
            loss = l_fatigue
            l_contrastive = z_ctx.new_zeros(())
            if encoder_trainable:
                loss = loss + cfg.loss.lambda_recon * l_recon
                if backend is not None:
                    l_contrastive = _transcript_contrastive(chunk, z_ctx, backend, cfg)  # Eq. (11)
                    loss = loss + cfg.loss.lambda_contrastive * l_contrastive
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.training.grad_clip)
            optimiser.step()
            epoch_losses.append(
                {
                    "loss": float(loss.detach()),
                    "fatigue_mse": float(l_fatigue.detach()),
                    "recon_mse": float(l_recon.detach()),
                    "contrastive": float(l_contrastive.detach()),
                    "fatigue_mae": float((state.fatigue - fatigue).abs().mean().detach()),
                }
            )
        scheduler.step()
        mean = (
            {k: float(np.mean([d[k] for d in epoch_losses])) for k in epoch_losses[0]}
            if epoch_losses
            else {k: float("nan") for k in ("loss", "fatigue_mse", "recon_mse", "contrastive", "fatigue_mae")}
        )
        history.append({"epoch": epoch, **mean})
        if epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            logger.info(
                "pretrain %s epoch %d/%d loss=%.5f (fatigue MSE %.5f, MAE %.4f | recon MSE %.4f)",
                name, epoch + 1, n_epochs, mean["loss"], mean["fatigue_mse"], mean["fatigue_mae"],
                mean["recon_mse"],
            )
    return history


def _transcript_contrastive(
    chunk: Sequence[PhysiologyRecord], z_ctx: torch.Tensor, backend, cfg: Config
) -> torch.Tensor:
    """Eq. (11) over the records in ``chunk`` that carry a transcript."""
    rows = [i for i, r in enumerate(chunk) if r.transcript]
    if len(rows) < 2:
        return z_ctx.new_zeros(())
    utt = torch.from_numpy(
        np.stack([backend.utterance_embedding(chunk[i].transcript) for i in rows]).astype(np.float32)
    ).to(z_ctx.device)
    ctx = z_ctx[rows]
    # Same width matching as stage 3 (ASSUMPTION A-29): crop or zero-pad.
    d = ctx.shape[-1]
    utt = utt[..., :d] if utt.shape[-1] >= d else torch.nn.functional.pad(utt, (0, d - utt.shape[-1]))
    return contrastive_loss(ctx, utt, cfg.loss.contrastive_temp)

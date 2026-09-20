

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..data.physiology import PhysiologySynthesiser
from ..data.public_datasets import PhysiologyRecord, load_public_corpora, synthetic_pretraining_corpus
from ..losses import fatigue_loss, reconstruction_loss
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
) -> Dict[str, object]:
    """Run stages 1-2 and checkpoint the encoder."""
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

    model = MCGALM(cfg).to(device)
    # Stage 1-2 touch the encoder, the TFT and the reconstruction decoder only.
    params = list(model.context_encoder.parameters()) + list(model.tft.parameters()) + list(model.decoder.parameters())
    optimiser = torch.optim.AdamW(params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    n_epochs = epochs or cfg.training.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(n_epochs, 1))

    indices = np.arange(len(records))
    rng = np.random.default_rng(cfg.seed)
    history: List[Dict[str, float]] = []

    for epoch in range(n_epochs):
        rng.shuffle(indices)
        model.train()
        epoch_losses: List[float] = []
        for start in range(0, len(indices), cfg.training.batch_size):
            chunk = [records[i] for i in indices[start : start + cfg.training.batch_size]]
            if len(chunk) < 2:
                continue
            batch, fatigue = records_to_batches(chunk, cfg, seed=cfg.seed + epoch)
            batch, fatigue = batch.to(device), fatigue.to(device)
            z_ctx, recon = model.encode_context(batch, with_reconstruction=True)
            state = model.cognitive_state(z_ctx)
            l_fatigue = fatigue_loss(state.fatigue, fatigue)  # Eq. (10)
            l_recon = reconstruction_loss(batch.phys, recon)  # Eq. (12)
            loss = l_fatigue + cfg.loss.lambda_recon * l_recon
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.training.grad_clip)
            optimiser.step()
            epoch_losses.append(
                {
                    "loss": float(loss.detach()),
                    "fatigue_mse": float(l_fatigue.detach()),
                    "recon_mse": float(l_recon.detach()),
                    "fatigue_mae": float((state.fatigue - fatigue).abs().mean().detach()),
                }
            )
        scheduler.step()
        mean = (
            {k: float(np.mean([d[k] for d in epoch_losses])) for k in epoch_losses[0]}
            if epoch_losses
            else {"loss": float("nan"), "fatigue_mse": float("nan"), "recon_mse": float("nan"),
                  "fatigue_mae": float("nan")}
        )
        mean_loss = mean["loss"]
        history.append({"epoch": epoch, **mean})
        if epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            logger.info(
                "pretrain epoch %d/%d loss=%.5f (fatigue MSE %.5f, MAE %.4f | recon MSE %.4f)",
                epoch + 1, n_epochs, mean_loss, mean["fatigue_mse"], mean["fatigue_mae"], mean["recon_mse"],
            )

    result: Dict[str, object] = {
        "source": source,
        "n_records": len(records),
        "epochs": n_epochs,
        "history": history,
        "final_loss": history[-1]["loss"] if history else float("nan"),
        "final_fatigue_mae": history[-1]["fatigue_mae"] if history else float("nan"),
        "final_recon_mse": history[-1]["recon_mse"] if history else float("nan"),
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

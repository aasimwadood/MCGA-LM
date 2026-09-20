
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

# Paper Sec. 4.9 / Table 6 reference figures, on an NVIDIA Jetson AGX Orin.
PAPER_SHARES = {
    "LLM generation": 0.613,
    "MC Dropout": 0.123,
    "Perceiver IO": 0.098,
    "Pre/post-processing": 0.092,
    "TFT": 0.048,
    "GAT retrieval": 0.026,
}
PAPER_DECODER_TFLOP = 0.32  # 2 * 8e9 * 20
PAPER_PEAK_MEMORY_GB = 5.6
JETSON_MEMORY_GB = 32.0

BYTES_PER_PARAM = {"nf4": 0.5, "int8": 1.0, "fp16": 2.0, "bf16": 2.0, "fp32": 4.0}


# --------------------------------------------------------------- decoder --- #
def decoder_flops(n_params: int = 8_000_000_000, n_tokens: int = 20) -> float:
    """``~ 2PL`` FLOPs for autoregressive decoding (Sec. 4.9, first sentence).

    The paper's own worked example: ``P = 8e9``, ``L = 20`` -> 0.32 TFLOP.
    Two FLOPs per parameter per token is the standard forward-pass convention
    (one multiply, one accumulate); attention and the KV cache are second-order
    at these lengths and are excluded, as the paper's formula excludes them.
    """
    return 2.0 * float(n_params) * float(n_tokens)


# ------------------------------------------------------------- perceiver --- #
def perceiver_flops(
    n_latents: int = 256,
    latent_dim: int = 512,
    n_inputs: int = 47,
    depth: int = 2,
    self_attn_per_block: int = 2,
    ff_mult: int = 2,
) -> float:
    """``O(MND + M^2 D)`` for the Perceiver IO encoder (Sec. 4.9).

    The first term is cross-attention against the ``N`` input tokens, the second
    is self-attention inside the fixed ``M x D`` latent array. Because ``M`` and
    ``D`` are fixed by Table 3, only the first term depends on the input length
    at all -- which is the property the paper is claiming when it says cost "is
    independent of input length ``N``" for the dominant self-attention term.
    """
    m, d, n = float(n_latents), float(latent_dim), float(n_inputs)
    # Cross-attention: QK^T and (attn)V, each M x N x D multiply-accumulates.
    cross = 2.0 * 2.0 * m * n * d
    # Projections for Q (from latents) and K, V (from inputs).
    cross += 2.0 * (m * d * d + 2.0 * n * d * d)
    # Self-attention over the latent array: the M^2 D term.
    self_attn = 2.0 * 2.0 * m * m * d + 2.0 * 4.0 * m * d * d
    # Position-wise feed-forward.
    ffn = 2.0 * 2.0 * m * d * (d * ff_mult)
    return depth * (cross + self_attn_per_block * (self_attn + ffn))


# ------------------------------------------------------------------- TFT --- #
def tft_flops(window: int = 32, state_dim: int = 64, lstm_hidden: int = 128, d_model: int = 512) -> float:
    """``O(W^2 D_c)`` for the Temporal Fusion Transformer (Sec. 4.9).

    The quadratic term is the self-attention over the ``W = 32`` step window;
    the LSTM and the variable-selection network are linear in ``W`` and are
    included because at ``W = 32`` they are not negligible, but they do not
    change the order.
    """
    w, dc, h, d = float(window), float(state_dim), float(lstm_hidden), float(d_model)
    vsn = 2.0 * w * d * d
    lstm = 2.0 * 4.0 * w * (d + h) * h
    attn = 2.0 * 2.0 * w * w * dc + 2.0 * 4.0 * w * dc * dc
    return vsn + lstm + attn


# ------------------------------------------------------------------- GAT --- #
def gat_flops(n_active_edges: int = 40, heads: int = 4, head_dim: int = 64, node_dim: int = 300) -> float:
    """``O(|E_active| K D_h)`` for graph retrieval (Sec. 4.9).

    Priced over the *active neighbourhood* only, not the whole 200-500 node
    graph -- that restriction is the paper's point. Node projections are counted
    over the endpoints the active edges touch, bounded by ``2 |E_active|``.
    """
    e, k, dh, dn = float(n_active_edges), float(heads), float(head_dim), float(node_dim)
    project = 2.0 * (2.0 * e) * dn * (k * dh)
    attend = 2.0 * e * k * dh
    aggregate = 2.0 * e * k * dh
    return project + attend + aggregate


# ----------------------------------------------------------- MC dropout --- #
def mc_dropout_flops(head_params: int = 300_000, n_passes: int = 20, n_tokens: int = 20) -> float:
    """``N`` stochastic passes over the *intent-scoring head only* (Sec. 4.9).

    This is the claim that matters for the latency budget: uncertainty
    estimation "applies ``N = 20`` stochastic passes to the intent-scoring head
    alone rather than re-running generation, which keeps it at 12% of the budget
    instead of a 20x multiplier". Compare with :func:`naive_mc_dropout_flops`.
    """
    return 2.0 * float(head_params) * float(n_tokens) * float(n_passes)


def naive_mc_dropout_flops(
    n_params: int = 8_000_000_000, n_tokens: int = 20, n_passes: int = 20
) -> float:
    """The multiplier that was avoided: ``N`` full re-generations of the decoder."""
    return decoder_flops(n_params, n_tokens) * float(n_passes)


# ---------------------------------------------------------------- memory --- #
def peak_memory_bytes(
    n_params: int = 8_000_000_000,
    quantisation: str = "nf4",
    lora_rank: int = 16,
    n_latents: int = 256,
    latent_dim: int = 512,
    n_tokens: int = 20,
    kv_layers: int = 32,
    kv_dim: int = 4096,
    activation_overhead: float = 1.35,
) -> Dict[str, float]:
    """Peak resident memory of the deployed configuration (Sec. 4.9: 5.6 GB).

    Broken into the terms that actually move: quantised decoder weights, the
    LoRA adapter kept in fp16, the KV cache for a short utterance, the fixed
    latent array, and a multiplicative activation/runtime overhead.
    ``activation_overhead`` is ASSUMPTION A-35 -- the paper reports the total,
    not the split.
    """
    bpp = BYTES_PER_PARAM.get(quantisation.lower())
    if bpp is None:
        raise ValueError(f"unknown quantisation {quantisation!r}; expected one of {sorted(BYTES_PER_PARAM)}")
    weights = float(n_params) * bpp
    # LoRA A and B for the Table 3 target modules, fp16.
    lora = 2.0 * float(lora_rank) * float(kv_dim) * 2.0 * float(kv_layers) * 2.0
    kv_cache = 2.0 * float(kv_layers) * float(kv_dim) * float(n_tokens) * 2.0
    latents = float(n_latents) * float(latent_dim) * 4.0
    subtotal = weights + lora + kv_cache + latents
    return {
        "weights": weights,
        "lora": lora,
        "kv_cache": kv_cache,
        "latents": latents,
        "subtotal": subtotal,
        "total": subtotal * activation_overhead,
    }


# ---------------------------------------------------------------- report --- #
@dataclass
class CostReport:
    """Per-turn analytic cost, with the paper's figures alongside."""

    flops: Dict[str, float]
    total_flops: float
    analytic_shares: Dict[str, float]
    measured_shares: Dict[str, float] = field(default_factory=dict)
    paper_shares: Dict[str, float] = field(default_factory=lambda: dict(PAPER_SHARES))
    memory: Dict[str, float] = field(default_factory=dict)
    mc_dropout_multiplier_avoided: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "flops": self.flops,
            "total_flops": self.total_flops,
            "total_tflops": self.total_flops / 1e12,
            "analytic_shares": self.analytic_shares,
            "measured_shares": self.measured_shares,
            "paper_shares": self.paper_shares,
            "memory_bytes": self.memory,
            "memory_gb": {k: v / 1e9 for k, v in self.memory.items()},
            "paper_peak_memory_gb": PAPER_PEAK_MEMORY_GB,
            "jetson_envelope_gb": JETSON_MEMORY_GB,
            "fits_envelope": self.memory.get("total", 0.0) / 1e9 < JETSON_MEMORY_GB,
            "mc_dropout_multiplier_avoided": self.mc_dropout_multiplier_avoided,
        }

    def to_markdown(self) -> str:
        rows = [
            "| Stage | analytic GFLOP | analytic share | measured share | paper share |",
            "|---|---|---|---|---|",
        ]
        for name, value in self.flops.items():
            measured = self.measured_shares.get(name)
            paper = self.paper_shares.get(name)
            rows.append(
                f"| {name} | {value / 1e9:.3f} | {self.analytic_shares[name] * 100:.1f}% | "
                f"{'-' if measured is None else f'{measured * 100:.1f}%'} | "
                f"{'-' if paper is None else f'{paper * 100:.1f}%'} |"
            )
        rows.append(f"| **Total** | **{self.total_flops / 1e9:.3f}** | 100% | | |")
        rows.append("")
        rows.append(
            "Analytic FLOP share and measured wall-clock share are *not* expected to "
            "agree. Decoding is memory-bandwidth bound rather than FLOP bound, so it "
            "takes a far larger share of arithmetic than of time, while the small "
            "perception kernels are latency bound and take more time than their "
            "arithmetic suggests. The paper's 61.3% for decode is a share of measured "
            "wall-clock time (Sec. 4.9); the FLOP column is what the shapes cost."
        )
        if self.memory:
            rows.append("")
            rows.append(
                f"Peak memory (analytic): {self.memory['total'] / 1e9:.2f} GB "
                f"(paper Sec. 4.9: {PAPER_PEAK_MEMORY_GB} GB; Jetson envelope {JETSON_MEMORY_GB:.0f} GB)"
            )
        if self.mc_dropout_multiplier_avoided:
            rows.append(
                f"MC Dropout on the scoring head costs "
                f"{1.0 / self.mc_dropout_multiplier_avoided:.2e} of the naive "
                f"N-times-regeneration alternative it replaces (Sec. 4.9)."
            )
        return "\n".join(rows)


def cost_report(
    cfg=None,
    n_decoder_params: int = 8_000_000_000,
    n_tokens: int = 20,
    n_inputs: int = 47,
    n_active_edges: int = 40,
    head_params: int = 300_000,
    measured_ms: Optional[Mapping[str, float]] = None,
) -> CostReport:
    """Assemble the Sec. 4.9 cost model, optionally beside measured latencies.

    ``cfg`` is an optional :class:`~mcga_lm.config.Config`; when given, the
    shapes come from it rather than from the Table 3 defaults, so the report
    describes the configuration that actually ran. ``measured_ms`` maps the
    Table 6 stage names to measured milliseconds and contributes the "measured
    share" column -- the quantity the paper compares its analytic model against.
    """
    if cfg is not None:
        p, t, g, s = cfg.perceiver, cfg.tft, cfg.graph, cfg.safety
        perceiver = perceiver_flops(
            n_latents=p.num_latents,
            latent_dim=p.latent_dim,
            n_inputs=n_inputs,
            depth=p.depth,
            self_attn_per_block=p.self_attn_per_block,
            ff_mult=p.ff_mult,
        )
        tft = tft_flops(window=t.window, state_dim=t.state_dim, lstm_hidden=t.lstm_hidden, d_model=p.latent_dim)
        gat = gat_flops(
            n_active_edges=n_active_edges, heads=g.gat_heads, head_dim=g.gat_hidden, node_dim=g.node_dim
        )
        mc = mc_dropout_flops(head_params=head_params, n_passes=s.mc_passes, n_tokens=n_tokens)
        naive = naive_mc_dropout_flops(n_decoder_params, n_tokens, s.mc_passes)
    else:
        perceiver = perceiver_flops(n_inputs=n_inputs)
        tft = tft_flops()
        gat = gat_flops(n_active_edges=n_active_edges)
        mc = mc_dropout_flops(head_params=head_params, n_tokens=n_tokens)
        naive = naive_mc_dropout_flops(n_decoder_params, n_tokens)

    flops = {
        "LLM generation": decoder_flops(n_decoder_params, n_tokens),
        "MC Dropout": mc,
        "Perceiver IO": perceiver,
        "TFT": tft,
        "GAT retrieval": gat,
    }
    total = sum(flops.values())
    shares = {k: (v / total if total > 0 else float("nan")) for k, v in flops.items()}

    measured_shares: Dict[str, float] = {}
    if measured_ms:
        denom = sum(measured_ms.values())
        if denom > 0:
            for name in flops:
                matched = [v for k, v in measured_ms.items() if k.split(" (N=")[0] == name]
                if matched:
                    measured_shares[name] = sum(matched) / denom

    return CostReport(
        flops=flops,
        total_flops=total,
        analytic_shares=shares,
        measured_shares=measured_shares,
        memory=peak_memory_bytes(n_params=n_decoder_params, n_tokens=n_tokens),
        mc_dropout_multiplier_avoided=naive / mc if mc > 0 else 0.0,
    )

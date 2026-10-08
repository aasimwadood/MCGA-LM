"""Configuration objects for MCGA-LM.

Every default here is traceable to the paper; the comment on each field gives the
section or table. Values marked ASSUMPTION are not fixed by the paper text.

Reference: Al-Nefaie et al., "MCGA-LM: Multimodal Context Graph-Augmented
Language Model for Adaptive Intent Reconstruction in Assistive Communication".
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, get_type_hints

import yaml


# --------------------------------------------------------------------------- #
# Phase I: multimodal contextual grounding (paper Sec. 3.2, Table 2)
# --------------------------------------------------------------------------- #
@dataclass
class PerceiverConfig:
    """Perceiver IO encoder (Sec. 3.2, Eqs. 3-4; Table 2)."""

    num_latents: int = 256  # M, Table 2
    latent_dim: int = 512  # D, Table 2
    cross_heads: int = 8  # Table 2
    self_heads: int = 8  # A-01: not given; mirrors cross_heads
    depth: int = 2  # A-01: number of (cross, self) blocks
    self_attn_per_block: int = 2  # A-01
    ff_mult: int = 2  # A-01: FFN expansion in Eq. (4)
    dropout: float = 0.1  # A-01
    # Ablation switch (Sec. 4.5 "\ cross-attention"): replace the cross-attention
    # bottleneck with early concatenation + linear projection.
    use_cross_attention: bool = True


@dataclass
class InputDims:


    # x_phys: 4-channel EEG + HRV (SDNN, RMSSD, LF/HF) + EDA (tonic, phasic)
    eeg_channels: int = 4  # Sec. 3.2
    hrv_features: int = 3  # SDNN, RMSSD, LF/HF (Sec. 3.2)
    eda_features: int = 2  # tonic, phasic (Sec. 3.2)
    phys_window: int = 128  # T_p. A-02: 2 s @ 64 Hz (Sec. 3.2 resampling)
    phys_rate_hz: int = 64  # Sec. 3.2
    # Number of pooled feature tokens the physiological window becomes before
    # entering the Perceiver (A-34). 0 feeds the raw window, one token per
    # sample, which buries the single environmental token under T_p noisy ones.
    phys_tokens: int = 8

    # x_beh: gaze (x, y), pupil diameter, blink rate, in 500 ms windows
    beh_features: int = 4  # Sec. 3.2
    beh_window: int = 8  # T_b. A-02: 8 x 500 ms = 4 s of behaviour

    # x_env: location one-hot + noise (dBA) + time-of-day + partner identity
    env_location_classes: int = 6  # A-02: bedroom/kitchen/living room/clinic/garden/outdoors
    env_partner_classes: int = 8  # A-02: max distinct diarised partners
    env_scalars: int = 3  # noise dBA + sin/cos time-of-day (Sec. 4.1 cyclic encoding)

    # x_ling: last L = 10 tokens embedded with the LLM tokeniser
    ling_tokens: int = 10  # L, Sec. 3.2
    # d_w: history is "embedded using the same tokeniser as the downstream LLM"
    # (Sec. 3.2). 256 for the default template backend; configs/llm_hf.yaml sets
    # 4096, LLaMA-3-8B's hidden size.
    ling_dim: int = 256

    @property
    def phys_dim(self) -> int:
        return self.eeg_channels + self.hrv_features + self.eda_features

    @property
    def env_dim(self) -> int:
        return self.env_location_classes + self.env_partner_classes + self.env_scalars


# --------------------------------------------------------------------------- #
# Phase II: temporal cognitive adaptation (paper Sec. 3.3, Table 2)
# --------------------------------------------------------------------------- #
@dataclass
class TFTConfig:
    """Temporal Fusion Transformer (Sec. 3.3, Eq. 5; Table 2)."""

    window: int = 32  # W, Table 2 (~160 s)
    lstm_hidden: int = 128  # Table 2
    attn_heads: int = 4  # Table 2
    state_dim: int = 64  # D_c, Sec. 3.3
    dropout: float = 0.1  # ASSUMPTION
    # Ablation switch (Sec. 4.5 "\ TFT"): clamp s_cog to the "low fatigue" state.
    enabled: bool = True


# --------------------------------------------------------------------------- #
# Phase III: personalised intent graph memory (paper Sec. 3.4, Table 2)
# --------------------------------------------------------------------------- #
@dataclass
class GraphConfig:
    """Dynamic User Intent Graph + GAT (Sec. 3.4, Eqs. 6-7; Table 2)."""

    node_dim: int = 300  # GloVe-sized node embeddings, Sec. 3.4 / Table 2
    gat_heads: int = 4  # K, Table 2
    gat_hidden: int = 64  # ASSUMPTION: per-head output width
    top_k: int = 5  # K_top, Table 2 ("typically K = 5")
    half_life_days: float = 30.0  # Sec. 3.4 edge-weight decay
    prune_threshold: float = 0.05  # ASSUMPTION: "infrequent edges pruned"
    target_nodes: int = 400  # Sec. 4.1: persona graphs hold ~400 nodes
    min_nodes: int = 200  # Sec. 3.4 steady state 200-500
    max_nodes: int = 500  # Sec. 3.4 steady state 200-500
    # Form of Eq. (7). 'paper' (default) is the printed equation, which is
    # additively separable and therefore yields near query-independent
    # attention; 'query_gated' adds a multiplicative query term that has no
    # counterpart in the paper. See the module docstring of models/gat.py.
    attention_form: str = "paper"
    # How the "aggregated attention score" of Sec. 3.4 is read. 'incoming'
    # (default) is the attention a node receives, which is what Eq. (13)
    # supervises ("the normalised attention coefficient from the GAT");
    # 'query' scores nodes by how strongly q_t attends to them, which has no
    # counterpart in the paper. See models/gat.py.
    node_scoring: str = "incoming"
    negative_slope: float = 0.2  # ASSUMPTION: standard GAT LeakyReLU slope
    dropout: float = 0.1  # ASSUMPTION
    # Ablation switch (Sec. 4.5 "\ GAT"): drop sub-graph retrieval entirely.
    enabled: bool = True


# --------------------------------------------------------------------------- #
# Phase IV: retrieval-augmented intent synthesis (paper Sec. 3.5, Table 2)
# --------------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    """RAG generation stage (Sec. 3.5; Table 2).

    ``backend='template'`` (default) is a weight-free template realiser that runs
    on CPU, as the paper's Data and Code Availability statement describes.
    ``backend='hf'`` is LLaMA-3-8B-Instruct (Table 2), which needs a CUDA GPU and
    gated weights; configs/llm_hf.yaml selects it.
    """

    backend: str = "template"  # 'template' | 'hf'
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"  # Table 3
    quantisation: str = "nf4"  # Table 2 (4-bit NF4)
    lora_rank: int = 16  # r, Table 2
    lora_alpha: int = 32  # alpha, Table 2
    lora_dropout: float = 0.05  # ASSUMPTION
    lora_targets: List[str] = field(  # Sec. 3.5: "all attention matrices and FF gates"
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )
    top_p: float = 0.9  # Sec. 3.5
    temp_low_fatigue: float = 1.2  # Sec. 3.5 / Table 2
    temp_high_fatigue: float = 0.5  # Sec. 3.5
    history_turns: int = 10  # Sec. 3.5 component 4
    max_new_tokens: int = 24  # ASSUMPTION: mean L = 20 tokens in Sec. 4.10
    # Verbosity ladder used by the fatigue-adaptive prompt (Sec. 3.5 component 2).
    max_words_by_fatigue: Dict[str, int] = field(
        default_factory=lambda: {"low": 14, "moderate": 9, "high": 5}
    )
    # Number of intent candidates offered, shrinking with fatigue (Sec. 1, 4.10).
    candidates_by_fatigue: Dict[str, int] = field(
        default_factory=lambda: {"low": 3, "moderate": 2, "high": 1}
    )
    # Ablation switch (Sec. 4.4 M-LLM / Sec. 4.5 "\ GAT"): generic prompt only.
    use_graph_prompt: bool = True
    # Per-user LoRA fine-tuning on accepted utterances (Sec. 3.5). Off for the
    # LLM-Only baseline, which Sec. 4.4 describes as "prompted with only
    # dialogue history and a brief user profile".
    lora_personalisation: bool = True
    # Template-backend scoring knobs (ASSUMPTION A-30, no counterpart in the
    # paper). They set how strongly retrieval outranks global statistics, and
    # therefore how often a hot decode drifts off-graph. The hallucination rate
    # is sensitive to them -- see
    # scripts/sensitivity_grounding.py.
    template_global_prior: float = 0.2
    template_grounded_gain: float = 4.0


# --------------------------------------------------------------------------- #
# Phase V: confidence-aware clinical filtering (paper Sec. 3.6, Table 2)
# --------------------------------------------------------------------------- #
@dataclass
class SafetyConfig:
    """Bayesian uncertainty gate (Sec. 3.6, Eq. 8; Sec. 3.6 calibration note)."""

    mc_passes: int = 20  # N, Table 2
    # Where the N stochastic passes of Eq. (8) run. 'scoring_head' (default) is
    # Sec. 3.6's intent-scoring head, which Table 5's latency figures assume.
    # 'lora' keeps dropout in the LLM's LoRA modules, as Sec. 3.6 said before
    # v3; it runs Eq. (8) over the LLM's token probabilities and needs the 'hf'
    # backend.
    mc_dropout_site: str = "scoring_head"  # 'scoring_head' | 'lora'
    tau_default: float = 0.15  # Sec. 3.6 "default tau = 0.15"
    tau_grid_start: float = 0.02  # Sec. 3.6 calibration grid
    tau_grid_stop: float = 0.30
    tau_grid_step: float = 0.02
    tau_init: float = 0.05  # tau_0, Sec. 3.6
    far_budget: float = 0.05  # FAR <= 5%, Sec. 3.6 / Sec. 4.3
    calibration_samples: int = 50  # "~50 low-confidence candidates", Sec. 3.6
    dropout_p: float = 0.1  # ASSUMPTION: dropout rate inside the scoring head
    # Ablation switch (Sec. 4.5 "\ Bayesian Gate").
    enabled: bool = True

    # Sec. 3.6: "select the smallest tau with FAR <= 0.05". FAR never falls as
    # tau rises, so this always returns the bottom of the grid when anything is
    # feasible. 'largest' -- the largest tau that still meets
    # the budget -- has no counterpart in the paper.
    tau_rule: str = "smallest"  # 'smallest' | 'largest'

    # --- FAR guard -------------------------------------------------------- #
    # Sec. 3.6's printed rule gates on Var(y_hat) < tau alone, and that is the
    # default. Predictive variance measures whether the MC-Dropout samples
    # agree with EACH OTHER, not whether they are right, so a scoring head that
    # is uniformly confident and uniformly wrong has near-zero variance and
    # passes every tau in the grid. Calibration then cannot bound FAR at all:
    # the budget is met at no grid point and the gate falls back to the
    # tightest tau, which still admits everything.
    #
    # 'guarded' additionally requires the predicted acceptance probability to
    # clear a calibrated floor, which makes the FAR budget reachable. It has no
    # counterpart in the paper.
    decision_rule: str = "variance_only"  # 'variance_only' | 'guarded'
    # Search grid for the confidence floor, used only when decision_rule is
    # 'guarded'. 0.0 is "no floor", i.e. the printed rule.
    confidence_floor: float = 0.0  # set by calibrate(); 0.0 until calibrated
    floor_grid_stop: float = 0.9
    floor_grid_step: float = 0.1
    # Raise instead of falling back when no setting meets the FAR budget. The
    # default records the miss and warns rather than raising, so a sweep over
    # many personas does not abort on one bad fit; the flag is on the gate as
    # ``budget_met`` and in the calibration trace either way.
    strict_far_budget: bool = False


# --------------------------------------------------------------------------- #
# Algorithm 1 (paper p. 14)
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    """Bounds enforced by Algorithm 1 (lines 9, 24)."""

    c_max: int = 2  # C_max, clarification rounds
    j_max: int = 3  # J_max, candidate presentations

    # --- retrieval floor -------------------------------------------------- #
    # Algorithm 1 generates from whatever the GAT returns, however weak the
    # match. When retrieval is at chance the top-K sub-graph is arbitrary, the
    # LLM writes a fluent utterance about the wrong entity, and the Bayesian
    # gate cannot catch it: the scoring head is confidently wrong, not
    # uncertain. The turn then burns switch activations on candidates that were
    # never going to be accepted.
    #
    # Node scores are a softmax over |V|, so K uniform nodes would hold
    # K/|V| of the mass. This floor requires the selected sub-graph to hold at
    # least ``min_retrieval_mass_ratio`` times that, which is scale-free in both
    # K and |V|. Below it, the turn goes straight to Clarification Mode -- the
    # paper's own low-effort fallback (Sec. 3.6) -- and abstains if that is
    # exhausted, rather than presenting a guess.
    #
    # DEFAULT 0.0 = DISABLED, deliberately. The mechanism is sound but the
    # threshold is not calibrated against a working retrieval stage, and a
    # miscalibrated floor would abstain everywhere and look like safety while
    # actually being breakage. 2.0-3.0 is the sensible starting range once
    # retrieval is fixed; check the abstention rate when enabling it.
    min_retrieval_mass_ratio: float = 0.0
    scan_select_seconds: float = 4.2

    @property
    def max_sact(self) -> int:
        """a <= C_max + J_max = 5 holds by construction (Algorithm 1, line 24)."""
        return self.c_max + self.j_max


# --------------------------------------------------------------------------- #
# Losses (paper Sec. 3.10, Eqs. 9-13)
# --------------------------------------------------------------------------- #
@dataclass
class LossConfig:
    lambda_contrastive: float = 0.5  # lambda_1, Eq. 9
    lambda_recon: float = 0.3  # lambda_2, Eq. 9
    lambda_intent: float = 1.0  # lambda_3, Eq. 9
    contrastive_temp: float = 0.07  # tau_c, Eq. 11
    recon_hidden: int = 256  # Sec. 3.10: 3-layer MLP, 256 hidden units
    recon_layers: int = 3


# --------------------------------------------------------------------------- #
# Training (paper Sec. 3.7, 4.9)
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    epochs: int = 100  # Sec. 4.9
    batch_size: int = 128  # Sec. 4.9
    lr: float = 1e-4  # Sec. 4.9 (AdamW, cosine decay)
    weight_decay: float = 0.01  # ASSUMPTION: AdamW default
    optimiser: str = "adamw"  # Sec. 4.9
    scheduler: str = "cosine"  # Sec. 4.9
    lora_epochs: int = 3  # Sec. 4.9 "LoRA fine-tuning converges in 3 epochs per user"
    # Sec. 3.6: accepted utterances "after a 24-hour on-device queue, fine-tune
    # the LoRA adapters".
    lora_queue_hours: float = 24.0
    # Stage 2 of Sec. 3.7 (TFT pre-training). Sec. 4.9 gives 100 epochs for
    # "the Perceiver IO and TFT", which is read here as 100 for each stage.
    tft_pretrain_epochs: int = 100
    # Stage-3 representation training: the Perceiver/TFT/GAT pass that fits the
    # retrieval head. The paper gives 100 epochs for encoder pre-training
    # (Sec. 4.9) and 3 for LoRA, but no figure for this stage, which is where
    # the GAT is actually fitted.
    #
    # This used to borrow ``lora_epochs``, i.e. 3 -- conflating "LoRA converges
    # in 3 epochs per user" with "the graph attention network converges in 3
    # epochs across 20 personas". Those are unrelated quantities, and 3 epochs
    # is almost certainly why retrieval sat at chance. ASSUMPTION: adopt the
    # paper's only stated representation-training figure, 100, and expose it.
    representation_epochs: int = 100
    grad_clip: float = 1.0  # ASSUMPTION
    device: str = "auto"  # 'auto' | 'cpu' | 'cuda' | 'mps'
    num_workers: int = 0


# --------------------------------------------------------------------------- #
# Simulation / evaluation (paper Sec. 4.1, 4.2, 4.6, 4.8)
# --------------------------------------------------------------------------- #
@dataclass
class SimulationConfig:
    """Synthetic persona suite and fatigue model (Sec. 4.1, 4.2, 4.6)."""

    n_personas: int = 20  # Sec. 4.1
    n_als: int = 10  # Sec. 4.1
    n_cerebral_palsy: int = 6  # Sec. 4.1
    n_brainstem_stroke: int = 4  # Sec. 4.1
    turns_per_session: int = 60  # ASSUMPTION: one turn per simulated minute
    session_minutes: float = 60.0  # Sec. 4.6
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Sec. 3.8: 5 reps
    # Exponential fatigue rise, Sec. 4.6 / sensitivity grid Sec. 5.5.
    fatigue_half_life_min: float = 30.0  # midpoint of the 15-60 min grid
    fatigue_peak: float = 0.8  # midpoint-ish of the 0.5-0.9 grid
    fatigue_noise: float = 0.02  # ASSUMPTION
    # Simulated-user behaviour used to score SACT/FAR without human raters.
    user_accept_threshold: float = 0.55  # ASSUMPTION: not specified by the paper
    reduced_sensor_set: bool = False  # Sec. 4.7 / 5.6: EDA + PPG + monocular gaze


@dataclass
class EvalConfig:
    words_per_turn_overhead_s: float = 1.0  # Sec. 4.2: "+ 1 s per turn"
    ece_bins: int = 10  # Sec. 4.3
    bootstrap_iters: int = 10_000  # Sec. 4.8
    holm_alpha: float = 0.05  # Sec. 4.8


@dataclass
class Config:
    """Top-level configuration."""

    name: str = "mcga-lm"
    seed: int = 1234
    inputs: InputDims = field(default_factory=InputDims)
    perceiver: PerceiverConfig = field(default_factory=PerceiverConfig)
    tft: TFTConfig = field(default_factory=TFTConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "runs"

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Config":
        return _build(cls, data or {})

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)


def _build(klass, data: Dict[str, Any]):
    """Recursively instantiate nested dataclasses from a plain dict.

    ``from __future__ import annotations`` turns every field annotation into a
    string, so the concrete types are recovered with ``get_type_hints`` rather
    than read off ``Field.type``.
    """
    if not dataclasses.is_dataclass(klass):
        return data
    hints = get_type_hints(klass)
    kwargs: Dict[str, Any] = {}
    fields = {f.name: f for f in dataclasses.fields(klass)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"unknown config keys for {klass.__name__}: {sorted(unknown)}")
    for name in fields:
        if name not in data:
            continue
        value = data[name]
        field_type = hints.get(name)
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[name] = _build(field_type, value)
        else:
            kwargs[name] = value
    return klass(**kwargs)


def merge_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Apply ``a.b.c=value`` CLI overrides to a config, returning a new object."""
    data = cfg.to_dict()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node:
                raise ValueError(f"unknown override path {key!r}")
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f"unknown override key {key!r}")
        node[parts[-1]] = yaml.safe_load(raw)
    return Config.from_dict(data)

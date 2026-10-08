"""Stage 3: representation training, scoring head, and per-user calibration.

Paper Sec. 3.7 step 3: the intent graph is initialised from a structured intake
interview and the model is personalised on accepted utterances, continuing
on-device throughout the product lifetime.

This stage fits the Perceiver/TFT/GAT against Eq. (9), trains the intent-scoring
head, calibrates tau per persona against the 5% false-acceptance budget of
Sec. 3.6, and then fine-tunes each persona's own LoRA adapter on the utterances
that persona accepted (Sec. 3.5; 3 epochs per user, Sec. 4.9). Calibration comes
first because Sec. 3.6 places it in "the initial 30-minute session", before any
LoRA update has left the 24-hour queue.

Session seeds are disjoint by role so calibration never sees training turns:
``TRAIN_SEEDS`` for representation learning, ``CALIBRATION_SEED`` for tau.

The encoder, TFT and GAT are shared across personas by default, which matches
Sec. 3.7's "no per-person fine-tuning for synthetic evaluation". tau and the LoRA
adapters are per user in either mode. ``per_persona=True`` additionally fits a
separate encoder/GAT per persona; the paper does not describe that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..data.dataset import SessionTensors, TurnEncoder
from ..data.personas import Persona
from ..inference import RETRIEVAL_DEPTH
from ..llm.prompt import DialogueTurn, build_prompt, temperature_for_fatigue
from ..losses import LossBreakdown, total_loss
from ..memory.linearise import linearise_subgraph
from ..models.gat import graph_tensors
from ..states import fatigue_level
from ..pipeline import MCGALM
from ..safety.gate import BayesianGate
from ..seed import set_seed
from ..utils import get_logger, resolve_device
from .lora import personalise as personalise_lora
from .lora import supports_lora

# One scoring example: (context, token embeddings, accepted?, rendered prompt,
# utterance). The last two feed LoRA training and LoRA-site MC Dropout.
ScoringExample = Tuple[torch.Tensor, torch.Tensor, float, str, str]

logger = get_logger(__name__)

# Disjoint session seeds (see module docstring).
TRAIN_SEEDS = (100, 101, 102)
CALIBRATION_SEED = 200


@dataclass
class TrainingReport:
    epochs: int
    history: List[Dict[str, float]] = field(default_factory=list)
    head_history: List[Dict[str, float]] = field(default_factory=list)
    tau_by_persona: Dict[str, float] = field(default_factory=dict)
    # Confidence floor chosen alongside each tau by the FAR guard.
    floor_by_persona: Dict[str, float] = field(default_factory=dict)
    tau_trace: Dict[str, list] = field(default_factory=dict)
    # Personas for which no (tau, floor) setting met the FAR budget. A non-empty
    # set means the gate cannot bound false acceptance for those users, which is
    # a safety result and must not be left to a log line.
    far_budget_missed: set = field(default_factory=set)
    per_persona: bool = False
    # Per-user LoRA fine-tuning (Sec. 3.5): pairs used, loss per epoch, or why
    # it was skipped (the template backend has no adapters).
    lora_by_persona: Dict[str, dict] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        taus = list(self.tau_by_persona.values())
        floors = list(self.floor_by_persona.values())
        n = max(len(self.tau_by_persona), 1)
        return {
            "per_persona": float(self.per_persona),
            "final_loss": self.history[-1]["total"] if self.history else float("nan"),
            "final_head_loss": self.head_history[-1]["bce"] if self.head_history else float("nan"),
            "tau_mean": float(np.mean(taus)) if taus else float("nan"),
            "tau_sd": float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0,
            "tau_min": float(np.min(taus)) if taus else float("nan"),
            "tau_max": float(np.max(taus)) if taus else float("nan"),
            "floor_mean": float(np.mean(floors)) if floors else float("nan"),
            "far_budget_missed_n": float(len(self.far_budget_missed)),
            "far_budget_missed_frac": float(len(self.far_budget_missed) / n),
        }


def build_sessions(
    personas: Sequence[Persona], cfg: Config, seeds: Sequence[int], encoder: TurnEncoder
) -> Dict[str, List[SessionTensors]]:
    """Simulate and encode training sessions for every persona."""
    out: Dict[str, List[SessionTensors]] = {}
    for persona in personas:
        sessions = []
        for s in seeds:
            turns = persona.simulate_session(seed=s)
            sessions.append(encoder.encode_session(persona, turns, persona.graph, seed=s))
        out[persona.spec.persona_id] = sessions
    return out


# --------------------------------------------------------------------------- #
# (a) representation training
# --------------------------------------------------------------------------- #
def train_representations(
    model: MCGALM,
    personas: Sequence[Persona],
    sessions: Dict[str, List[SessionTensors]],
    cfg: Config,
    device: torch.device,
    epochs: int,
    include_fatigue_loss: bool = False,
) -> List[Dict[str, float]]:
    """Optimise Eq. (9) over simulated sessions."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))
    graph_cache = {
        p.spec.persona_id: graph_tensors(p.graph, device) for p in personas
    }
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        epoch_stats: List[Dict[str, float]] = []
        for persona in personas:
            features, edge_index, edge_weight = graph_cache[persona.spec.persona_id]
            for session in sessions[persona.spec.persona_id]:
                s = session.to(device)
                z_ctx, recon = model.encode_context(s.batch, with_reconstruction=True)
                state = model.cognitive_state(z_ctx)
                query = model.build_query(z_ctx, state)
                node_scores = None
                if model.gat is not None:
                    _, node_scores = model.attend_graph(features, edge_index, query, edge_weight)
                breakdown: LossBreakdown = total_loss(
                    cfg.loss,
                    fatigue_pred=state.fatigue if include_fatigue_loss else None,
                    fatigue_target=s.fatigue if include_fatigue_loss else None,
                    z_ctx=z_ctx,
                    z_utt=_project_utterance(z_ctx, s.utterance_embeddings),
                    x_phys=s.batch.phys,
                    x_recon=recon,
                    node_scores=node_scores,
                    intent_targets=s.intent_targets if node_scores is not None else None,
                    device=device,
                )
                optimiser.zero_grad(set_to_none=True)
                breakdown.total.backward()
                nn.utils.clip_grad_norm_(params, cfg.training.grad_clip)
                optimiser.step()
                epoch_stats.append(breakdown.as_floats())
        scheduler.step()
        mean = {k: float(np.mean([d[k] for d in epoch_stats])) for k in epoch_stats[0]} if epoch_stats else {}
        mean["epoch"] = epoch
        history.append(mean)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            logger.info(
                "stage3 epoch %d/%d total=%.4f intent=%.4f contrastive=%.4f",
                epoch + 1,
                epochs,
                mean.get("total", float("nan")),
                mean.get("intent", float("nan")),
                mean.get("contrastive", float("nan")),
            )
    return history


def _project_utterance(z_ctx: torch.Tensor, utterance: torch.Tensor) -> torch.Tensor:
    """Pad/crop the pooled utterance embedding to ``z_ctx``'s width for Eq. (11).

    ASSUMPTION A-29: the paper takes ``z_utt`` from the frozen LLM's last hidden
    state, whose width matches the model. With the template backend the widths
    differ, so the smaller vector is zero-padded (a fixed, non-learned map).
    """
    d = z_ctx.shape[-1]
    if utterance.shape[-1] == d:
        return utterance
    if utterance.shape[-1] > d:
        return utterance[..., :d]
    pad = d - utterance.shape[-1]
    return F.pad(utterance, (0, pad))


# --------------------------------------------------------------------------- #
# (b) scoring-head training and (c) tau calibration
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_scoring_examples(
    model: MCGALM,
    backend,
    persona: Persona,
    session: SessionTensors,
    cfg: Config,
    device: torch.device,
    seed: int,
) -> List[ScoringExample]:
    """Roll out generation on a session and label each candidate accept/reject."""
    rng = np.random.default_rng(seed)
    features, edge_index, edge_weight = graph_tensors(persona.graph, device)
    s = session.to(device)
    z_ctx, _ = model.encode_context(s.batch)
    state = model.cognitive_state(z_ctx)
    query = model.build_query(z_ctx, state)
    node_scores = node_features = None
    if model.gat is not None:
        node_features, node_scores = model.attend_graph(features, edge_index, query, edge_weight)

    examples: List[ScoringExample] = []
    for i, turn in enumerate(session.turns):
        ids: List[int] = []
        if node_scores is not None:
            ids = model.gat.select_active_subgraph(node_scores, batch_index=i).node_ids
            ids = [j for j in ids if j < len(persona.graph.nodes)]
        names = [persona.graph.nodes[j].name for j in ids]
        level = fatigue_level(turn.context.fatigue)
        prompt = build_prompt(
            profile=persona.profile,
            state_level=level,
            graph_text=linearise_subgraph(persona.graph, ids) if ids else "",
            history=[DialogueTurn(sp, tx) for sp, tx in turn.history],
            cfg=cfg.llm,
            active_nodes=tuple(
                zip(names, [persona.graph.nodes[j].type for j in ids], [1.0] * len(ids))
            ),
        )
        candidates = backend.generate(
            prompt,
            n=RETRIEVAL_DEPTH,
            temperature=temperature_for_fatigue(turn.context.fatigue, cfg.llm),
            top_p=cfg.llm.top_p,
            rng=rng,
        )
        if not candidates:
            continue
        sub_features = node_features[i, ids, :] if (node_features is not None and ids) else None
        context = model.scoring_context(query[i : i + 1], sub_features)
        prompt_text = prompt.render()
        for cand in candidates[:2]:  # the top candidates are the ones ever shown
            tokens = torch.from_numpy(
                backend.embed_tokens(cand.text, cfg.llm.max_new_tokens)
            ).unsqueeze(0).to(device)
            label = float(persona.accepts(cand.function, cand.entities, turn.intent, rng))
            examples.append((context.detach().cpu(), tokens.detach().cpu(), label, prompt_text, cand.text))
    return examples


def train_scoring_head(
    model: MCGALM,
    examples: Sequence[ScoringExample],
    cfg: Config,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 32,
) -> List[Dict[str, float]]:
    """Binary cross-entropy training of the intent-scoring head (A-28)."""
    if not examples:
        return []
    head = model.intent_head
    optimiser = torch.optim.AdamW(head.parameters(), lr=cfg.training.lr * 5)
    history: List[Dict[str, float]] = []
    rng = np.random.default_rng(cfg.seed)
    order = np.arange(len(examples))

    for epoch in range(epochs):
        rng.shuffle(order)
        head.train()
        losses: List[float] = []
        for start in range(0, len(order), batch_size):
            chunk = [examples[i] for i in order[start : start + batch_size]]
            loss = torch.zeros((), device=device)
            for context, tokens, label, *_ in chunk:
                probs = head(context.to(device), tokens.to(device))
                target = torch.full_like(probs, float(label))
                loss = loss + F.binary_cross_entropy(probs, target)
            loss = loss / len(chunk)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "bce": float(np.mean(losses))})
    return history


@torch.no_grad()
def calibrate_thresholds(
    model: MCGALM,
    backend,
    personas: Sequence[Persona],
    cfg: Config,
    device: torch.device,
    encoder: TurnEncoder,
    rule: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, list], set]:
    """Per-persona threshold calibration (paper Sec. 3.6).

    Returns ``(taus, floors, traces, far_budget_missed)``. ``floors`` is the
    confidence floor the FAR guard picked alongside each tau, and the final
    element names the personas for which no setting met the budget at all.

    "tau is calibrated per user during the initial 30-minute session: collect
    ~50 low-confidence candidates ... compute FAR on held-out data ... select
    [the] smallest tau with FAR <= 0.05." ``rule`` defaults to
    ``SafetyConfig.tau_rule``; see safety/gate.py.

    Var(y_hat) comes from wherever ``SafetyConfig.mc_dropout_site`` says Eq. (8)
    runs, so tau is calibrated on the same quantity the gate will see.
    """
    taus: Dict[str, float] = {}
    floors: Dict[str, float] = {}
    traces: Dict[str, list] = {}
    missed: set = set()
    lora_site = cfg.safety.mc_dropout_site == "lora"
    if supports_lora(backend):
        # Calibration is the initial session, before any personal adapter
        # exists, so it runs on the shared one.
        backend.use_adapter(backend.base_adapter, create=False)
    for persona in personas:
        turns = persona.simulate_session(
            seed=CALIBRATION_SEED, n_turns=max(cfg.safety.calibration_samples // 2, 10)
        )
        session = encoder.encode_session(persona, turns, persona.graph, seed=CALIBRATION_SEED)
        examples = collect_scoring_examples(
            model, backend, persona, session, cfg, device, seed=CALIBRATION_SEED
        )
        variances: List[float] = []
        confidences: List[float] = []
        accepted: List[bool] = []
        for context, tokens, label, prompt_text, utterance in examples[: cfg.safety.calibration_samples]:
            if lora_site:
                variance, confidence = backend.mc_dropout(prompt_text, utterance, cfg.safety.mc_passes)
            else:
                estimate = model.uncertainty(context.to(device), tokens.to(device))
                variance, confidence = float(estimate.variance[0]), float(estimate.confidence[0])
            variances.append(variance)
            # The FAR guard calibrates a confidence floor alongside tau, so the
            # predicted acceptance probability has to be kept too.
            confidences.append(confidence)
            accepted.append(bool(label))
        gate = BayesianGate(cfg.safety)
        if variances:
            tau, trace = gate.calibrate(variances, accepted, rule=rule, confidences=confidences)
            if gate.budget_met is False:
                missed.add(persona.spec.persona_id)
                logger.warning(
                    "persona %s: no gate setting met the %.0f%% FAR budget",
                    persona.spec.persona_id, cfg.safety.far_budget * 100,
                )
        else:
            tau, trace = cfg.safety.tau_default, []
        taus[persona.spec.persona_id] = float(tau)
        floors[persona.spec.persona_id] = float(gate.confidence_floor)
        traces[persona.spec.persona_id] = trace
    if missed:
        logger.warning(
            "FAR budget unreachable for %d/%d personas: %s. Reported FAR for these "
            "users is not bounded by the gate.",
            len(missed), len(personas), ", ".join(sorted(missed)),
        )
    return taus, floors, traces, missed


# --------------------------------------------------------------------------- #
def train(
    cfg: Config,
    personas: Sequence[Persona],
    backend,
    epochs: Optional[int] = None,
    head_epochs: int = 5,
    pretrained: Optional[str] = None,
    out_dir: Optional[str] = None,
    include_fatigue_loss: bool = False,
    per_persona: bool = False,
) -> Tuple[Union[MCGALM, Dict[str, MCGALM]], TrainingReport]:
    """Run stage 3 end to end.

    Returns a single model, or -- with ``per_persona=True`` -- a mapping from
    persona id to that person's own model. Both are accepted by
    :func:`mcga_lm.eval.runner.run_generative_system`. Either way each persona
    gets its own tau and, when the backend has LoRA adapters and
    ``LLMConfig.lora_personalisation`` is on, its own adapter (Sec. 3.5).
    """
    if per_persona:
        return _train_per_persona(
            cfg, personas, backend, epochs, head_epochs, pretrained, out_dir, include_fatigue_loss
        )
    set_seed(cfg.seed)
    device = resolve_device(cfg.training.device)
    model = MCGALM(cfg).to(device)
    if supports_lora(backend):
        backend.use_adapter(backend.base_adapter, create=False)

    if pretrained:
        state = torch.load(pretrained, map_location=device, weights_only=False)
        try:
            model.context_encoder.load_state_dict(state["context_encoder"])
            model.tft.load_state_dict(state["tft"])
            model.decoder.load_state_dict(state["decoder"])
        except RuntimeError as exc:
            # Almost always a checkpoint pre-trained under a different config --
            # typically one produced by --quick (32x128 encoder) being loaded into
            # the default 256x512 one. Say that, rather than printing tensor
            # shapes and leaving the reader to infer it.
            raise RuntimeError(
                f"the checkpoint at {pretrained} does not match this configuration "
                f"(perceiver {cfg.perceiver.num_latents}x{cfg.perceiver.latent_dim}, "
                f"depth {cfg.perceiver.depth}; tft window {cfg.tft.window}). Pre-train "
                f"with the same config you intend to evaluate with:\n"
                f"    python scripts/pretrain_encoder.py --out runs/"
                + (f" --config <your config>" if cfg.name != "mcga-lm" else "")
                + f"\nUnderlying error: {exc}"
            ) from exc
        logger.info("loaded pre-trained encoder from %s (source=%s)", pretrained, state.get("pretraining_source"))
    else:
        logger.warning(
            "No pre-trained encoder supplied: the TFT fatigue head is randomly initialised, "
            "so the fatigue index carries no physiological meaning. Run scripts/pretrain_encoder.py first."
        )

    encoder = TurnEncoder(
        cfg.inputs, reduced_sensor_set=cfg.simulation.reduced_sensor_set, backend=backend
    )
    sessions = build_sessions(personas, cfg, TRAIN_SEEDS, encoder)
    # Stage 3 fits the GAT. It must not borrow the LoRA epoch count.
    n_epochs = epochs if epochs is not None else cfg.training.representation_epochs

    history = train_representations(
        model, personas, sessions, cfg, device, n_epochs, include_fatigue_loss
    )

    examples: List[ScoringExample] = []
    accepted_pairs: Dict[str, List[Tuple[str, str]]] = {}
    for persona in personas:
        pid = persona.spec.persona_id
        for k, session in enumerate(sessions[pid]):
            rolled = collect_scoring_examples(model, backend, persona, session, cfg, device, seed=TRAIN_SEEDS[k])
            examples.extend(rolled)
            # Sec. 3.5: adapters are trained "on the user's accepted utterances
            # paired with their prompts".
            accepted_pairs.setdefault(pid, []).extend(
                (prompt_text, utterance) for _, _, label, prompt_text, utterance in rolled if label
            )
    head_history = train_scoring_head(model, examples, cfg, device, epochs=head_epochs)

    taus, floors, traces, missed = calibrate_thresholds(model, backend, personas, cfg, device, encoder)
    lora_reports = _personalise_adapters(backend, personas, accepted_pairs, cfg)

    report = TrainingReport(
        epochs=n_epochs,
        history=history,
        head_history=head_history,
        tau_by_persona=taus,
        floor_by_persona=floors,
        far_budget_missed=missed,
        tau_trace=traces,
        per_persona=False,
        lora_by_persona=lora_reports,
    )
    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "config": cfg.to_dict(), "tau": taus, "per_persona": False},
            path / "mcga_lm.pt",
        )
        logger.info("saved model to %s", path / "mcga_lm.pt")
        _save_adapters(backend, lora_reports, path)
    return model, report


def _personalise_adapters(
    backend,
    personas: Sequence[Persona],
    accepted_pairs: Mapping[str, Sequence[Tuple[str, str]]],
    cfg: Config,
) -> Dict[str, dict]:
    """Fit each persona's LoRA adapter on its accepted utterances (Sec. 3.5, 4.9).

    Sec. 4.9: "LoRA fine-tuning converges in 3 epochs per user". The paper gives
    no LoRA learning rate, so its only stated rate (Sec. 4.9, 1e-4) is used.
    """
    reports: Dict[str, dict] = {}
    for persona in personas:
        pid = persona.spec.persona_id
        if not cfg.llm.lora_personalisation:
            reports[pid] = {"adapter": pid, "skipped": "lora_personalisation is off for this system"}
            continue
        reports[pid] = personalise_lora(
            backend,
            list(accepted_pairs.get(pid, ())),
            adapter=pid,
            epochs=cfg.training.lora_epochs,
            lr=cfg.training.lr,
        )
    if supports_lora(backend):
        backend.use_adapter(backend.base_adapter, create=False)
    return reports


def _save_adapters(backend, lora_reports: Mapping[str, dict], path: Path) -> None:
    trained = [pid for pid, r in lora_reports.items() if "history" in r]
    if trained and supports_lora(backend):
        backend.model.save_pretrained(str(path / "lora_adapters"), selected_adapters=trained)
        logger.info("saved %d LoRA adapters to %s", len(trained), path / "lora_adapters")


def _train_per_persona(
    cfg: Config,
    personas: Sequence[Persona],
    backend,
    epochs: Optional[int],
    head_epochs: int,
    pretrained: Optional[str],
    out_dir: Optional[str],
    include_fatigue_loss: bool,
) -> Tuple[Dict[str, MCGALM], TrainingReport]:
    """Fit a separate encoder/TFT/GAT per persona.

    Each persona is trained in isolation on its own sessions. The paper does not
    describe this: Sec. 3.7 gives the TFT "no per-person fine-tuning for synthetic
    evaluation", and only tau and the LoRA adapters are per user, which the
    shared path already does. Cost is linear in the number of personas.
    """
    models: Dict[str, MCGALM] = {}
    combined = TrainingReport(epochs=epochs or cfg.training.representation_epochs, per_persona=True)
    for i, persona in enumerate(personas):
        pid = persona.spec.persona_id
        logger.info("per-persona training %d/%d (%s)", i + 1, len(personas), pid)
        model, report = train(
            cfg,
            [persona],
            backend,
            epochs=epochs,
            head_epochs=head_epochs,
            pretrained=pretrained,
            out_dir=None,
            include_fatigue_loss=include_fatigue_loss,
            per_persona=False,
        )
        models[pid] = model
        combined.tau_by_persona.update(report.tau_by_persona)
        combined.floor_by_persona.update(report.floor_by_persona)
        combined.far_budget_missed |= report.far_budget_missed
        combined.tau_trace.update(report.tau_trace)
        combined.lora_by_persona.update(report.lora_by_persona)
        if report.history:
            combined.history.append({"persona": pid, **report.history[-1]})
        if report.head_history:
            combined.head_history.append({"persona": pid, **report.head_history[-1]})
    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "models": {pid: m.state_dict() for pid, m in models.items()},
                "config": cfg.to_dict(),
                "tau": combined.tau_by_persona,
                "per_persona": True,
            },
            path / "mcga_lm_per_persona.pt",
        )
        logger.info("saved %d per-persona models to %s", len(models), path / "mcga_lm_per_persona.pt")
        _save_adapters(backend, combined.lora_by_persona, path)
    return models, combined

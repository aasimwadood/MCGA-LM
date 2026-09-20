

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
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
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

    examples: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
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
        for cand in candidates[:2]:  # the top candidates are the ones ever shown
            tokens = torch.from_numpy(
                backend.embed_tokens(cand.text, cfg.llm.max_new_tokens)
            ).unsqueeze(0).to(device)
            label = float(persona.accepts(cand.function, cand.entities, turn.intent, rng))
            examples.append((context.detach().cpu(), tokens.detach().cpu(), label))
    return examples


def train_scoring_head(
    model: MCGALM,
    examples: Sequence[Tuple[torch.Tensor, torch.Tensor, float]],
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
            for context, tokens, label in chunk:
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
    rule: str = "largest",
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, list], set]:
    """Per-persona threshold calibration (paper Sec. 3.6).

    Returns ``(taus, floors, traces, far_budget_missed)``. ``floors`` is the
    confidence floor the FAR guard picked alongside each tau, and the final
    element names the personas for which no setting met the budget at all.

    "tau is calibrated per user during the initial 30-minute session: collect
    ~50 low-confidence candidates ... compute FAR on held-out data ... select
    [the] tau with FAR <= 0.05." See D-01 in safety/gate.py about ``rule``.
    """
    taus: Dict[str, float] = {}
    floors: Dict[str, float] = {}
    traces: Dict[str, list] = {}
    missed: set = set()
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
        for context, tokens, label in examples[: cfg.safety.calibration_samples]:
            estimate = model.uncertainty(context.to(device), tokens.to(device))
            variances.append(float(estimate.variance[0]))
            # The FAR guard calibrates a confidence floor alongside tau, so the
            # head's predicted acceptance probability has to be kept too.
            confidences.append(float(estimate.confidence[0]))
            accepted.append(bool(label))
        gate = BayesianGate(cfg.safety)
        if variances:
            tau, trace = gate.calibrate(variances, accepted, rule=rule, confidences=confidences)
            if gate.budget_met is False:
                missed.add(persona.spec.persona_id)
                logger.warning(
                    "persona %s: no (tau, floor) setting met the %.0f%% FAR budget",
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
    persona id to that person's own model, which is the deployment story the
    paper describes (Sec. 3.7). Both are accepted by
    :func:`mcga_lm.eval.runner.run_generative_system`.
    """
    if per_persona:
        return _train_per_persona(
            cfg, personas, backend, epochs, head_epochs, pretrained, out_dir, include_fatigue_loss
        )
    set_seed(cfg.seed)
    device = resolve_device(cfg.training.device)
    model = MCGALM(cfg).to(device)

    if pretrained:
        state = torch.load(pretrained, map_location=device, weights_only=False)
        model.context_encoder.load_state_dict(state["context_encoder"])
        model.tft.load_state_dict(state["tft"])
        model.decoder.load_state_dict(state["decoder"])
        logger.info("loaded pre-trained encoder from %s (source=%s)", pretrained, state.get("pretraining_source"))
    else:
        logger.warning(
            "No pre-trained encoder supplied: the TFT fatigue head is randomly initialised, "
            "so the fatigue index carries no physiological meaning. Run scripts/pretrain_encoder.py first."
        )

    encoder = TurnEncoder(cfg.inputs, reduced_sensor_set=cfg.simulation.reduced_sensor_set)
    sessions = build_sessions(personas, cfg, TRAIN_SEEDS, encoder)
    n_epochs = epochs if epochs is not None else cfg.training.lora_epochs

    history = train_representations(
        model, personas, sessions, cfg, device, n_epochs, include_fatigue_loss
    )

    examples: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
    for persona in personas:
        for k, session in enumerate(sessions[persona.spec.persona_id]):
            examples.extend(
                collect_scoring_examples(model, backend, persona, session, cfg, device, seed=TRAIN_SEEDS[k])
            )
    head_history = train_scoring_head(model, examples, cfg, device, epochs=head_epochs)

    taus, floors, traces, missed = calibrate_thresholds(model, backend, personas, cfg, device, encoder)

    report = TrainingReport(
        epochs=n_epochs,
        history=history,
        head_history=head_history,
        tau_by_persona=taus,
        floor_by_persona=floors,
        far_budget_missed=missed,
        tau_trace=traces,
        per_persona=False,
    )
    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "config": cfg.to_dict(), "tau": taus, "per_persona": False},
            path / "mcga_lm.pt",
        )
        logger.info("saved model to %s", path / "mcga_lm.pt")
    return model, report


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
    """Fit one model per persona (paper Sec. 3.7's on-device personalisation).

    Each persona is trained in isolation on its own sessions and gets its own
    calibrated tau. Cost is linear in the number of personas; the shared-model
    path exists for exactly that reason.
    """
    models: Dict[str, MCGALM] = {}
    combined = TrainingReport(epochs=epochs or cfg.training.lora_epochs, per_persona=True)
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
    return models, combined

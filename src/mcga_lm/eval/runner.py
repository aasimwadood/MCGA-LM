

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..baselines.intent_classifier import IntentClassifierBaseline, PhraseStore
from ..baselines.keystroke import (
    AdaptiveFrequencyRanker,
    BigramRanker,
    CompletionScanningSimulator,
    GridScanningSimulator,
    KLMParams,
)
from ..config import Config
from ..data.dataset import TurnEncoder
from ..data.personas import Persona
from ..inference import AdaptiveCommunicator, TurnResult
from ..memory.graph import IntentMemoryGraph
from ..models.gat import graph_tensors
from ..pipeline import MCGALM
from ..safety.gate import BayesianGate
from ..utils import get_logger, mean_sd, median_iqr
from . import metrics as M
from .hallucination import HallucinationDetector, summarise

logger = get_logger(__name__)

IN_DISTRIBUTION = "in_distribution"
HELD_OUT = "held_out"


@dataclass
class SystemResult:
    """Per-persona scores plus the across-persona aggregate for one system."""

    name: str
    split: str
    per_persona: Dict[str, Dict[str, float]] = field(default_factory=dict)
    aggregate: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notes: str = ""

    def column(self, metric: str) -> np.ndarray:
        """Per-persona values for one metric, in persona order."""
        return np.asarray(
            [self.per_persona[p].get(metric, np.nan) for p in sorted(self.per_persona)], dtype=float
        )

    def finalise(self) -> "SystemResult":
        if not self.per_persona:
            return self
        keys = sorted({k for v in self.per_persona.values() for k in v})
        for key in keys:
            values = [v[key] for v in self.per_persona.values() if key in v and np.isfinite(v[key])]
            self.aggregate[key] = mean_sd(values)
            if key == "sact":
                self.aggregate[key].update(median_iqr(values))
        return self


def latin_square(n: int) -> List[List[int]]:
    """Cyclic Latin square used to counterbalance conditions (Sec. 3.8).

    Recorded for provenance: the simulator is deterministic given its seeds, so
    presentation order does not change the numbers, but the paper states the
    design and the code should show it.
    """
    return [[(row + col) % n for col in range(n)] for row in range(n)]


# --------------------------------------------------------------------------- #
# Generative systems (MCGA-LM and its ablations / LLM baselines)
# --------------------------------------------------------------------------- #
def _model_for(model: Union[MCGALM, Mapping[str, MCGALM]], persona_id: str) -> MCGALM:
    """Accept either one shared model or a per-persona mapping (Sec. 3.7).

    The per-persona mapping is what :func:`mcga_lm.training.personalise.train`
    returns with ``per_persona=True``.
    """
    if isinstance(model, MCGALM):
        return model
    try:
        return model[persona_id]
    except KeyError as exc:  # pragma: no cover - configuration error
        raise KeyError(
            f"no model for persona {persona_id!r}; the mapping holds {sorted(model)}"
        ) from exc


@torch.no_grad()
def run_generative_system(
    name: str,
    cfg: Config,
    model: Union[MCGALM, Mapping[str, MCGALM]],
    backend,
    personas: Sequence[Persona],
    seeds: Sequence[int],
    taus: Optional[Dict[str, float]] = None,
    device: Optional[torch.device] = None,
    split: str = IN_DISTRIBUTION,
    intake_fraction: float = 0.25,
    adapted_fatigue: bool = True,
    fatigue_half_life: Optional[float] = None,
    fatigue_peak: Optional[float] = None,
    keep_turn_records: bool = False,
) -> Tuple[SystemResult, List[TurnResult]]:
    """Run Algorithm 1 over every persona and seed, and aggregate (Sec. 4.2)."""
    device = device or torch.device("cpu")
    encoder = TurnEncoder(cfg.inputs, reduced_sensor_set=cfg.simulation.reduced_sensor_set)
    detector = HallucinationDetector()
    result = SystemResult(name=name, split=split)
    result.notes = "per-persona models" if not isinstance(model, MCGALM) else "single shared model"
    all_turns: List[TurnResult] = []

    for persona in personas:
        persona_model = _model_for(model, persona.spec.persona_id).eval()
        tau = (taus or {}).get(persona.spec.persona_id, cfg.safety.tau_default)
        per_seed: List[Dict[str, float]] = []
        for seed in seeds:
            turns = persona.simulate_session(
                seed=seed,
                fatigue_half_life=fatigue_half_life,
                fatigue_peak=fatigue_peak,
                adapted=adapted_fatigue and cfg.tft.enabled,
            )
            # The graph the system is allowed to see at generation time.
            graph = (
                persona.graph
                if split == IN_DISTRIBUTION
                else persona.intake_graph(fraction=intake_fraction, seed=seed)
            )
            working = IntentMemoryGraph.from_dict(graph.to_dict())  # never mutate the source
            session = encoder.encode_session(persona, turns, working, seed=seed).to(device)

            z_ctx, _ = persona_model.encode_context(session.batch)
            state = persona_model.cognitive_state(z_ctx)
            query = persona_model.build_query(z_ctx, state)
            node_scores = node_features = None
            if persona_model.gat is not None:
                features, edge_index, edge_weight = graph_tensors(working, device)
                node_features, node_scores = persona_model.attend_graph(
                    features, edge_index, query, edge_weight
                )

            communicator = AdaptiveCommunicator(
                persona_model, backend, cfg, gate=BayesianGate(cfg.safety, tau=tau), device=device
            )
            rng = np.random.default_rng(seed * 7919 + persona.seed)
            turn_results: List[TurnResult] = []
            verdicts = []
            for i, turn in enumerate(turns):
                res = communicator.run_turn(
                    persona,
                    working,
                    turn,
                    query_row=query[i : i + 1],
                    node_scores=node_scores[0:1] if node_scores is not None else None,
                    node_features=node_features[0:1] if node_features is not None else None,
                    rng=rng,
                )
                turn_results.append(res)
                verdicts.append(
                    detector.classify(
                        res.entities,
                        working,
                        history=turn.history,
                        known_partners=[p for p, _ in persona.partners],
                    )
                )
            if keep_turn_records:
                all_turns.extend(turn_results)
            per_seed.append(_score_turns(turn_results, verdicts, turns, cfg))
        result.per_persona[persona.spec.persona_id] = {
            k: float(np.mean([d[k] for d in per_seed])) for k in per_seed[0]
        }
    return result.finalise(), all_turns


def _score_turns(
    results: Sequence[TurnResult], verdicts: Sequence, turns: Sequence, cfg: Config
) -> Dict[str, float]:
    """All Sec. 4.2 metrics for one persona-session."""
    halluc = summarise(verdicts)
    references = [t.intent.reference for t, r in zip(turns, results) if r.accepted and r.utterance]
    hypotheses = [r.utterance for r in results if r.accepted and r.utterance]
    confidences = [r.confidence for r in results if r.n_offered > 0]
    correct = [r.accepted for r in results if r.n_offered > 0]
    ihr3 = M.intent_hit_rate(results, 3)
    return {
        "sact": M.sact(results),
        "wpm": M.words_per_minute(results, cfg.inference.scan_select_seconds, cfg.evaluation.words_per_turn_overhead_s),
        "kspc": M.kspc_from_results(results),
        "ihr@1": M.intent_hit_rate(results, 1),
        "ihr@3": ihr3,
        "ihr@5": M.intent_hit_rate(results, 5),
        "hallucination_hard": halluc["hard_rate"],
        "hallucination_soft": halluc["soft_rate"],
        "far": M.false_acceptance_rate(results),
        "abstention": M.abstention_rate(results),
        "bleu4": M.corpus_bleu(references, hypotheses),
        "rouge_l": M.corpus_rouge_l(references, hypotheses),
        "ece": M.expected_calibration_error(confidences, correct, cfg.evaluation.ece_bins),
        "itr_intent": M.information_transfer_rate(3, ihr3, cfg.inference.scan_select_seconds),
        "mean_clarifications": float(np.mean([r.clarification_rounds for r in results])),
        "acceptance": float(np.mean([r.accepted for r in results])),
    }


# --------------------------------------------------------------------------- #
# Keystroke-level baselines (Sec. 4.3 items 1, 5, 6)
# --------------------------------------------------------------------------- #
def run_keystroke_baseline(
    name: str,
    cfg: Config,
    personas: Sequence[Persona],
    seeds: Sequence[int],
    corpus_sentences: Sequence[str],
    params: Optional[KLMParams] = None,
) -> SystemResult:
    """Simulate a non-generative baseline over the same reference utterances."""
    params = params or KLMParams()
    result = SystemResult(name=name, split=IN_DISTRIBUTION)
    # Baseline 5 is global: "no dialogue context, no personalisation" (Sec. 4.3).
    global_ranker = BigramRanker(corpus_sentences) if name == "Static-WP-bigram" else None

    for persona in personas:
        # Baseline 6 is personalised: cell ranking is "initialised per persona from
        # that persona's training-split utterances" (Sec. 4.3).
        ranker = global_ranker
        if name == "Adaptive-grid":
            ranker = AdaptiveFrequencyRanker(_persona_training_text(persona))
        per_seed: List[Dict[str, float]] = []
        for seed in seeds:
            rng = np.random.default_rng(seed * 104729 + persona.seed)
            turns = persona.simulate_session(seed=seed, adapted=False)
            if name == "TouchChat":
                sim = GridScanningSimulator(params)
            else:
                assert ranker is not None
                sim = CompletionScanningSimulator(ranker, params)
            presses = selections = seconds = chars = words = 0.0
            for turn in turns:
                out = sim.simulate_utterance(turn.intent.reference, rng)
                presses += out.presses
                selections += out.selections
                seconds += out.seconds
                chars += out.characters
                words += out.words
            n = len(turns)
            per_seed.append(
                {
                    # Table 7 marks SACT "n/r" for the two character-level
                    # baselines; only the symbol-level grid reports it.
                    "sact": presses / n if name == "TouchChat" else float("nan"),
                    "wpm": words / (seconds / 60.0) if seconds > 0 else float("nan"),
                    "kspc": selections / chars if chars else float("nan"),
                    "hallucination_hard": 0.0,  # no generation: cannot hallucinate
                    "hallucination_soft": 0.0,
                    "far": 0.0 if name == "TouchChat" else float(params.error_rate),
                    "acceptance": 1.0,
                }
            )
        result.per_persona[persona.spec.persona_id] = {
            k: float(np.nanmean([d[k] for d in per_seed])) for k in per_seed[0]
        }
    return result.finalise()


def _persona_training_text(persona) -> List[str]:
    """That persona's own training-split utterances (Sec. 4.3 baseline 6).

    Drawn from the disjoint training seeds used by stage 3, never from the
    evaluation sessions.
    """
    from ..training.personalise import TRAIN_SEEDS

    out: List[str] = []
    for seed in TRAIN_SEEDS:
        out.extend(t.intent.reference for t in persona.simulate_session(seed=seed))
    return out


# --------------------------------------------------------------------------- #
# Non-LLM intent classifier (Sec. 4.3 item 7)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_intent_classifier(
    cfg: Config,
    classifier: IntentClassifierBaseline,
    personas: Sequence[Persona],
    seeds: Sequence[int],
    device: Optional[torch.device] = None,
) -> SystemResult:
    """Intent selection with no generative component (Sec. 4.3)."""
    from ..data.taxonomy import PRAGMATIC_FUNCTIONS

    device = device or torch.device("cpu")
    classifier.eval()
    encoder = TurnEncoder(cfg.inputs, reduced_sensor_set=cfg.simulation.reduced_sensor_set)
    detector = HallucinationDetector()
    result = SystemResult(name="Non-LLM-Intent", split=IN_DISTRIBUTION)

    for persona in personas:
        store = PhraseStore(persona)
        per_seed: List[Dict[str, float]] = []
        for seed in seeds:
            rng = np.random.default_rng(seed * 31337 + persona.seed)
            turns = persona.simulate_session(seed=seed, adapted=False)
            session = encoder.encode_session(persona, turns, persona.graph, seed=seed).to(device)
            ranked = classifier.predict(session.batch, top_k=5).cpu().numpy()

            sact_values: List[int] = []
            accepted_flags: List[bool] = []
            hits: List[Optional[int]] = []
            hypotheses: List[str] = []
            references: List[str] = []
            verdicts = []
            for i, turn in enumerate(turns):
                rank = None
                for k, idx in enumerate(ranked[i][: cfg.inference.j_max], start=1):
                    if PRAGMATIC_FUNCTIONS[idx] == turn.intent.function:
                        rank = k
                        break
                hits.append(
                    next(
                        (k for k, idx in enumerate(ranked[i], start=1) if PRAGMATIC_FUNCTIONS[idx] == turn.intent.function),
                        None,
                    )
                )
                a = 0
                accepted = False
                for idx in ranked[i][: cfg.inference.j_max]:
                    a += 1
                    phrase = store.retrieve(PRAGMATIC_FUNCTIONS[idx])
                    if phrase is None:
                        continue
                    if persona.accepts(phrase.function, (phrase.entity,), turn.intent, rng):
                        accepted = True
                        hypotheses.append(phrase.text)
                        references.append(turn.intent.reference)
                        verdicts.append(
                            detector.classify(
                                (phrase.entity,),
                                persona.graph,
                                history=turn.history,
                                known_partners=[p for p, _ in persona.partners],
                            )
                        )
                        break
                sact_values.append(a)
                accepted_flags.append(accepted)
            n = len(turns)
            seconds = sum(sact_values) * cfg.inference.scan_select_seconds + n * cfg.evaluation.words_per_turn_overhead_s
            words = sum(len(h.split()) for h in hypotheses)
            halluc = summarise(verdicts) if verdicts else {"hard_rate": 0.0, "soft_rate": 0.0}
            per_seed.append(
                {
                    "sact": float(np.mean(sact_values)),
                    "wpm": words / (seconds / 60.0) if seconds > 0 else float("nan"),
                    "ihr@1": float(np.mean([h == 1 for h in hits])),
                    "ihr@3": float(np.mean([h is not None and h <= 3 for h in hits])),
                    "ihr@5": float(np.mean([h is not None and h <= 5 for h in hits])),
                    "hallucination_hard": halluc["hard_rate"],
                    "hallucination_soft": halluc["soft_rate"],
                    "far": float(1.0 - np.mean(accepted_flags)),
                    "acceptance": float(np.mean(accepted_flags)),
                    "bleu4": M.corpus_bleu(references, hypotheses),
                    "rouge_l": M.corpus_rouge_l(references, hypotheses),
                }
            )
        result.per_persona[persona.spec.persona_id] = {
            k: float(np.nanmean([d[k] for d in per_seed])) for k in per_seed[0]
        }
    return result.finalise()

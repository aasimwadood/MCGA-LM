"""Algorithm 1 -- MCGA-LM inference for one communicative turn (paper p. 14).

The control flow, the counters and the caps are transcribed line by line:

     8:  a <- 0; c <- 0                     SACT counter; clarification counter
     9:  constants C_max = 2, J_max = 3
    10:  while Var(y_hat) >= tau and c < C_max do
    11:      Clarification Mode: present top-3 intent nodes from G_active,t
    12:      wait for binary switch selection; a <- a+1; c <- c+1
    13:      refine G_active,t; re-run steps 5-7
    15:  if c = C_max then proceed with the best candidate regardless of Var
    16:  for j = 1 to J_max do
    17:      present candidate y_hat_j for accept/dismiss; a <- a+1
    18-20:  if the user accepts: update G and return (y_hat_j, a)
    23:  return ABSTAIN and surface the manual grid fallback
    24:  Note: a <= C_max + J_max = 5 holds by construction.

The bound on line 24 is architectural, and every exit path from ``run_turn``
asserts it.

Line 7 runs Eq. (8) wherever ``SafetyConfig.mc_dropout_site`` says: the
intent-scoring head (default, Table 6) or the LLM's LoRA modules (Sec. 3.6); see
ERRATA.md, E-12. Accepted utterances update the graph and, when a
:class:`~mcga_lm.training.lora.LoRAUpdateQueue` is attached, wait in the 24-hour
queue before fine-tuning the user's LoRA adapter (Sec. 3.6).

One addition has no counterpart in Algorithm 1: a retrieval floor that routes a
turn to Clarification Mode when the active sub-graph holds no more attention
mass than chance. It is disabled by default -- see
``InferenceConfig.min_retrieval_mass_ratio``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import Config
from .data.personas import Persona, PersonaTurn
from .llm.base import GeneratedUtterance
from .llm.hf_backend import check_embedding_dim
from .llm.prompt import (
    DialogueTurn,
    build_prompt,
    candidates_for_fatigue,
    temperature_for_fatigue,
)
from .memory.graph import IntentMemoryGraph, SemanticFrame
from .memory.linearise import linearise_subgraph
from .states import fatigue_level
from .pipeline import MCGALM
from .safety.gate import BayesianGate, clarification_bubbles, retrieval_mass_ok

if TYPE_CHECKING:  # training imports this module, so only for annotations
    from .training.lora import LoRAUpdateQueue

# Depth of the ranked candidate list retained for IHR@K measurement
# (Sec. 4.2 reports K = 1, 3, 5). Presentation is still capped at J_max.
RETRIEVAL_DEPTH = 5


@dataclass
class TurnResult:
    """Outcome of one communicative turn."""

    utterance: Optional[str]
    sact: int  # a -- switch activations this turn (primary endpoint, Sec. 4.2)
    accepted: bool
    abstained: bool  # Algorithm 1 line 23
    clarification_rounds: int  # c
    variance: float  # Var(y_hat), Eq. (8)
    confidence: float
    gate_accepted: bool  # did the gate pass the presented candidate?
    forced: bool  # line 15: proceeded despite Var >= tau
    function: str = ""
    entities: Tuple[str, ...] = ()
    grounded: bool = True
    active_nodes: Tuple[str, ...] = ()
    candidates: Tuple[str, ...] = ()
    fatigue: float = 0.0
    fatigue_level: str = "low"
    latency_ms: Dict[str, float] = field(default_factory=dict)
    hit_rank: Optional[int] = None  # rank of the true intent in the ranked list
    n_offered: int = 0  # how many candidates were actually presented
    # Why the turn abstained, when it did: "all candidates dismissed"
    # (Algorithm 1 line 23) or "retrieval below floor" (the retrieval
    # guard fired before anything was presented). Empty on accepted turns.
    abstain_reason: str = ""


class AdaptiveCommunicator:
    """Runs Algorithm 1 against a persona, a graph and a language backend."""

    def __init__(
        self,
        model: MCGALM,
        backend,
        cfg: Config,
        gate: Optional[BayesianGate] = None,
        device: Optional[torch.device] = None,
        lora_queue: Optional[LoRAUpdateQueue] = None,
    ) -> None:
        check_embedding_dim(backend, cfg.inputs.ling_dim)
        site = cfg.safety.mc_dropout_site
        if site not in ("scoring_head", "lora"):
            raise ValueError(f"unknown mc_dropout_site {site!r}; expected 'scoring_head' or 'lora'")
        if site == "lora" and not getattr(backend, "supports_lora", False):
            raise ValueError(
                "mc_dropout_site='lora' runs Eq. (8) through the LLM's LoRA modules, so it needs "
                "the 'hf' backend; the template backend has no weights"
            )
        self.model = model
        self.backend = backend
        self.cfg = cfg
        self.gate = gate or BayesianGate(cfg.safety)
        self.device = device or torch.device("cpu")
        # Sec. 3.6: accepted utterances wait 24 hours, then fine-tune LoRA.
        self.lora_queue = lora_queue

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def run_turn(
        self,
        persona: Persona,
        graph: IntentMemoryGraph,
        turn: PersonaTurn,
        query_row: torch.Tensor,  # (1, query_dim) -- q_t from lines 1-3
        node_scores: Optional[torch.Tensor],  # (1, |V|) -- line 4
        node_features: Optional[torch.Tensor],  # (1, |V|, K*D_h)
        rng: np.random.Generator,
        update_graph: bool = True,
        now_h: float = 0.0,  # simulation clock for the LoRA update queue
    ) -> TurnResult:
        timings: Dict[str, float] = {}
        level = fatigue_level(turn.context.fatigue)

        # --- line 4: Active Intent Sub-graph ----------------------------- #
        t0 = time.perf_counter()
        active_ids, active_names = self._active_nodes(graph, node_scores)
        timings["gat_select_ms"] = (time.perf_counter() - t0) * 1e3

        a = 0  # line 8: SACT counter
        c = 0  # line 8: clarification-round counter
        c_max, j_max = self.cfg.inference.c_max, self.cfg.inference.j_max  # line 9

        candidates, variance, confidence, prompt_text = self._propose(
            persona, graph, turn, query_row, node_features, active_ids, active_names, level, rng, timings
        )

        # --- retrieval floor ---------------------------------------------- #
        # Weak retrieval is not the same failure as high predictive variance,
        # and the gate cannot see it (the head is confidently wrong). Treat it
        # as its own trigger for Clarification Mode. Disabled by default.
        retrieval_ok = self._retrieval_ok(node_scores, active_ids)

        # --- lines 10-14: clarification loop ------------------------------ #
        forced = False
        while candidates and c < c_max and (
            not retrieval_ok or self.gate.decide(variance, confidence).accept is False
        ):
            bubbles = clarification_bubbles(active_names, active_ids, k=3)  # line 11
            picked = self._simulated_bubble_choice(bubbles, turn, graph)  # line 12
            a += 1
            c += 1
            active_ids, active_names = self._refine(graph, active_ids, active_names, picked)  # line 13
            # The user has just named a node, so the sub-graph is now grounded in
            # an explicit choice rather than in attention alone.
            retrieval_ok = True
            candidates, variance, confidence, prompt_text = self._propose(
                persona, graph, turn, query_row, node_features, active_ids, active_names, level, rng, timings
            )
        if c == c_max and not self.gate.decide(variance, confidence).accept:
            forced = True  # line 15: proceed regardless of Var

        gate_decision = self.gate.decide(variance, confidence)

        # --- lines 16-22: bounded candidate presentation ------------------ #
        hit_rank = self._hit_rank(candidates, turn)
        if not retrieval_ok:
            # Clarification could not rescue the turn. Presenting now would
            # spend up to J_max activations on candidates drawn from an
            # arbitrary sub-graph, so go straight to Algorithm 1's line-23
            # fallback and keep those activations.
            assert a <= c_max + j_max, "Algorithm 1 line 24 violated"
            return self._abstain(
                a=a, c=c, variance=variance, confidence=confidence,
                gate_accepted=gate_decision.accept or forced, forced=forced,
                offered=[], active_names=active_names, turn=turn,
                level=level, timings=timings, hit_rank=hit_rank,
                reason="retrieval below floor",
            )
        # The option set shrinks as cognitive reserves deplete (Sec. 1, 6.1),
        # bounded above by J_max (Algorithm 1 line 16).
        n_offer = max(1, min(j_max, candidates_for_fatigue(level, self.cfg.llm)))
        offered = candidates[:n_offer]
        for j, candidate in enumerate(offered, start=1):
            a += 1  # line 17
            accepted = persona.accepts(candidate.function, candidate.entities, turn.intent, rng)
            if accepted:  # lines 18-20
                if update_graph:
                    self._commit(graph, turn, candidate)
                if self.lora_queue is not None:
                    self.lora_queue.add(prompt_text, candidate.text, now_h)
                assert a <= c_max + j_max, "Algorithm 1 line 24 violated"
                return TurnResult(
                    utterance=candidate.text,
                    sact=a,
                    accepted=True,
                    abstained=False,
                    clarification_rounds=c,
                    variance=variance,
                    confidence=confidence,
                    gate_accepted=gate_decision.accept or forced,
                    forced=forced,
                    function=candidate.function,
                    entities=tuple(candidate.entities),
                    grounded=candidate.grounded,
                    active_nodes=tuple(active_names),
                    candidates=tuple(x.text for x in offered),
                    fatigue=turn.context.fatigue,
                    fatigue_level=level,
                    latency_ms=timings,
                    hit_rank=hit_rank,
                    n_offered=len(offered),
                )

        # --- line 23: abstain, hand over to the manual grid ---------------- #
        assert a <= c_max + j_max, "Algorithm 1 line 24 violated"
        return self._abstain(
            a=a, c=c, variance=variance, confidence=confidence,
            gate_accepted=gate_decision.accept or forced, forced=forced,
            offered=offered, active_names=active_names, turn=turn,
            level=level, timings=timings, hit_rank=hit_rank,
            reason="all candidates dismissed",
        )

    # ------------------------------------------------------------------ #
    def _retrieval_ok(self, node_scores, active_ids) -> bool:
        """Retrieval floor (see :func:`~mcga_lm.safety.gate.retrieval_mass_ok`).

        Disabled by default; ``InferenceConfig.min_retrieval_mass_ratio`` says why.
        """
        return retrieval_mass_ok(
            node_scores, active_ids, getattr(self.cfg.inference, "min_retrieval_mass_ratio", 0.0)
        )

    def _abstain(
        self, *, a, c, variance, confidence, gate_accepted, forced, offered,
        active_names, turn, level, timings, hit_rank, reason,
    ) -> TurnResult:
        """Algorithm 1 line 23: emit nothing and surface the manual grid."""
        first = offered[0] if offered else None
        return TurnResult(
            utterance=None,
            sact=a,
            accepted=False,
            abstained=True,
            clarification_rounds=c,
            variance=variance,
            confidence=confidence,
            gate_accepted=gate_accepted,
            forced=forced,
            function=first.function if first else "",
            entities=tuple(first.entities) if first else (),
            grounded=first.grounded if first else True,
            active_nodes=tuple(active_names),
            candidates=tuple(x.text for x in offered),
            fatigue=turn.context.fatigue,
            fatigue_level=level,
            latency_ms=timings,
            hit_rank=hit_rank,
            n_offered=len(offered),
            abstain_reason=reason,
        )

    # ------------------------------------------------------------------ #
    def _propose(
        self,
        persona: Persona,
        graph: IntentMemoryGraph,
        turn: PersonaTurn,
        query_row: torch.Tensor,
        node_features: Optional[torch.Tensor],
        active_ids: Sequence[int],
        active_names: Sequence[str],
        level: str,
        rng: np.random.Generator,
        timings: Dict[str, float],
    ) -> Tuple[List[GeneratedUtterance], float, float, str]:
        """Lines 5-7: assemble the prompt, decode once, score with MC Dropout.

        Returns ``(candidates, Var(y_hat), confidence, rendered prompt)``.
        """
        t0 = time.perf_counter()
        graph_text = linearise_subgraph(graph, active_ids) if active_ids else ""
        scores = self._active_scores(graph, active_ids)
        prompt = build_prompt(
            profile=persona.profile,
            state_level=level,
            graph_text=graph_text,
            history=[DialogueTurn(sp, tx) for sp, tx in turn.history],
            cfg=self.cfg.llm,
            active_nodes=tuple(zip(active_names, [graph.nodes[i].type for i in active_ids], scores)),
        )
        timings["prompt_ms"] = timings.get("prompt_ms", 0.0) + (time.perf_counter() - t0) * 1e3

        # Line 6: the utterance is decoded ONCE (Sec. 3.6).
        # We decode a ranked list of RETRIEVAL_DEPTH candidates so IHR@1/3/5
        # (Sec. 4.2) is measurable, but only ``n_offer`` of them are ever put in
        # front of the user -- that is what costs switch activations.
        t0 = time.perf_counter()
        candidates = self.backend.generate(
            prompt,
            n=RETRIEVAL_DEPTH,
            temperature=temperature_for_fatigue(turn.context.fatigue, self.cfg.llm),
            top_p=self.cfg.llm.top_p,
            rng=rng,
        )
        timings["generate_ms"] = timings.get("generate_ms", 0.0) + (time.perf_counter() - t0) * 1e3
        prompt_text = prompt.render()
        if not candidates:
            return [], float("inf"), 0.0, prompt_text
        candidates = sorted(candidates, key=lambda x: -x.score)

        # Line 7: N MC-Dropout passes re-score the already-decoded sequence.
        t0 = time.perf_counter()
        if self.cfg.safety.mc_dropout_site == "lora":
            # Sec. 3.6 / Eq. (8): dropout in the LLM's LoRA modules.
            variance, confidence = self.backend.mc_dropout(
                prompt_text, candidates[0].text, self.cfg.safety.mc_passes
            )
        else:
            # Table 6 / Sec. 4.10: the intent-scoring head only.
            context = self.model.scoring_context(
                query_row, self._subgraph_features(node_features, active_ids)
            )
            tokens = torch.from_numpy(
                self.backend.embed_tokens(candidates[0].text, self.cfg.llm.max_new_tokens)
            ).unsqueeze(0).to(query_row.device)
            estimate = self.model.uncertainty(context, tokens)
            variance, confidence = float(estimate.variance[0]), float(estimate.confidence[0])
        timings["mc_dropout_ms"] = timings.get("mc_dropout_ms", 0.0) + (time.perf_counter() - t0) * 1e3
        return candidates, variance, confidence, prompt_text

    # ------------------------------------------------------------------ #
    def _active_nodes(
        self, graph: IntentMemoryGraph, node_scores: Optional[torch.Tensor]
    ) -> Tuple[List[int], List[str]]:
        if node_scores is None or self.model.gat is None:
            return [], []
        sub = self.model.gat.select_active_subgraph(node_scores, batch_index=0)
        ids = [i for i in sub.node_ids if i < len(graph.nodes)]
        return ids, [graph.nodes[i].name for i in ids]

    def _active_scores(self, graph: IntentMemoryGraph, ids: Sequence[int]) -> List[float]:
        # Attention-ranked order is preserved; the numeric score shown to the
        # backend is the graph's own association weight to the user root.
        return [max(graph.neighbour_weight("User", graph.nodes[i].name), 0.1) for i in ids]

    def _subgraph_features(
        self, node_features: Optional[torch.Tensor], ids: Sequence[int]
    ) -> Optional[torch.Tensor]:
        if node_features is None or not ids:
            return None
        return node_features[0, list(ids), :]

    def _simulated_bubble_choice(self, bubbles, turn: PersonaTurn, graph: IntentMemoryGraph) -> int:
        """Line 12: the user picks one bubble with a single binary scan.

        ASSUMPTION A-22: the simulated user picks the bubble whose node matches
        the ground-truth intent if one is offered, else the first bubble (the
        scan's default landing position).
        """
        truth = {e.lower() for e in turn.intent.entities}
        for node_id in bubbles.node_ids:
            if graph.nodes[node_id].name.lower() in truth:
                return node_id
        return bubbles.node_ids[0] if bubbles.node_ids else -1

    def _refine(
        self, graph: IntentMemoryGraph, ids: Sequence[int], names: Sequence[str], picked: int
    ) -> Tuple[List[int], List[str]]:
        """Line 13: refine G_active around the chosen bubble.

        The chosen node is promoted to the head of the sub-graph and its graph
        neighbours are pulled in, which is what makes the regeneration
        lower-uncertainty (Sec. 3.6).
        """
        if picked < 0:
            return list(ids), list(names)
        neighbours = [
            e.dst if e.src == picked else e.src
            for e in graph.incident_edges([picked])
        ]
        ordered = [picked] + [n for n in neighbours if n != picked]
        ordered += [i for i in ids if i not in ordered]
        ordered = ordered[: self.cfg.graph.top_k]
        return ordered, [graph.nodes[i].name for i in ordered]

    def _commit(self, graph: IntentMemoryGraph, turn: PersonaTurn, candidate: GeneratedUtterance) -> None:
        """Line 19: accepted utterances update the intent graph (Sec. 3.4, 3.6)."""
        entity = candidate.entities[0] if candidate.entities else ""
        frame = SemanticFrame(
            predicate="expressed_as" if candidate.function.startswith("express") else "associated_with",
            obj=entity,
            obj_type=graph.nodes[graph.node_id(entity)].type if graph.has_node(entity) else "Object",
            partner=turn.context.partner_name,
            intent_nodes=(entity,) if entity else (),
        )
        graph.update_from_frame(frame)

    def _hit_rank(self, candidates: Sequence[GeneratedUtterance], turn: PersonaTurn) -> Optional[int]:
        """Rank of the true intent among the candidates -- feeds IHR@K (Sec. 4.2)."""
        truth = {e.lower() for e in turn.intent.entities}
        for i, cand in enumerate(candidates, start=1):
            if {e.lower() for e in cand.entities} & truth and cand.function == turn.intent.function:
                return i
        return None

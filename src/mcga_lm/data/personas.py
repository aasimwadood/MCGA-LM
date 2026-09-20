

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import GraphConfig, SimulationConfig
from ..llm.prompt import UserProfile
from ..memory.graph import USER_NODE, IntentMemoryGraph
from .lexicons import (
    ABSTRACT_STATES,
    ACTIVITIES,
    DIAGNOSES,
    FIRST_NAMES,
    IDIOLECT_FRAMES,
    LOCATIONS,
    OBJECTS,
    ROLES,
)
from .taxonomy import PRAGMATIC_FUNCTIONS, SLOT_TYPE

# Functions that become likelier as cognitive reserves deplete (Sec. 6.1).
FATIGUE_FUNCTIONS = ("express_fatigue", "request_position_change", "request_privacy", "express_discomfort")


@dataclass
class TurnContext:
    """Everything the encoder sees about one communicative turn."""

    minute: float
    location_id: int
    location_name: str
    partner_id: int
    partner_name: str
    partner_role: str
    time_of_day: float  # fraction of a day in [0, 1)
    noise_db: float
    fatigue: float  # ground-truth fatigue index (Sec. 4.5)
    arousal: float


@dataclass
class GroundTruthIntent:
    """What the user actually wants to say on this turn."""

    function: str
    entities: Tuple[str, ...]
    reference: str


@dataclass
class PersonaTurn:
    context: TurnContext
    intent: GroundTruthIntent
    history: Tuple[Tuple[str, str], ...] = ()  # (speaker, text)


@dataclass
class PersonaSpec:
    persona_id: str
    cohort: str  # 'ALS' | 'CP' | 'STROKE'
    diagnosis: str
    age: int
    verbosity: str
    formality: str
    people: Tuple[Tuple[str, str], ...]  # (name, role)
    idiolect: Tuple[str, ...]
    # The handful of things this person reaches for when reserves deplete.
    # Fatigue is physiologically observable, so a system that reads state should
    # be able to anticipate these -- which is the claim of Sec. 6.1.
    comfort_entities: Tuple[str, ...] = ()


class Persona:
    """A synthetic AAC user: graph, routines, profile and acceptance behaviour."""

    def __init__(
        self,
        spec: PersonaSpec,
        graph: IntentMemoryGraph,
        routines: Dict[Tuple[int, int], List[Tuple[str, str, float]]],
        sim_cfg: SimulationConfig,
        seed: int,
    ) -> None:
        self.spec = spec
        self.graph = graph
        self.routines = routines
        self.sim_cfg = sim_cfg
        self.seed = seed

    # ------------------------------------------------------------------ #
    @property
    def profile(self) -> UserProfile:
        """Component U of the structured prompt (Sec. 3.5)."""
        return UserProfile(
            persona_id=self.spec.persona_id,
            diagnosis=self.spec.diagnosis,
            age=self.spec.age,
            verbosity=self.spec.verbosity,
            formality=self.spec.formality,
            pre_disability_corpus=self.spec.cohort in ("ALS", "STROKE"),
            idiolect=self.spec.idiolect,
        )

    @property
    def partners(self) -> Tuple[Tuple[str, str], ...]:
        return self.spec.people

    # ------------------------------------------------------------- session --
    def simulate_session(
        self,
        seed: int,
        n_turns: Optional[int] = None,
        session_minutes: Optional[float] = None,
        fatigue_half_life: Optional[float] = None,
        fatigue_peak: Optional[float] = None,
        adapted: bool = False,
    ) -> List[PersonaTurn]:
        """Simulate one 60-minute conversation (Sec. 4.5)."""
        from .physiology import FatigueModel

        rng = np.random.default_rng((self.seed * 100003 + seed) % (2**32))
        n_turns = n_turns or self.sim_cfg.turns_per_session
        minutes = session_minutes or self.sim_cfg.session_minutes
        model = FatigueModel(
            half_life_min=fatigue_half_life or self.sim_cfg.fatigue_half_life_min,
            peak=fatigue_peak or self.sim_cfg.fatigue_peak,
            noise=self.sim_cfg.fatigue_noise,
        )

        turns: List[PersonaTurn] = []
        history: List[Tuple[str, str]] = []
        # Location/partner change in blocks rather than every turn.
        block_len = max(1, n_turns // 6)
        loc_id = int(rng.integers(len(LOCATIONS)))
        partner_idx = int(rng.integers(len(self.spec.people)))

        for i in range(n_turns):
            if i % block_len == 0 and i > 0:
                loc_id = int(rng.integers(len(LOCATIONS)))
                partner_idx = int(rng.integers(len(self.spec.people)))
            minute = minutes * i / max(n_turns - 1, 1)
            fatigue = float(
                model.adapted_value(minute, rng) if adapted else model.value(minute, rng)
            )
            arousal = float(np.clip(rng.beta(2.0, 5.0) + 0.3 * fatigue, 0.0, 1.0))
            partner_name, partner_role = self.spec.people[partner_idx]
            ctx = TurnContext(
                minute=minute,
                location_id=loc_id,
                location_name=LOCATIONS[loc_id],
                partner_id=partner_idx,
                partner_name=partner_name,
                partner_role=partner_role,
                time_of_day=(9.0 + minute / 60.0) / 24.0,  # sessions start at 09:00
                noise_db=float(np.clip(rng.normal(45 + 6 * loc_id / len(LOCATIONS), 4), 30, 75)),
                fatigue=fatigue,
                arousal=arousal,
            )
            intent = self._sample_intent(ctx, rng)
            turns.append(PersonaTurn(context=ctx, intent=intent, history=tuple(history[-10:])))
            history.append(("user", intent.reference))
            history.append((partner_name, self._partner_reply(intent, rng)))
        return turns

    def _sample_intent(self, ctx: TurnContext, rng: np.random.Generator) -> GroundTruthIntent:
        from .corpus import realise_reference

        # Fatigue shifts the intent distribution towards comfort/rest requests
        # (Sec. 6.1). ASSUMPTION A-19: probability equal to the fatigue index/2.
        if rng.random() < ctx.fatigue * 0.5:
            fn = str(rng.choice(FATIGUE_FUNCTIONS))
            # Drawn from this persona's own comfort repertoire rather than at
            # random: fatigue changes *what* they ask for, predictably.
            pool = [e for e in self.spec.comfort_entities if self.graph.has_node(e)]
            entity = str(rng.choice(pool)) if pool else self._entity_for(fn, rng)
        else:
            options = self.routines.get((ctx.location_id, ctx.partner_id))
            if not options:
                fn = str(rng.choice(PRAGMATIC_FUNCTIONS))
                entity = self._entity_for(fn, rng)
            else:
                weights = np.asarray([w for _, _, w in options], dtype=float)
                idx = int(rng.choice(len(options), p=weights / weights.sum()))
                fn, entity, _ = options[idx]
        return GroundTruthIntent(
            function=fn, entities=(entity,), reference=realise_reference(fn, entity, rng)
        )

    def _entity_for(self, function: str, rng: np.random.Generator) -> str:
        pool = self.nodes_of_type(SLOT_TYPE[function])
        if not pool:
            pool = self.nodes_of_type("Object") or ["water"]
        return str(rng.choice(pool))

    def _partner_reply(self, intent: GroundTruthIntent, rng: np.random.Generator) -> str:
        replies = ("Of course.", "Let me sort that.", "Right away.", "Tell me more.", "I see.")
        return str(rng.choice(replies))

    # --------------------------------------------------------------- graph --
    def nodes_of_type(self, node_type: str) -> List[str]:
        return [n.name for n in self.graph.nodes if n.type == node_type and n.name != USER_NODE]

    def intake_graph(self, fraction: float = 0.25, seed: int = 0) -> IntentMemoryGraph:
        """Cold-start graph "initialised from a structured intake interview"
        (Sec. 3.7 step 3, Sec. 6.2). Keeps people plus a fraction of the rest."""
        rng = np.random.default_rng(seed)
        keep = {USER_NODE.lower()}
        for name, _ in self.spec.people:
            keep.add(name.lower())
        others = [n.name for n in self.graph.nodes if n.name.lower() not in keep]
        n_keep = int(round(fraction * len(others)))
        for name in rng.choice(others, size=n_keep, replace=False):
            keep.add(str(name).lower())
        data = self.graph.to_dict()
        kept_nodes = [nd for nd in data["nodes"] if nd["name"].lower() in keep]
        index = {nd["name"]: i for i, nd in enumerate(kept_nodes)}
        old_names = [nd["name"] for nd in data["nodes"]]
        edges = []
        for ed in data["edges"]:
            s, d = old_names[ed["src"]], old_names[ed["dst"]]
            if s in index and d in index:
                ed = dict(ed)
                ed["src"], ed["dst"] = index[s], index[d]
                edges.append(ed)
        data["nodes"], data["edges"] = kept_nodes, edges
        return IntentMemoryGraph.from_dict(data)

    # ---------------------------------------------------------- acceptance --
    def accepts(
        self,
        function: str,
        entities: Sequence[str],
        intent: GroundTruthIntent,
        rng: np.random.Generator,
    ) -> bool:
        """Simulated user accept/reject (Sec. 3.6; ASSUMPTION A-17).

        Exact intent match (function *and* entity) is accepted. An entity match
        with the wrong pragmatic function is accepted with probability 0.30,
        modelling a user who tolerates a near-miss paraphrase rather than paying
        another switch activation. Anything else is rejected.
        """
        ents = {e.lower() for e in entities}
        truth = {e.lower() for e in intent.entities}
        entity_hit = bool(ents & truth)
        if entity_hit and function == intent.function:
            return True
        if entity_hit:
            return bool(rng.random() < 0.30)
        return False


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def build_persona(
    index: int, cohort: str, sim_cfg: SimulationConfig, graph_cfg: GraphConfig, seed: int = 0
) -> Persona:
    """Build one persona and its ~400-node graph (Sec. 4.1)."""
    rng = np.random.default_rng((seed * 7919 + index * 104729) % (2**32))
    persona_id = f"P{index:02d}"

    # --- people -----------------------------------------------------------
    n_people = int(rng.integers(6, 9))
    names = rng.choice(FIRST_NAMES, size=n_people, replace=False)
    roles = list(ROLES[:n_people])
    people = tuple((str(n), r) for n, r in zip(names, roles))

    diagnosis = str(rng.choice(DIAGNOSES[cohort]))
    age = int(rng.integers(52, 75)) if cohort in ("ALS", "STROKE") else int(rng.integers(19, 45))
    idiolect = tuple(rng.choice(IDIOLECT_FRAMES, size=3, replace=False).tolist())

    spec = PersonaSpec(
        persona_id=persona_id,
        cohort=cohort,
        diagnosis=diagnosis,
        age=age,
        verbosity=str(rng.choice(("low", "medium", "high"))),
        formality=str(rng.choice(("informal", "neutral", "formal"))),
        people=people,
        idiolect=idiolect,
    )

    graph = _build_graph(spec, rng, graph_cfg)
    comfort_pool = [n.name for n in graph.nodes if n.type in ("Object", "AbstractState")]
    spec.comfort_entities = tuple(
        str(x) for x in rng.choice(comfort_pool, size=min(6, len(comfort_pool)), replace=False)
    )
    routines = _build_routines(spec, graph, rng)
    return Persona(spec=spec, graph=graph, routines=routines, sim_cfg=sim_cfg, seed=seed + index)


def _build_graph(spec: PersonaSpec, rng: np.random.Generator, cfg: GraphConfig) -> IntentMemoryGraph:
    """Assemble a dense personal graph of roughly ``cfg.target_nodes`` nodes."""
    g = IntentMemoryGraph(
        node_dim=cfg.node_dim,
        half_life_days=cfg.half_life_days,
        prune_threshold=cfg.prune_threshold,
        max_nodes=cfg.max_nodes,
    )
    user = g.node_id(USER_NODE)

    # People: strongly linked to the user (Sec. 3.4 'interacts_with').
    person_ids = []
    for name, role in spec.people:
        pid = g.add_node(name, "Person")
        person_ids.append(pid)
        g.add_edge(user, pid, "interacts_with", weight=float(rng.uniform(0.7, 0.98)))
        role_node = g.add_node(role, "Activity")
        g.add_edge(pid, role_node, "associated_with", weight=float(rng.uniform(0.6, 0.95)))

    # Places (modelled as Object nodes so 'located_in' has a target).
    place_ids = [g.add_node(loc, "Object") for loc in LOCATIONS]

    # Objects and activities: persona-specific subsets so graphs differ.
    n_obj = min(170, len(OBJECTS))
    n_act = min(150, len(ACTIVITIES))
    objects = rng.choice(OBJECTS, size=n_obj, replace=False)
    activities = rng.choice(ACTIVITIES, size=n_act, replace=False)

    object_ids = []
    for name in objects:
        oid = g.add_node(str(name), "Object")
        object_ids.append(oid)
        g.add_edge(oid, int(rng.choice(place_ids)), "located_in", weight=float(rng.uniform(0.3, 0.9)))
        if rng.random() < 0.55:
            g.add_edge(user, oid, "associated_with", weight=float(rng.uniform(0.2, 0.85)))

    activity_ids = []
    for name in activities:
        aid = g.add_node(str(name), "Activity")
        activity_ids.append(aid)
        g.add_edge(user, aid, "associated_with", weight=float(rng.uniform(0.2, 0.9)))
        if rng.random() < 0.6:
            g.add_edge(aid, int(rng.choice(object_ids)), "associated_with", weight=float(rng.uniform(0.2, 0.8)))
        if rng.random() < 0.4:
            g.add_edge(int(rng.choice(activity_ids)), aid, "precedes", weight=float(rng.uniform(0.2, 0.7)))

    # Abstract states, with causal links from activities/objects.
    for name in ABSTRACT_STATES:
        sid = g.add_node(name, "AbstractState")
        g.add_edge(user, sid, "expressed_as", weight=float(rng.uniform(0.2, 0.9)))
        for _ in range(int(rng.integers(1, 4))):
            g.add_edge(int(rng.choice(activity_ids)), sid, "causes", weight=float(rng.uniform(0.2, 0.8)))

    # Persona-unique compound nodes (visits, photographs, weekday routines).
    for name, _ in spec.people:
        vid = g.add_node(f"{name}_visit", "Activity")
        g.add_edge(g.node_id(name), vid, "associated_with", weight=float(rng.uniform(0.5, 0.95)))
        pid = g.add_node(f"{name}_photo", "Object")
        g.add_edge(g.node_id(name), pid, "associated_with", weight=float(rng.uniform(0.4, 0.9)))
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        rid = g.add_node(f"{day}_routine", "Activity")
        g.add_edge(user, rid, "associated_with", weight=float(rng.uniform(0.3, 0.8)))
        g.add_edge(rid, int(rng.choice(activity_ids)), "precedes", weight=float(rng.uniform(0.3, 0.8)))
    return g


def _build_routines(
    spec: PersonaSpec, graph: IntentMemoryGraph, rng: np.random.Generator
) -> Dict[Tuple[int, int], List[Tuple[str, str, float]]]:
    """Per-(location, partner) intent distributions -- the "routines" of Sec. 4.1.

    This is the structure the GAT has to learn: the same user wants different
    things in the bedroom at night with a caregiver than in the garden with a
    friend (Sec. 6.1's "bedroom-evening thirst request versus a casual
    kitchen-noon remark").
    """
    by_type = {
        t: [n.name for n in graph.nodes if n.type == t and n.name != USER_NODE]
        for t in ("Person", "Object", "Activity", "AbstractState")
    }
    routines: Dict[Tuple[int, int], List[Tuple[str, str, float]]] = {}
    for loc in range(len(LOCATIONS)):
        for partner in range(len(spec.people)):
            n_topics = int(rng.integers(3, 7))
            topics: List[Tuple[str, str, float]] = []
            for _ in range(n_topics):
                fn = str(rng.choice(PRAGMATIC_FUNCTIONS))
                pool = by_type[SLOT_TYPE[fn]]
                if not pool:
                    continue
                entity = str(rng.choice(pool))
                topics.append((fn, entity, float(rng.uniform(0.5, 1.5))))
            if topics:
                routines[(loc, partner)] = topics
    return routines


def build_persona_suite(
    sim_cfg: SimulationConfig, graph_cfg: GraphConfig, seed: int = 0
) -> List[Persona]:
    """The 20-persona suite of Sec. 4.1 (10 ALS, 6 CP, 4 brainstem stroke)."""
    cohorts = (
        ["ALS"] * sim_cfg.n_als
        + ["CP"] * sim_cfg.n_cerebral_palsy
        + ["STROKE"] * sim_cfg.n_brainstem_stroke
    )
    if len(cohorts) != sim_cfg.n_personas:
        raise ValueError(
            f"cohort counts ({len(cohorts)}) must sum to n_personas ({sim_cfg.n_personas})"
        )
    return [build_persona(i, c, sim_cfg, graph_cfg, seed=seed) for i, c in enumerate(cohorts)]

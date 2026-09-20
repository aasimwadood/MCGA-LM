

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Sec. 3.4: the four node categories.
NODE_TYPES = ("Person", "Object", "Activity", "AbstractState")
# Sec. 3.4: the relation set R.
RELATIONS = (
    "interacts_with",
    "located_in",
    "causes",
    "associated_with",
    "precedes",
    "expressed_as",
)

# The graph always carries a distinguished root standing for the AAC user; the
# linearised prompt in Sec. 3.5 begins "User is connected to 'pain' [0.91]".
USER_NODE = "User"


def embed_lexeme(name: str, dim: int = 300, glove: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """300-d node embedding (Sec. 3.4).

    Uses GloVe when a vector table is provided, otherwise a deterministic
    hash-seeded Gaussian so runs are reproducible without downloading vectors.
    """
    key = name.strip().lower()
    if glove is not None and key in glove:
        vec = np.asarray(glove[key], dtype=np.float32)
        if vec.shape[0] == dim:
            return vec
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


@dataclass
class Node:
    name: str
    type: str
    created_day: float = 0.0
    last_seen_day: float = 0.0
    count: int = 1

    def __post_init__(self) -> None:
        if self.type not in NODE_TYPES:
            raise ValueError(f"unknown node type {self.type!r}; expected one of {NODE_TYPES}")


@dataclass
class Edge:
    src: int
    dst: int
    relation: str
    weight: float = 0.5
    last_update_day: float = 0.0
    count: int = 1

    def __post_init__(self) -> None:
        if self.relation not in RELATIONS:
            raise ValueError(f"unknown relation {self.relation!r}; expected one of {RELATIONS}")
        self.weight = float(np.clip(self.weight, 0.0, 1.0))


@dataclass
class SemanticFrame:
    """What the "lightweight dependency parser" of Sec. 3.4 extracts.

    ASSUMPTION A-05: the paper does not name the parser. In this repository the
    frame is produced either by the persona simulator (which knows the ground
    truth) or by the rule-based extractor in ``memory.frames``.
    """

    subject: str = USER_NODE
    subject_type: str = "Person"
    predicate: str = "associated_with"
    obj: str = ""
    obj_type: str = "Object"
    partner: Optional[str] = None
    intent_nodes: Sequence[str] = field(default_factory=tuple)


class IntentMemoryGraph:
    """The user's evolving intent memory graph (paper Sec. 3.4)."""

    def __init__(
        self,
        node_dim: int = 300,
        half_life_days: float = 30.0,
        prune_threshold: float = 0.05,
        max_nodes: int = 500,
        glove: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.node_dim = node_dim
        self.half_life_days = half_life_days
        self.prune_threshold = prune_threshold
        self.max_nodes = max_nodes
        self._glove = glove
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self._index: Dict[str, int] = {}
        self._embeddings: List[np.ndarray] = []
        self.now_day: float = 0.0
        self.add_node(USER_NODE, "Person")

    # -------------------------------------------------------------- nodes --
    def __len__(self) -> int:
        return len(self.nodes)

    def has_node(self, name: str) -> bool:
        return name.strip().lower() in self._index

    def node_id(self, name: str) -> int:
        return self._index[name.strip().lower()]

    def add_node(self, name: str, node_type: str = "Object") -> int:
        key = name.strip().lower()
        if key in self._index:
            idx = self._index[key]
            self.nodes[idx].count += 1
            self.nodes[idx].last_seen_day = self.now_day
            return idx
        idx = len(self.nodes)
        self.nodes.append(Node(name=name, type=node_type, created_day=self.now_day, last_seen_day=self.now_day))
        self._embeddings.append(embed_lexeme(name, self.node_dim, self._glove))
        self._index[key] = idx
        return idx

    def node_names(self) -> List[str]:
        return [n.name for n in self.nodes]

    def embeddings(self) -> np.ndarray:
        """(N, 300) node embedding matrix."""
        if not self._embeddings:
            return np.zeros((0, self.node_dim), dtype=np.float32)
        return np.stack(self._embeddings).astype(np.float32)

    # -------------------------------------------------------------- edges --
    def find_edge(self, src: int, dst: int, relation: str) -> Optional[int]:
        for i, e in enumerate(self.edges):
            if e.src == src and e.dst == dst and e.relation == relation:
                return i
        return None

    def add_edge(self, src: int, dst: int, relation: str, weight: float = 0.5) -> int:
        existing = self.find_edge(src, dst, relation)
        if existing is not None:
            self.reinforce(existing, amount=weight)
            return existing
        self.edges.append(
            Edge(src=src, dst=dst, relation=relation, weight=weight, last_update_day=self.now_day)
        )
        return len(self.edges) - 1

    def reinforce(self, edge_idx: int, amount: float = 0.2) -> None:
        """Increment an existing edge weight (Sec. 3.4).

        ASSUMPTION A-06: the increment rule is not given. We use a saturating
        update ``w <- w + amount * (1 - w)`` so weights stay in [0, 1] and
        repeated reinforcement shows diminishing returns.
        """
        e = self.edges[edge_idx]
        e.weight = float(np.clip(e.weight + amount * (1.0 - e.weight), 0.0, 1.0))
        e.last_update_day = self.now_day
        e.count += 1

    def decay_to(self, day: float) -> None:
        """Apply the 30-day half-life decay up to ``day`` (Sec. 3.4)."""
        if day < self.now_day:
            raise ValueError("time cannot run backwards")
        for e in self.edges:
            elapsed = day - e.last_update_day
            if elapsed > 0:
                e.weight *= 0.5 ** (elapsed / self.half_life_days)
                e.last_update_day = day
        self.now_day = day

    def prune(self) -> int:
        """Remove edges below the pruning threshold (Sec. 3.4, Sec. 4.10).

        Nodes left with no incident edges (other than the user root) are dropped
        too, which is what bounds the graph at 200-500 nodes.
        """
        before = len(self.edges)
        self.edges = [e for e in self.edges if e.weight >= self.prune_threshold]
        if before != len(self.edges):
            self._drop_isolated_nodes()
        return before - len(self.edges)

    def _drop_isolated_nodes(self) -> None:
        connected = {0}  # keep the User root
        for e in self.edges:
            connected.add(e.src)
            connected.add(e.dst)
        if len(connected) == len(self.nodes):
            return
        keep = sorted(connected)
        remap = {old: new for new, old in enumerate(keep)}
        self.nodes = [self.nodes[i] for i in keep]
        self._embeddings = [self._embeddings[i] for i in keep]
        self._index = {n.name.strip().lower(): i for i, n in enumerate(self.nodes)}
        for e in self.edges:
            e.src = remap[e.src]
            e.dst = remap[e.dst]

    # ------------------------------------------------------------- updates --
    def update_from_frame(self, frame: SemanticFrame, day: Optional[float] = None) -> None:
        """Fold an accepted utterance's semantic frame into the graph (Sec. 3.4)."""
        if day is not None:
            self.decay_to(day)
        src = self.add_node(frame.subject, frame.subject_type)
        if frame.obj:
            dst = self.add_node(frame.obj, frame.obj_type)
            self.add_edge(src, dst, frame.predicate, weight=0.5)
        if frame.partner:
            partner = self.add_node(frame.partner, "Person")
            self.add_edge(self.node_id(USER_NODE), partner, "interacts_with", weight=0.5)
            if frame.obj:
                self.add_edge(partner, self.node_id(frame.obj), "associated_with", weight=0.3)
        for name in frame.intent_nodes:
            if self.has_node(name):
                idx = self.node_id(name)
            else:
                idx = self.add_node(name, "AbstractState")
            self.add_edge(self.node_id(USER_NODE), idx, "expressed_as", weight=0.4)
        self.prune()

    # ---------------------------------------------------------- structure --
    def adjacency(self) -> Dict[int, List[Tuple[int, float, str]]]:
        """Neighbourhoods ``N_i`` used by the GAT (Eq. 6-7).

        Self-loops are included so isolated nodes still attend to themselves.
        """
        adj: Dict[int, List[Tuple[int, float, str]]] = {i: [(i, 1.0, "associated_with")] for i in range(len(self.nodes))}
        for e in self.edges:
            adj[e.src].append((e.dst, e.weight, e.relation))
            adj[e.dst].append((e.src, e.weight, e.relation))  # undirected message passing
        return adj

    def edge_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """Dense COO edge index + weights, symmetrised, with self-loops."""
        src: List[int] = []
        dst: List[int] = []
        w: List[float] = []
        for i in range(len(self.nodes)):
            src.append(i)
            dst.append(i)
            w.append(1.0)
        for e in self.edges:
            src.extend([e.src, e.dst])
            dst.extend([e.dst, e.src])
            w.extend([e.weight, e.weight])
        return np.asarray([src, dst], dtype=np.int64), np.asarray(w, dtype=np.float32)

    def incident_edges(self, node_ids: Iterable[int]) -> List[Edge]:
        wanted = set(node_ids)
        return [e for e in self.edges if e.src in wanted or e.dst in wanted]

    def neighbour_weight(self, a: str, b: str) -> float:
        if not (self.has_node(a) and self.has_node(b)):
            return 0.0
        i, j = self.node_id(a), self.node_id(b)
        best = 0.0
        for e in self.edges:
            if {e.src, e.dst} == {i, j}:
                best = max(best, e.weight)
        return best

    # ------------------------------------------------------ serialisation --
    def to_dict(self) -> Dict:
        return {
            "node_dim": self.node_dim,
            "half_life_days": self.half_life_days,
            "prune_threshold": self.prune_threshold,
            "max_nodes": self.max_nodes,
            "now_day": self.now_day,
            "nodes": [vars(n) for n in self.nodes],
            "edges": [vars(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Dict, glove: Optional[Dict[str, np.ndarray]] = None) -> "IntentMemoryGraph":
        g = cls(
            node_dim=data.get("node_dim", 300),
            half_life_days=data.get("half_life_days", 30.0),
            prune_threshold=data.get("prune_threshold", 0.05),
            max_nodes=data.get("max_nodes", 500),
            glove=glove,
        )
        g.nodes = []
        g._embeddings = []
        g._index = {}
        for nd in data["nodes"]:
            g.nodes.append(Node(**nd))
            g._embeddings.append(embed_lexeme(nd["name"], g.node_dim, glove))
            g._index[nd["name"].strip().lower()] = len(g.nodes) - 1
        g.edges = [Edge(**ed) for ed in data["edges"]]
        g.now_day = data.get("now_day", 0.0)
        return g

    def save(self, path: str | Path) -> None:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str | Path, glove: Optional[Dict[str, np.ndarray]] = None) -> "IntentMemoryGraph":
        import json

        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh), glove=glove)

    # ------------------------------------------------------------ metrics --
    def storage_bytes(self) -> int:
        """Storage estimate of Sec. 4.10: ``|V| . d . 4`` bytes + edge metadata."""
        node_bytes = len(self.nodes) * self.node_dim * 4
        edge_bytes = len(self.edges) * 24  # src, dst, weight, count, day (ASSUMPTION)
        return node_bytes + edge_bytes

    def summary(self) -> Dict[str, float]:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "mean_weight": float(np.mean([e.weight for e in self.edges])) if self.edges else 0.0,
            "storage_mb": self.storage_bytes() / 1e6,
        }

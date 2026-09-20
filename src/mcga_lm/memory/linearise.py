
from __future__ import annotations

from typing import List, Sequence

from .graph import IntentMemoryGraph


def linearise_subgraph(
    graph: IntentMemoryGraph, node_ids: Sequence[int], max_edges: int = 12
) -> str:
    """Render ``G_active,t`` as the textual summary ``T_graph`` of Sec. 3.5."""
    wanted = list(dict.fromkeys(int(i) for i in node_ids))
    if not wanted:
        return "No personal context available."
    edges = graph.incident_edges(wanted)
    edges = sorted(edges, key=lambda e: e.weight, reverse=True)[:max_edges]
    clauses: List[str] = []
    for e in edges:
        src = graph.nodes[e.src].name
        dst = graph.nodes[e.dst].name
        verb = e.relation.replace("_", " ")
        clauses.append(f"{src} {verb} '{dst}' [{e.weight:.2f}]")
    if not clauses:
        names = ", ".join(f"'{graph.nodes[i].name}'" for i in wanted)
        return f"Salient concepts: {names}."
    return "; ".join(clauses) + "."


def active_node_names(graph: IntentMemoryGraph, node_ids: Sequence[int]) -> List[str]:
    return [graph.nodes[int(i)].name for i in node_ids]

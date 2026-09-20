

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GraphConfig


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Numerically stable softmax over groups given by ``index``.

    scores: (B, E) edge scores; index: (E,) target node of each edge.
    """
    b = scores.shape[0]
    expanded = index.unsqueeze(0).expand(b, -1)
    # ``scatter_reduce`` rather than ``index_reduce``: the latter has no MPS
    # kernel as of torch 2.9, and this keeps the module device-agnostic.
    max_per_segment = scores.new_full((b, num_segments), float("-inf")).scatter_reduce(
        1, expanded, scores, reduce="amax", include_self=True
    )
    shifted = scores - max_per_segment.gather(1, expanded)
    exp = shifted.exp()
    denom = torch.zeros(b, num_segments, device=scores.device, dtype=scores.dtype)
    denom.index_add_(1, index, exp)
    return exp / (denom.gather(1, expanded) + 1e-16)


@dataclass
class ActiveSubgraph:
    """``G_active,t`` -- the system's belief over the user's pragmatic intents."""

    node_ids: List[int]  # top-K_top node indices
    node_scores: torch.Tensor  # (N,) normalised aggregated attention (y_hat of Eq. 13)
    node_features: Optional[torch.Tensor] = None  # (N, K * D_h) from Eq. (6)

    def top_scores(self) -> List[float]:
        return [float(self.node_scores[i]) for i in self.node_ids]


class GraphAttentionIntentMemory(nn.Module):
    """Query-conditioned multi-head GAT over the intent graph (Sec. 3.4)."""

    def __init__(self, query_dim: int, cfg: GraphConfig) -> None:
        super().__init__()
        self.cfg = cfg
        k, dh = cfg.gat_heads, cfg.gat_hidden
        self.heads, self.head_dim = k, dh
        self.w_query = nn.Linear(query_dim, k * dh, bias=False)  # W_q^{(k)}
        self.w_node = nn.Linear(cfg.node_dim, k * dh, bias=False)  # W_h^{(k)}
        # a^{(k)} in R^{3 D_h}, split into its query / self / neighbour parts.
        self.attn_q = nn.Parameter(torch.empty(k, dh))
        self.attn_i = nn.Parameter(torch.empty(k, dh))
        self.attn_j = nn.Parameter(torch.empty(k, dh))
        for p in (self.attn_q, self.attn_i, self.attn_j):
            nn.init.xavier_uniform_(p)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.attention_form not in ("paper", "query_gated"):
            raise ValueError(
                f"attention_form must be 'paper' or 'query_gated', got {cfg.attention_form!r}"
            )
        if cfg.node_scoring not in ("query", "incoming"):
            raise ValueError(f"node_scoring must be 'query' or 'incoming', got {cfg.node_scoring!r}")

    # ------------------------------------------------------------------ #
    def forward(
        self,
        node_features: torch.Tensor,  # (N, node_dim)
        edge_index: torch.Tensor,  # (2, E) rows are [source j, target i]
        query: torch.Tensor,  # (B, query_dim)
        edge_weight: Optional[torch.Tensor] = None,  # (E,) association strengths
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(h', node_scores)`` with shapes (B, N, K*D_h) and (B, N)."""
        n = node_features.shape[0]
        b = query.shape[0]
        k, dh = self.heads, self.head_dim

        wh = self.w_node(node_features).view(n, k, dh)  # (N, K, D_h)
        wq = self.w_query(query).view(b, k, dh)  # (B, K, D_h)

        src, dst = edge_index[0], edge_index[1]  # j, i
        # a . [W_q q || W_h h_i || W_h h_j] decomposes into three dot products.
        q_term = (wq * self.attn_q.unsqueeze(0)).sum(-1)  # (B, K)
        i_term = (wh * self.attn_i.unsqueeze(0)).sum(-1)  # (N, K)
        j_term = (wh * self.attn_j.unsqueeze(0)).sum(-1)  # (N, K)

        # (B, K, E) -- Eq. (7) as printed: three additive dot products.
        logits = (
            q_term.unsqueeze(-1) + i_term[dst].permute(1, 0).unsqueeze(0) + j_term[src].permute(1, 0).unsqueeze(0)
        )
        if self.cfg.attention_form == "query_gated":
            # Multiplicative query-neighbour interaction (see D-06 above). Scaled
            # like dot-product attention so the two terms stay commensurate.
            logits = logits + torch.einsum("bkd,ekd->bke", wq, wh[src]) / math.sqrt(dh)
        logits = F.leaky_relu(logits, negative_slope=self.cfg.negative_slope)
        if edge_weight is not None:
            # Association strength / recency biases attention towards strong
            # edges (Sec. 3.4 "weights reflecting association strength and
            # recency"). ASSUMPTION A-09: added as a log-weight bias.
            logits = logits + torch.log(edge_weight.clamp_min(1e-6)).view(1, 1, -1)

        flat = logits.reshape(b * k, -1)
        alpha = segment_softmax(flat, dst, n).reshape(b, k, -1)  # Eq. (7)
        alpha = self.dropout(alpha)

        # Eq. (6): message passing then per-head concatenation.
        messages = alpha.unsqueeze(-1) * wh[src].permute(1, 0, 2).unsqueeze(0)  # (B, K, E, D_h)
        out = torch.zeros(b, k, n, dh, device=node_features.device, dtype=messages.dtype)
        out.index_add_(2, dst, messages)
        out = torch.sigmoid(out)  # sigma in Eq. (6)
        h_prime = out.permute(0, 2, 1, 3).reshape(b, n, k * dh)

        node_scores = self._node_scores(wq, wh, alpha, src, n)
        return h_prime, node_scores

    def _node_scores(
        self, wq: torch.Tensor, wh: torch.Tensor, alpha: torch.Tensor, src: torch.Tensor, n: int
    ) -> torch.Tensor:
        """Aggregated attention per node -- the ``y_hat`` of Eq. (13). See D-07."""
        if self.cfg.node_scoring == "incoming":
            received = torch.zeros(alpha.shape[0], self.heads, n, device=wh.device, dtype=alpha.dtype)
            received.index_add_(2, src, alpha)
            scores = received.mean(dim=1)
            return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        # Query-to-node attention: how strongly q_t reaches for each node.
        logits = torch.einsum("bkd,nkd->bkn", wq, wh) / math.sqrt(self.head_dim)
        return torch.softmax(logits.mean(dim=1), dim=-1)

    # ------------------------------------------------------------------ #
    def select_active_subgraph(
        self, node_scores: torch.Tensor, top_k: Optional[int] = None, batch_index: int = 0
    ) -> ActiveSubgraph:
        """Top-``K_top`` nodes by aggregated attention (Sec. 3.4)."""
        top_k = top_k or self.cfg.top_k
        scores = node_scores[batch_index]
        k = min(top_k, scores.shape[0])
        ids = torch.topk(scores, k=k).indices.tolist()
        return ActiveSubgraph(node_ids=ids, node_scores=scores.detach())


def graph_tensors(graph, device=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an :class:`~mcga_lm.memory.graph.IntentMemoryGraph` to tensors."""
    import numpy as np

    ei, ew = graph.edge_index()
    features = torch.from_numpy(np.ascontiguousarray(graph.embeddings()))
    edge_index = torch.from_numpy(ei)
    edge_weight = torch.from_numpy(ew)
    if device is not None:
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
    return features, edge_index, edge_weight

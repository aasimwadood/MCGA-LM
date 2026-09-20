

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import Config
from .data.dataset import sliding_windows
from .models.gat import ActiveSubgraph, GraphAttentionIntentMemory
from .models.intent_head import IntentScoringHead, PhysiologyDecoder, UncertaintyEstimate
from .models.perceiver_io import MultimodalBatch, build_context_encoder
from .models.tft import CognitiveState, TemporalFusionTransformer


@dataclass
class PipelineOutput:
    """Everything one forward pass over a session produces."""

    z_ctx: torch.Tensor  # (N, D) contextual embeddings, Sec. 3.2
    state: CognitiveState  # s_cog,t for every turn, Sec. 3.3
    query: torch.Tensor  # q_t = [z_ctx ; s_cog], Sec. 3.4
    node_scores: Optional[torch.Tensor] = None  # (N, |V|) y_hat of Eq. (13)
    node_features: Optional[torch.Tensor] = None  # (N, |V|, K*D_h) from Eq. (6)
    reconstruction: Optional[torch.Tensor] = None  # Decoder(Z) for Eq. (12)


class MCGALM(nn.Module):
    """Perceiver IO -> TFT -> GAT -> intent-scoring head (paper Sec. 3.2-3.6)."""

    def __init__(self, cfg: Config, ablate_perceiver: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        # Phase I (Sec. 3.2, Eqs. 2-4)
        self.context_encoder = build_context_encoder(cfg.inputs, cfg.perceiver, ablate_perceiver)
        # Phase II (Sec. 3.3, Eq. 5)
        self.tft = TemporalFusionTransformer(self.context_encoder.output_dim, cfg.tft)
        # Phase III (Sec. 3.4, Eqs. 6-7)
        self.query_dim = self.context_encoder.output_dim + cfg.tft.state_dim
        self.gat = GraphAttentionIntentMemory(self.query_dim, cfg.graph) if cfg.graph.enabled else None
        # Phase V scoring head (Sec. 3.6, Eq. 8)
        graph_feat = cfg.graph.gat_heads * cfg.graph.gat_hidden if cfg.graph.enabled else 0
        self.intent_head = IntentScoringHead(
            context_dim=self.query_dim + graph_feat,
            token_dim=cfg.inputs.ling_dim,
            dropout=cfg.safety.dropout_p,
        )
        # Auxiliary decoder for L_recon (Sec. 3.10, Eq. 12)
        self.decoder = PhysiologyDecoder(
            latent_dim=cfg.perceiver.latent_dim,
            out_steps=cfg.inputs.phys_window,
            out_features=cfg.inputs.phys_dim,
            hidden=cfg.loss.recon_hidden,
            layers=cfg.loss.recon_layers,
        )
        self.graph_feature_dim = graph_feat

    # ------------------------------------------------------------------ #
    def encode_context(self, batch: MultimodalBatch, with_reconstruction: bool = False):
        """Phase I. Returns ``z_ctx`` and optionally ``Decoder(Z)`` for Eq. (12)."""
        if with_reconstruction:
            latents = self.context_encoder.encode_latents(batch)
            z = self.context_encoder.out_norm(latents.mean(dim=1))
            return z, self.decoder(latents)
        return self.context_encoder(batch), None

    def cognitive_state(self, z_ctx: torch.Tensor) -> CognitiveState:
        """Phase II over left-padded sliding windows of width W (Sec. 3.3)."""
        windows = sliding_windows(z_ctx, self.cfg.tft.window)
        return self.tft(windows)

    def build_query(self, z_ctx: torch.Tensor, state: CognitiveState) -> torch.Tensor:
        """``q_t = [z_ctx,t ; s_cog,t]`` (Sec. 3.4)."""
        return torch.cat([z_ctx, state.vector], dim=-1)

    def attend_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        query: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Phase III (Eqs. 6-7). Returns ``(h', node_scores)``."""
        if self.gat is None:
            raise RuntimeError("GAT is disabled in this configuration (ablation '\\ GAT')")
        return self.gat(node_features, edge_index, query, edge_weight)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        batch: MultimodalBatch,
        graph_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        with_reconstruction: bool = False,
    ) -> PipelineOutput:
        z_ctx, recon = self.encode_context(batch, with_reconstruction)
        state = self.cognitive_state(z_ctx)
        query = self.build_query(z_ctx, state)
        scores = features = None
        if self.gat is not None and graph_features is not None and edge_index is not None:
            features, scores = self.attend_graph(graph_features, edge_index, query, edge_weight)
        return PipelineOutput(
            z_ctx=z_ctx,
            state=state,
            query=query,
            node_scores=scores,
            node_features=features,
            reconstruction=recon,
        )

    # ------------------------------------------------------------------ #
    def scoring_context(
        self, query_row: torch.Tensor, subgraph_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Fuse ``[q_t ; pooled G_active,t]`` for the intent-scoring head (Sec. 3.6)."""
        if self.graph_feature_dim == 0:
            return query_row
        if subgraph_features is None:
            pooled = query_row.new_zeros(query_row.shape[0], self.graph_feature_dim)
        else:
            pooled = subgraph_features.mean(dim=0, keepdim=True).expand(query_row.shape[0], -1)
        return torch.cat([query_row, pooled], dim=-1)

    def uncertainty(
        self, context: torch.Tensor, token_embeddings: torch.Tensor, n_passes: Optional[int] = None
    ) -> UncertaintyEstimate:
        """Eq. (8): MC-Dropout predictive variance over the decoded tokens."""
        return self.intent_head.mc_dropout(
            context, token_embeddings, n_passes or self.cfg.safety.mc_passes
        )

    def select_subgraph(self, node_scores: torch.Tensor, turn_index: int) -> ActiveSubgraph:
        """Top-``K_top`` active intent sub-graph for one turn (Sec. 3.4)."""
        if self.gat is None:
            return ActiveSubgraph(node_ids=[], node_scores=node_scores[turn_index].detach())
        return self.gat.select_active_subgraph(node_scores, batch_index=turn_index)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

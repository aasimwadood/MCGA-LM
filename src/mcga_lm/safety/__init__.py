"""Phase V -- confidence-aware clinical filtering (paper Sec. 3.6)."""

from .gate import BayesianGate, ClarificationPrompt, GateDecision, clarification_bubbles

__all__ = ["BayesianGate", "GateDecision", "ClarificationPrompt", "clarification_bubbles"]

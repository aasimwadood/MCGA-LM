"""Hallucination taxonomy and detection (paper Sec. 4.2 taxonomy box, Sec. 5.2)."""

from __future__ import annotations

import inspect

import pytest

from mcga_lm.eval.hallucination import HallucinationDetector, HallucinationKind, summarise
from mcga_lm.memory.graph import IntentMemoryGraph


@pytest.fixture
def graph() -> IntentMemoryGraph:
    g = IntentMemoryGraph(node_dim=8)
    for name, node_type in (
        ("Nurse_Smith", "Person"), ("kettle", "Object"), ("kitchen", "Object"),
        ("pain", "AbstractState"), ("riluzole", "Object"),
    ):
        idx = g.add_node(name, node_type)
        g.add_edge(0, idx, "associated_with", weight=0.8)
    g.add_edge(g.node_id("kettle"), g.node_id("kitchen"), "located_in", weight=0.9)
    return g


def test_entity_in_graph_is_not_a_hallucination(graph) -> None:
    d = HallucinationDetector()
    assert d.classify(["kettle"], graph).kind is HallucinationKind.NONE


def test_wrong_caregiver_is_a_hard_hallucination(graph) -> None:
    """Sec. 4.2's canonical example: "naming the wrong caregiver"."""
    verdict = HallucinationDetector().classify(["nurse"], graph, known_partners=["Nurse_Smith"])
    assert verdict.is_hard and "person" in verdict.reason


def test_unknown_medication_is_a_hard_hallucination(graph) -> None:
    """Sec. 2: "misrepresenting a request for pain medication" can cause real harm."""
    assert HallucinationDetector().classify(["morphine"], graph).is_hard


def test_medication_present_in_the_graph_is_fine(graph) -> None:
    assert HallucinationDetector().classify(["riluzole"], graph).kind is HallucinationKind.NONE


def test_absent_but_uncontradicted_entity_is_soft(graph) -> None:
    """Sec. 4.2: "an entity absent from G and history but not contradicted"."""
    verdict = HallucinationDetector().classify(["television"], graph)
    assert verdict.is_soft


def test_entity_licensed_by_the_conversation_history_is_not_flagged(graph) -> None:
    history = [("partner", "shall I fetch the television remote")]
    assert HallucinationDetector().classify(["television"], graph, history=history).kind is HallucinationKind.NONE


def test_hard_outranks_soft_within_one_utterance(graph) -> None:
    verdict = HallucinationDetector().classify(["television", "morphine"], graph)
    assert verdict.is_hard


def test_relation_contradiction_detected(graph) -> None:
    d = HallucinationDetector()
    assert d.relation_contradiction("kettle", "located_in", "garden", graph).is_hard
    assert d.relation_contradiction("kettle", "located_in", "kitchen", graph) is None


def test_summary_reports_hard_and_soft_separately(graph) -> None:
    d = HallucinationDetector()
    verdicts = [
        d.classify(["kettle"], graph),
        d.classify(["television"], graph),
        d.classify(["morphine"], graph),
        d.classify(["kettle"], graph),
    ]
    s = summarise(verdicts)
    assert s["hard_rate"] == pytest.approx(0.25)
    assert s["soft_rate"] == pytest.approx(0.25)
    assert s["n"] == 4


# ---------------------------------------------------------------- D-11 ----- #
def test_detection_is_automatic_with_no_human_annotation_path() -> None:
    """DEVIATION D-11: the paper contradicts itself on human annotation.

    Sec. 3.8 mentions "two annotators ... blinded to system identity", but
    Sec. 4.2 ("No human adjudication of generated utterances was performed"),
    the hallucination taxonomy ("no human annotation was involved") and Sec. 6.3
    ("at any point in this study") all say the opposite. We implement the
    automatic-only path those three describe. This test records that choice, so
    that a human-adjudication path cannot appear without the discrepancy being
    revisited.
    """
    from mcga_lm.eval import hallucination as H

    source = inspect.getsource(H)
    assert "annotator" not in source.lower(), (
        "an annotator path would contradict Sec. 4.2/6.3; see D-11"
    )
    # Classification's only evidence is the graph and the dialogue history --
    # exactly the "rule-based entity linking against G and the dialogue history"
    # of Sec. 4.2, with no slot for a human label.
    signature = inspect.signature(H.HallucinationDetector.classify)
    assert {"entities", "graph", "history"} <= set(signature.parameters)
    assert not {"label", "annotation", "rater"} & set(signature.parameters)

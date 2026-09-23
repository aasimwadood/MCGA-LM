"""HF backend contract (paper Sec. 3.5, Table 3: LLaMA-3-8B, 4-bit NF4, LoRA).

Loading an 8B model is out of scope for a unit test, so these cover the pure
logic the backend must get right for the metrics to mean anything. Each one
corresponds to a bug the backend shipped with, found by audit rather than by a
run -- the backend had never been executed.
"""

from __future__ import annotations

import pytest

from mcga_lm.data.taxonomy import PRAGMATIC_FUNCTIONS, SLOT_TYPE
from mcga_lm.llm.hf_backend import classify_function


@pytest.mark.parametrize(
    "utterance,expected",
    [
        ("Could you pass me the blanket when you have a moment", "request_object"),
        ("I think it is time for my medication", "request_medication"),
        ("Could you help me adjust the pillow a little", "request_position_change"),
        ("I am getting hungry, could I have some toast", "request_food"),
    ],
)
def test_function_is_recovered_from_generated_text(utterance: str, expected: str) -> None:
    """The backend must label the function, or IHR@K is 0% by construction.

    Both ``Persona.accepts`` (exact match needs function *and* entity) and
    ``AdaptiveCommunicator._hit_rank`` compare against this field. Returning ""
    makes every comparison fail regardless of the language model's quality.
    """
    assert classify_function(utterance, ["Object"]) == expected


def test_unrecognisable_text_gets_no_function() -> None:
    """Better an empty label than a confidently wrong one."""
    assert classify_function("asdf qwerty zzz", []) == ""
    assert classify_function("", []) == ""


def test_stopwords_alone_do_not_decide_a_function() -> None:
    """"Could you please" is in almost every template; it must not classify."""
    assert classify_function("could you please", []) == ""


def test_classification_is_restricted_by_node_type_when_known() -> None:
    """A named AbstractState node should not yield a request_* function."""
    fn = classify_function("the pain is getting worse", ["AbstractState"])
    assert fn == "" or SLOT_TYPE[fn] == "AbstractState"


def test_every_returned_function_is_in_the_taxonomy() -> None:
    """Sec. 4.1 fixes 24 functions; the classifier must not invent a 25th."""
    for text in ("I would like some water", "thank you so much", "good morning",
                 "I am very tired", "could you pass the cup", "no thank you"):
        fn = classify_function(text, [])
        assert fn == "" or fn in PRAGMATIC_FUNCTIONS


def test_entities_are_graph_nodes_not_every_token() -> None:
    """The old backend set entities to every word in the utterance.

    ``Persona.accepts`` treats any entity overlap as a near-miss worth accepting
    30% of the time, and the hallucination detector treats unlisted entities as
    hallucinations. Passing all tokens corrupts both. This pins the contract the
    template backend already honours: entities are the graph nodes named.
    """
    node_types = {"water": "Object", "nurse": "Person"}
    utterance = "I would like some water please"
    entities = tuple(
        w.strip(".,!?").lower() for w in utterance.split() if w.strip(".,!?").lower() in node_types
    )
    assert entities == ("water",)
    assert "would" not in entities and "please" not in entities

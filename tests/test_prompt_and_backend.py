"""Structured prompt (Sec. 3.5) and the stand-in language backend."""

from __future__ import annotations

import numpy as np
import pytest

from mcga_lm.config import Config
from mcga_lm.llm.prompt import (
    INSTRUCTION,
    STATE_INSTRUCTIONS,
    DialogueTurn,
    UserProfile,
    build_prompt,
    candidates_for_fatigue,
    max_words_for_fatigue,
    render_history,
    temperature_for_fatigue,
)
from mcga_lm.llm.template_backend import TemplateLanguageBackend
from mcga_lm.states import fatigue_level


@pytest.fixture
def profile() -> UserProfile:
    return UserProfile(persona_id="P00", diagnosis="ALS (bulbar onset)", age=61,
                       idiolect=("no rush at all",))


def test_prompt_contains_all_four_components_and_the_instruction(profile) -> None:
    """Sec. 3.5: [INST] U S T_graph H <instruction> [/INST]."""
    cfg = Config()
    prompt = build_prompt(profile, "low", "User is connected to 'pain' [0.91]",
                          [DialogueTurn("Nurse", "how are you today")], cfg.llm)
    text = prompt.render()
    assert text.startswith("[INST]") and text.endswith("[/INST]")
    assert "ALS (bulbar onset)" in text                    # U
    assert STATE_INSTRUCTIONS["low"] in text               # S
    assert "User is connected to 'pain' [0.91]" in text    # T_graph
    assert "Nurse: how are you today" in text              # H
    assert INSTRUCTION in text


def test_graph_component_is_dropped_for_the_ungrounded_baselines(profile) -> None:
    cfg = Config()
    cfg.llm.use_graph_prompt = False
    prompt = build_prompt(profile, "low", "User is connected to 'pain' [0.91]", [], cfg.llm)
    assert "Personal memory" not in prompt.render()


def test_history_is_truncated_to_ten_turns() -> None:
    """Sec. 3.5 component 4: "the last 10 dialogue turns, with speaker labels"."""
    turns = [DialogueTurn("s", f"utterance {i}") for i in range(30)]
    rendered = render_history(turns, max_turns=10)
    assert "utterance 29" in rendered and "utterance 19" not in rendered
    assert rendered.count(";") == 9


def test_temperature_schedule_endpoints_match_section_3_5() -> None:
    """T = 1.2 at low fatigue, decreasing to T = 0.5 at high fatigue."""
    cfg = Config()
    assert temperature_for_fatigue(0.0, cfg.llm) == pytest.approx(1.2)
    assert temperature_for_fatigue(1.0, cfg.llm) == pytest.approx(0.5)
    assert temperature_for_fatigue(0.5, cfg.llm) == pytest.approx(0.85)
    xs = [temperature_for_fatigue(f, cfg.llm) for f in np.linspace(0, 1, 11)]
    assert all(b <= a for a, b in zip(xs, xs[1:]))  # monotonically decreasing


def test_option_count_and_verbosity_shrink_with_fatigue() -> None:
    cfg = Config()
    assert candidates_for_fatigue("low", cfg.llm) > candidates_for_fatigue("high", cfg.llm)
    assert max_words_for_fatigue("low", cfg.llm) > max_words_for_fatigue("high", cfg.llm)


def test_fatigue_discretisation_thirds() -> None:
    assert fatigue_level(0.1) == "low"
    assert fatigue_level(0.5) == "moderate"
    assert fatigue_level(0.9) == "high"


def test_backend_respects_the_word_budget(profile) -> None:
    cfg = Config()
    backend = TemplateLanguageBackend()
    rng = np.random.default_rng(0)
    for level in ("low", "moderate", "high"):
        prompt = build_prompt(profile, level, "", [], cfg.llm,
                              active_nodes=(("kettle", "Object", 0.9),))
        for cand in backend.generate(prompt, n=3, temperature=0.5, top_p=0.9, rng=rng):
            assert cand.n_words <= max_words_for_fatigue(level, cfg.llm) + 1  # +1 for punctuation


def test_grounded_generation_prefers_retrieved_entities(profile) -> None:
    cfg = Config()
    backend = TemplateLanguageBackend()
    prompt = build_prompt(profile, "high", "", [], cfg.llm,
                          active_nodes=(("kettle", "Object", 0.95),))
    rng = np.random.default_rng(0)
    cands = [backend.generate(prompt, n=1, temperature=0.5, top_p=0.9, rng=rng)[0] for _ in range(20)]
    assert all(c.grounded for c in cands)
    assert all("kettle" in c.text.lower() for c in cands)


def test_hot_decoding_can_leave_the_retrieved_subgraph(profile) -> None:
    """Retrieval biases generation without constraining it (A-30)."""
    cfg = Config()
    backend = TemplateLanguageBackend()
    prompt = build_prompt(profile, "low", "", [], cfg.llm,
                          active_nodes=(("kettle", "Object", 0.95),))
    rng = np.random.default_rng(0)
    grounded = [backend.generate(prompt, n=1, temperature=1.2, top_p=0.9, rng=rng)[0].grounded
                for _ in range(200)]
    assert 0.0 < 1.0 - float(np.mean(grounded)) < 1.0


def test_ungrounded_prompt_uses_only_global_statistics(profile) -> None:
    cfg = Config()
    backend = TemplateLanguageBackend()
    prompt = build_prompt(profile, "low", "", [], cfg.llm, active_nodes=())
    rng = np.random.default_rng(0)
    cands = backend.generate(prompt, n=5, temperature=1.0, top_p=0.9, rng=rng)
    assert cands and all(not c.grounded for c in cands)


def test_token_embeddings_are_deterministic() -> None:
    backend = TemplateLanguageBackend(embedding_dim=32)
    a = backend.embed_tokens("I would like some tea")
    b = backend.embed_tokens("I would like some tea")
    assert a.shape == (5, 32)
    np.testing.assert_allclose(a, b)

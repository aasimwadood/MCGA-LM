"""Keystroke-level baselines (paper Sec. 4.3 items 1, 5, 6)."""

from __future__ import annotations

import numpy as np
import pytest

from mcga_lm.baselines.keystroke import (
    AdaptiveFrequencyRanker,
    BigramRanker,
    CompletionScanningSimulator,
    GridScanningSimulator,
    KLMParams,
)

SENTENCES = [
    "I would like some tea please",
    "the kettle is in the kitchen",
    "I would like some water",
    "my pain is worse today",
    "thank you nurse smith",
] * 40


def test_grid_costs_two_activations_per_selection() -> None:
    """Row-column scanning on a 16-cell grid: one stop on the row, one on the cell."""
    params = KLMParams(error_rate=0.0, page_change_presses=0)
    sim = GridScanningSimulator(params, on_page_rate=1.0)
    out = sim.simulate_utterance("one two three four five", np.random.default_rng(0))
    assert out.words == 5
    assert out.presses == 10


def test_page_changes_add_cost() -> None:
    rng = np.random.default_rng(0)
    cheap = GridScanningSimulator(KLMParams(error_rate=0.0), on_page_rate=1.0)
    dear = GridScanningSimulator(KLMParams(error_rate=0.0), on_page_rate=0.0)
    text = "one two three four five six"
    assert dear.simulate_utterance(text, rng).presses > cheap.simulate_utterance(text, rng).presses


def test_grid_wpm_is_in_the_published_range() -> None:
    """Sec. 4.3 validates the grid simulation against 10.5 +/- 2.1 WPM."""
    sim = GridScanningSimulator(KLMParams())
    rng = np.random.default_rng(0)
    rates = [sim.simulate_utterance(s, rng).wpm for s in SENTENCES[:60]]
    assert 6.0 <= float(np.mean(rates)) <= 16.0


def test_completion_ranker_returns_prefix_matches() -> None:
    ranker = BigramRanker(SENTENCES)
    ranked = ranker.rank("some", "te", k=5)
    assert all(w.startswith("te") for w in ranked)
    assert "tea" in ranked


def test_word_prediction_gives_kspc_below_two_presses_per_character() -> None:
    sim = CompletionScanningSimulator(BigramRanker(SENTENCES), KLMParams())
    rng = np.random.default_rng(0)
    out = sim.simulate_utterance("I would like some tea please", rng)
    assert 0.0 < out.kspc < 2.0


def test_personalised_ranking_beats_a_global_one_on_that_persons_own_text() -> None:
    """Sec. 4.3 item 6: the adaptive grid is initialised from *that persona's*
    training-split utterances, which is where its advantage comes from."""
    generic = ["the weather is fine today", "please close the window", "what time is it"] * 40
    personal = ["my baclofen is due at four", "baclofen makes the cramp easier"] * 40
    target = "my baclofen is due at four"
    global_sim = CompletionScanningSimulator(AdaptiveFrequencyRanker(generic), KLMParams())
    personal_sim = CompletionScanningSimulator(AdaptiveFrequencyRanker(personal), KLMParams())
    a = personal_sim.simulate_utterance(target, np.random.default_rng(0)).presses
    b = global_sim.simulate_utterance(target, np.random.default_rng(0)).presses
    assert a < b


def test_a_word_absent_from_the_inventory_is_typed_out_in_full() -> None:
    """The completion list cannot help with vocabulary it has never seen."""
    sim = CompletionScanningSimulator(AdaptiveFrequencyRanker(["alpha beta gamma"]), KLMParams())
    out = sim.simulate_utterance("zzzz", np.random.default_rng(0))
    assert out.presses >= 2 * len("zzzz")


def test_adaptive_ranker_tracks_recency() -> None:
    ranker = AdaptiveFrequencyRanker(["alpha beta", "alpha beta"])
    for _ in range(20):
        ranker.observe("gamma")
    assert "gamma" in ranker.rank("<s>", "", k=3)


def test_kspc_counts_keystrokes_not_switch_activations() -> None:
    """KSPC is a text-entry measure: one keystroke per chosen cell. Row-column
    scanning spends two switch activations per keystroke, and conflating the two
    would put every system above 1.0 and make Table 7's 0.89 / 0.78 unreachable."""
    sim = CompletionScanningSimulator(BigramRanker(SENTENCES), KLMParams())
    out = sim.simulate_utterance("I would like some tea please", np.random.default_rng(0))
    assert out.presses == 2 * out.selections
    assert out.kspc == pytest.approx(out.selections / out.characters)
    assert out.kspc < 1.0  # word prediction saves keystrokes


def test_grid_kspc_is_well_below_one_for_whole_word_cells() -> None:
    sim = GridScanningSimulator(KLMParams(error_rate=0.0), on_page_rate=1.0)
    out = sim.simulate_utterance("I would like some tea please", np.random.default_rng(0))
    assert out.selections == out.words
    assert out.kspc < 0.5

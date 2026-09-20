"""Evaluation metrics (paper Sec. 4.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pytest

from mcga_lm.eval import metrics as M


@dataclass
class FakeTurn:
    sact: int = 1
    accepted: bool = True
    utterance: Optional[str] = "I would like some tea"
    abstained: bool = False
    gate_accepted: bool = True
    n_offered: int = 1
    hit_rank: Optional[int] = 1
    confidence: float = 0.9


def test_sact_is_a_plain_mean() -> None:
    assert M.sact([FakeTurn(sact=1), FakeTurn(sact=3), FakeTurn(sact=5)]) == pytest.approx(3.0)


def test_wolpaw_itr_matches_the_printed_formula() -> None:
    """Sec. 4.2 / Sec. 5.2: ITR = (1/T)[log2 N + P log2 P + (1-P) log2((1-P)/(N-1))]."""
    n, p, t = 3, 0.89, 4.2
    expected = (math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))) / (t / 60.0)
    assert M.information_transfer_rate(n, p, t) == pytest.approx(expected)


def test_itr_is_zero_at_chance_and_maximal_at_perfect_accuracy() -> None:
    assert M.information_transfer_rate(4, 0.25, 4.0) == pytest.approx(0.0, abs=1e-9)
    assert M.information_transfer_rate(4, 1.0, 4.0) == pytest.approx(2.0 / (4.0 / 60.0))


def test_intent_hit_rate_is_monotone_in_k() -> None:
    turns = [FakeTurn(hit_rank=r) for r in (1, 2, 3, 5, None)]
    assert M.intent_hit_rate(turns, 1) == pytest.approx(0.2)
    assert M.intent_hit_rate(turns, 3) == pytest.approx(0.6)
    assert M.intent_hit_rate(turns, 5) == pytest.approx(0.8)


def test_false_acceptance_rate_counts_only_gated_presentations() -> None:
    turns = [
        FakeTurn(gate_accepted=True, accepted=True),
        FakeTurn(gate_accepted=True, accepted=False),
        FakeTurn(gate_accepted=False, accepted=False),  # suppressed: not in the denominator
    ]
    assert M.false_acceptance_rate(turns) == pytest.approx(0.5)


def test_words_per_minute_uses_selection_and_overhead_time() -> None:
    turns = [FakeTurn(sact=2, utterance="one two three four five")]  # 5 words
    wpm = M.words_per_minute(turns, seconds_per_activation=4.0, overhead_s=1.0)
    assert wpm == pytest.approx(5 / ((2 * 4.0 + 1.0) / 60.0))


def test_bleu_is_one_for_an_exact_match_and_falls_for_a_mismatch() -> None:
    ref = "i would like some tea please now"
    assert M.sentence_bleu(ref, ref) == pytest.approx(1.0)
    mismatch = M.sentence_bleu(ref, "the cat sat on the mat today")
    assert mismatch < 0.05


def test_bleu_brevity_penalty_punishes_short_hypotheses() -> None:
    ref = "i would like some tea please now"
    assert M.sentence_bleu(ref, "i would") < M.sentence_bleu(ref, "i would like some tea")


def test_rouge_l_uses_the_longest_common_subsequence() -> None:
    assert M.lcs_length("a b c d".split(), "a x c d".split()) == 3
    assert M.rouge_l("a b c", "a b c") == pytest.approx(1.0)
    assert M.rouge_l("a b c", "x y z") == pytest.approx(0.0)
    assert 0 < M.rouge_l("a b c d", "a c d") < 1


def test_ece_is_zero_when_perfectly_calibrated() -> None:
    conf = [0.95] * 100
    correct = [True] * 95 + [False] * 5
    assert M.expected_calibration_error(conf, correct, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_overconfidence() -> None:
    conf = [0.99] * 100
    correct = [True] * 50 + [False] * 50
    assert M.expected_calibration_error(conf, correct) == pytest.approx(0.49, abs=1e-9)


def test_calibration_curve_bins_sum_to_the_sample() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 500).tolist()
    correct = [bool(rng.random() < c) for c in conf]
    curve = M.calibration_curve(conf, correct, n_bins=10)
    assert sum(curve["count"]) == 500
    assert len(curve["confidence"]) == len(curve["accuracy"]) == len(curve["count"])


def test_kspc_definition() -> None:
    assert M.keystrokes_per_character(89, 100) == pytest.approx(0.89)
    turns = [FakeTurn(sact=3, accepted=True, utterance="abcdefghij")]  # 10 chars
    assert M.kspc_from_results(turns) == pytest.approx(0.3)


def test_summary_helpers_tolerate_empty_samples() -> None:
    """Table 7 marks SACT "n/r" for the character-level baselines, so the
    aggregation helpers must survive an all-missing column."""
    from mcga_lm.utils import mean_sd, median_iqr

    for out in (mean_sd([]), median_iqr([])):
        assert all(np.isnan(v) for k, v in out.items() if k != "n")
    single = mean_sd([3.0])
    assert single["mean"] == pytest.approx(3.0) and single["sd"] == 0.0

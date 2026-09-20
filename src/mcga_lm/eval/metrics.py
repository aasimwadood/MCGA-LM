

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np


# ------------------------------------------------------- efficiency ------- #
def sact(results: Sequence) -> float:
    """Mean switch activations per communicative turn (Sec. 4.2)."""
    values = [r.sact for r in results]
    return float(np.mean(values)) if values else float("nan")


def words_per_minute(
    results: Sequence, seconds_per_activation: float, overhead_s: float = 1.0
) -> float:
    """WPM: total words produced divided by session time (Sec. 4.2).

    Session time is modelled as ``SACT * T_select + overhead`` per turn, which
    includes selection, reading and error-correction time as the paper requires.
    Turns that abstain contribute their cost but no words.
    """
    total_words = 0
    total_seconds = 0.0
    for r in results:
        total_seconds += r.sact * seconds_per_activation + overhead_s
        if r.accepted and r.utterance:
            total_words += len(r.utterance.split())
    if total_seconds <= 0:
        return float("nan")
    return float(total_words / (total_seconds / 60.0))


def information_transfer_rate(n_choices: int, accuracy: float, select_seconds: float) -> float:
    """Wolpaw et al. (2002) ITR in bits/min, as written in Sec. 4.2.

        ITR = (1/T) [ log2 N + P log2 P + (1-P) log2((1-P)/(N-1)) ]

    Sec. 5.2 reports this only for the intent-selection stage (N = K
    candidates), never for the open-ended generation pipeline.

    DEVIATION D-12: the paper's own worked example does not check out. Sec. 5.2
    gives K = 3, P = 0.89, T_select = 4.2 s and reports 18.3 bits/min; this
    formula returns 13.93 for those inputs. T_select ~ 3.2 s would give 18.3.
    We keep the stated 4.2 s, so this function cannot reproduce the printed
    figure -- see docs/ASSUMPTIONS.md.
    """
    if n_choices < 2 or select_seconds <= 0:
        return float("nan")
    p = min(max(float(accuracy), 0.0), 1.0)
    bits = math.log2(n_choices)
    if p > 0:
        bits += p * math.log2(p)
    if p < 1:
        bits += (1 - p) * math.log2((1 - p) / (n_choices - 1))
    return float(bits / (select_seconds / 60.0))


def intent_hit_rate(results: Sequence, k: int) -> float:
    """IHR@K: share of turns whose true intent is in the top-K candidates."""
    if not results:
        return float("nan")
    hits = sum(1 for r in results if r.hit_rank is not None and r.hit_rank <= k)
    return float(hits / len(results))


def keystrokes_per_character(n_keystrokes: float, n_characters: float) -> float:
    """KSPC -- used for the character-level baselines (Sec. 4.3, Table 7 note)."""
    if n_characters <= 0:
        return float("nan")
    return float(n_keystrokes / n_characters)


def kspc_from_results(results: Sequence, seconds_per_activation: float = 1.0) -> float:
    """KSPC for a generative system: activations divided by characters emitted.

    ASSUMPTION A-23: Sec. 5.1 quotes 0.31 KSPC for MCGA-LM without defining it
    for a system that never types characters. We define it as switch
    activations divided by the characters of the accepted utterance, which is
    the natural generalisation and is what makes the grid comparison meaningful.
    """
    keys = sum(r.sact for r in results)
    chars = sum(len(r.utterance) for r in results if r.accepted and r.utterance)
    return keystrokes_per_character(keys, chars)


# ------------------------------------------------------------ safety ------ #
def false_acceptance_rate(results: Sequence) -> float:
    """FAR: share of gate-passing utterances the user then rejected (Sec. 4.2).

    Denominator is the set of candidates that passed the Bayesian gate and were
    presented; numerator is those the simulated user rejected.
    """
    passed = [r for r in results if r.gate_accepted and r.n_offered > 0]
    if not passed:
        return float("nan")
    rejected = sum(1 for r in passed if not r.accepted)
    return float(rejected / len(passed))


def abstention_rate(results: Sequence) -> float:
    """Share of turns that ended in Algorithm 1's line-23 abstention."""
    if not results:
        return float("nan")
    return float(sum(1 for r in results if r.abstained) / len(results))


# ------------------------------------------------------- calibration ------ #
def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """ECE over ``n_bins`` equal-width bins (Sec. 4.2; Guo et al., 2017)."""
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def calibration_curve(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> Dict[str, List[float]]:
    """Reliability-diagram points for Fig. 2."""
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs: List[float] = []
    ys: List[float] = []
    ns: List[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        xs.append(float(conf[mask].mean()))
        ys.append(float(acc[mask].mean()))
        ns.append(float(mask.sum()))
    return {"confidence": xs, "accuracy": ys, "count": ns}


# --------------------------------------------------------- text quality --- #
def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference: str, hypothesis: str, max_n: int = 4, smooth: float = 0.1) -> float:
    """BLEU-4 with additive smoothing and the standard brevity penalty.

    Additive (Chen & Cherry style) smoothing keeps short AAC utterances from
    collapsing to zero when a higher-order n-gram happens to miss, while leaving
    an exact match at 1.0. ASSUMPTION A-31: the paper does not state its BLEU
    variant, so absolute BLEU values are not comparable across implementations."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not hyp or not ref:
        return 0.0
    log_precisions = []
    for n in range(1, max_n + 1):
        ref_counts = _ngrams(ref, n)
        hyp_counts = _ngrams(hyp, n)
        total = max(sum(hyp_counts.values()), 0)
        if total == 0:
            log_precisions.append(math.log(smooth / (smooth + 1.0)))
            continue
        overlap = sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())
        log_precisions.append(math.log((overlap + smooth) / (total + smooth)))
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
    return float(bp * math.exp(sum(log_precisions) / max_n))


def corpus_bleu(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    pairs = [(r, h) for r, h in zip(references, hypotheses) if h]
    if not pairs:
        return float("nan")
    return float(np.mean([sentence_bleu(r, h) for r, h in pairs]))


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length (for ROUGE-L)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(reference: str, hypothesis: str, beta: float = 1.2) -> float:
    """ROUGE-L F-measure (Sec. 4.2)."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref or not hyp:
        return 0.0
    lcs = lcs_length(ref, hyp)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    b2 = beta**2
    return float((1 + b2) * precision * recall / (recall + b2 * precision))


def corpus_rouge_l(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    pairs = [(r, h) for r, h in zip(references, hypotheses) if h]
    if not pairs:
        return float("nan")
    return float(np.mean([rouge_l(r, h) for r, h in pairs]))

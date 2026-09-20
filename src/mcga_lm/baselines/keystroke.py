

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class KLMParams:
    """Keystroke-level model constants for single-switch row-column scanning."""

    scan_dwell_s: float = 0.85  # time the highlight rests on each row/column
    activation_s: float = 0.30  # time to make one switch activation
    grid_rows: int = 4  # 16-cell grid (Sec. 4.3)
    grid_cols: int = 4
    page_change_presses: int = 2  # cost of navigating to another vocabulary page
    prediction_cells: int = 5  # completion candidates offered
    error_rate: float = 0.04  # ASSUMPTION: mis-selections needing a correction


@dataclass
class KeystrokeResult:
    """One simulated utterance.

    ``selections`` counts *keystrokes* in the text-entry sense -- one chosen
    cell, whether that cell is a character or a whole predicted word -- while
    ``presses`` counts *switch activations*, which under row-column scanning is
    two per selection. KSPC is defined on selections, as in the text-entry
    literature (KSPC = 1 means one keystroke per character), so a system with
    working word prediction lands below 1. SACT is defined on presses.
    """

    presses: int
    selections: int
    seconds: float
    characters: int
    words: int

    @property
    def kspc(self) -> float:
        """Keystrokes per character (Sec. 4.2, Table 7 footnote)."""
        return self.selections / self.characters if self.characters else float("nan")

    @property
    def wpm(self) -> float:
        return (self.words / (self.seconds / 60.0)) if self.seconds > 0 else float("nan")


class GridScanningSimulator:
    """TouchChat-with-WordPower stand-in: symbol/word selection on a 16-cell grid.

    Row-column scanning costs two activations per selection (one to stop on the
    row, one to stop on the cell). Vocabulary that is not on the current page
    costs an extra page change. This is the only baseline reported on SACT.
    """

    def __init__(self, params: Optional[KLMParams] = None, on_page_rate: float = 0.72) -> None:
        self.p = params or KLMParams()
        # ASSUMPTION A-26: share of target words reachable without a page change.
        self.on_page_rate = on_page_rate

    def simulate_utterance(self, text: str, rng: np.random.Generator) -> KeystrokeResult:
        words = [w for w in text.split() if w]
        presses = 0
        selections_total = 0
        seconds = 0.0
        for _ in words:
            selections = 1  # one symbol/word cell per word (WordPower-style)
            if rng.random() > self.on_page_rate:
                presses += self.p.page_change_presses
                selections_total += 1
                seconds += self.p.page_change_presses * (self.p.scan_dwell_s + self.p.activation_s)
            if rng.random() < self.p.error_rate:
                selections += 1  # mis-selection plus its correction
            presses += 2 * selections
            selections_total += selections
            # Expected scan time: half a row sweep plus half a column sweep.
            seconds += selections * (
                (self.p.grid_rows / 2 + self.p.grid_cols / 2) * self.p.scan_dwell_s
                + 2 * self.p.activation_s
            )
        return KeystrokeResult(
            presses=presses,
            selections=selections_total,
            seconds=seconds,
            characters=len(text.replace(" ", "")),
            words=len(words),
        )


class CompletionScanningSimulator:
    """Character-level entry with a word-completion list (baselines 5 and 6).

    The user types characters until the target word appears among the top
    ``prediction_cells`` completions, then spends one selection on it. The
    ranking function is what distinguishes the two baselines:

      * static bigram      -- global corpus statistics, no personalisation
      * adaptive grid      -- the persona's own frequency/recency statistics
    """

    def __init__(self, ranker: "Ranker", params: Optional[KLMParams] = None) -> None:
        self.ranker = ranker
        self.p = params or KLMParams()

    def simulate_utterance(self, text: str, rng: np.random.Generator) -> KeystrokeResult:
        words = [w.lower() for w in text.split() if w]
        selections = 0
        seconds = 0.0
        previous = "<s>"
        for word in words:
            typed = 0
            selected = False
            for prefix_len in range(0, len(word)):
                ranked = self.ranker.rank(previous, word[:prefix_len], self.p.prediction_cells)
                if word in ranked:
                    selections += 1  # one pick from the completion list
                    seconds += self._selection_time()
                    selected = True
                    break
                selections += 1  # one character
                seconds += self._selection_time()
                typed += 1
            if not selected:
                selections += (len(word) - typed) + 1  # finish the word, then space
                seconds += (len(word) - typed + 1) * self._selection_time()
            previous = word
        return KeystrokeResult(
            presses=2 * selections,  # row-column scanning: two activations each
            selections=selections,
            seconds=seconds,
            characters=len(text.replace(" ", "")),
            words=len(words),
        )

    def _selection_time(self) -> float:
        return (
            (self.p.grid_rows / 2 + self.p.grid_cols / 2) * self.p.scan_dwell_s
            + 2 * self.p.activation_s
        )


class Ranker:
    """Interface for the two completion rankers."""

    def rank(self, previous_word: str, prefix: str, k: int) -> List[str]:  # pragma: no cover
        raise NotImplementedError


class BigramRanker(Ranker):
    """Baseline 5: smoothed word bigram over the AAC-Intent-Corpus training split.

    "It uses no dialogue context, no personalisation, no physiological input, and
    performs no generation" (Sec. 4.3).
    """

    def __init__(self, sentences: Iterable[str], add_k: float = 0.1) -> None:
        self.add_k = add_k
        self.unigrams: Counter = Counter()
        self.bigrams: Dict[str, Counter] = defaultdict(Counter)
        for sentence in sentences:
            tokens = ["<s>"] + [t.lower() for t in sentence.replace(".", "").split()]
            for a, b in zip(tokens[:-1], tokens[1:]):
                self.bigrams[a][b] += 1
                self.unigrams[b] += 1

    def rank(self, previous_word: str, prefix: str, k: int) -> List[str]:
        following = self.bigrams.get(previous_word, Counter())
        scores: Dict[str, float] = {}
        for word, count in self.unigrams.items():
            if prefix and not word.startswith(prefix):
                continue
            scores[word] = (following[word] + self.add_k) * (count + self.add_k)
        return [w for w, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]


class AdaptiveFrequencyRanker(Ranker):
    """Baseline 6: frequency- and recency-adaptive cell ranking (Sec. 4.3).

    "cell positions are re-ordered as the persona's usage statistics accumulate,
    initialised per persona from that persona's training-split utterances."
    No language model, no intent inference, no generation.
    """

    def __init__(self, sentences: Iterable[str], recency_halflife: float = 50.0) -> None:
        self.counts: Counter = Counter()
        self.last_seen: Dict[str, int] = {}
        self.t = 0
        for sentence in sentences:
            for token in sentence.replace(".", "").split():
                self.observe(token.lower())

    def observe(self, word: str) -> None:
        self.t += 1
        self.counts[word] += 1
        self.last_seen[word] = self.t

    def rank(self, previous_word: str, prefix: str, k: int) -> List[str]:
        scores: Dict[str, float] = {}
        for word, count in self.counts.items():
            if prefix and not word.startswith(prefix):
                continue
            recency = math.exp(-(self.t - self.last_seen.get(word, 0)) / 200.0)
            scores[word] = count * (1.0 + recency)
        return [w for w, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]

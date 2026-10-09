"""Loaders for the three public pre-training corpora (paper Sec. 4.1, Table 3).

    MAMEM   36 participants; EEG, GSR, heart rate, eye gaze   10.5281/zenodo.834154
    CLAS    62 participants; ECG, PPG, EDA, accelerometry     10.1109/BIA48344.2019.8967457
    WESAD   15 subjects; ECG, EDA, EMG, respiration, accel.   10.1145/3267305.3267350

NOT BUNDLED. None of these are redistributed here; download them from the DOIs
above into ``data/raw/<name>/``. Each loader raises a clear error if the
directory is missing, and ``synthetic_pretraining_corpus`` provides a
clearly-labelled substitute so pre-training runs offline.

STATUS: the MAMEM loader follows the stream structure of the actual Phase I
archive (inspected file by file); the CLAS loader follows the folder and
Block_Details layout of the released archive (checked against it with
``clas_inventory``), with its signal columns detected from the file headers; the
WESAD loader follows the published pickle layout. None has yet been run over a
complete archive here, so each logs what it loaded and skipped, and pre-training
reports the window count per corpus.

Every recording is resampled to 64 Hz and cut into 2-s windows with 50% overlap
(Sec. 4.1), each labelled with its condition's proxy target (Table 4).

The participant counts and channels above differ from Sec. 4.1's description of
the same datasets;
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import InputDims
from ..utils import get_logger

logger = get_logger(__name__)

DATASET_DOIS = {
    "mamem": "10.5281/zenodo.834154",
    "clas": "10.1109/BIA48344.2019.8967457",
    "wesad": "10.1145/3267305.3267350",
}

# Sec. 4.1: labels are harmonised to low/medium/high load with continuous
# self-report where available.
LOAD_LABELS = ("low", "medium", "high")


@dataclass
class PhysiologyRecord:
    """One windowed physiological example with a fatigue/load label."""

    signals: np.ndarray  # (T, C)
    fatigue: float  # Borg CR-10 scaled to [0, 1] (Eq. 10 target)
    load_label: str
    subject: str
    source: str
    # The subsequent utterance, for the stage-1 contrastive term (Sec. 3.7).
    # Empty for MAMEM/CLAS/WESAD, which carry no language.
    transcript: str = ""


def _require(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found at {path}. It is not redistributed with this repository; "
            f"download it from DOI {DATASET_DOIS[name]} into {path}. See data/README.md."
        )
    return path


# --------------------------------------------------------------- windowing -- #
@dataclass
class Windowing:
    """How recordings become pre-training examples (Sec. 3.2, Sec. 4.1).

    Every recording is resampled to the encoder's ``x_phys`` rate and cut into
    windows of ``window`` samples -- 2 s at 64 Hz -- with 50% overlap, and each
    window inherits the load label of the condition it was recorded in (Table 4).
    ``max_windows`` keeps that many evenly spaced windows per recording; it is a
    compute budget, not part of the paper, and ``None`` keeps them all.
    """

    rate_hz: int = 64
    window: int = 128
    stride: int = 64
    max_windows: Optional[int] = None


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate non-finite samples per channel; an all-NaN channel becomes 0."""
    x = np.array(x, dtype=np.float64)
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    idx = np.arange(len(x))
    for c in range(x.shape[1]):
        m = bad[:, c]
        x[m, c] = np.interp(idx[m], idx[~m], x[~m, c]) if (~m).any() else 0.0
    return x


def _resample(x: np.ndarray, rate_hz: float, target_hz: int) -> np.ndarray:
    """``(T, C)`` at ``rate_hz`` to ``target_hz``, with polyphase anti-aliasing."""
    from fractions import Fraction

    from scipy.signal import resample_poly

    x = _fill_nan(x.reshape(len(x), -1))
    if abs(rate_hz - target_hz) < 1e-9:
        return x
    ratio = Fraction(target_hz / rate_hz).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=0)


def _windows(x: np.ndarray, w: Windowing) -> List[np.ndarray]:
    n = (len(x) - w.window) // w.stride + 1 if len(x) >= w.window else 0
    starts = np.arange(n) * w.stride
    if w.max_windows is not None and n > w.max_windows:
        starts = starts[np.linspace(0, n - 1, w.max_windows).round().astype(int)]
    return [x[s : s + w.window].astype(np.float32) for s in starts]


def _records(
    x: np.ndarray, rate_hz: float, w: Windowing, fatigue: float, label: str, subject: str, source: str
) -> List[PhysiologyRecord]:
    return [
        PhysiologyRecord(seg, fatigue, label, subject, source)
        for seg in _windows(_resample(x, rate_hz, w.rate_hz), w)
    ]


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """``[start, end)`` of each contiguous run of ``True``."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def _summary(name: str, records: Sequence[PhysiologyRecord], skipped: Sequence[str]) -> None:
    labels = {lab: sum(r.load_label == lab for r in records) for lab in LOAD_LABELS}
    logger.info("%s: %d windows %s from %d recordings; %d files skipped",
                name, len(records), labels, len({r.subject for r in records}), len(skipped))
    for s in skipped[:10]:
        logger.warning("%s: skipped %s", name, s)


# ------------------------------------------------------------------- WESAD -- #
WESAD_RATE_HZ = 700  # RespiBAN chest unit
# Table 4. WESAD protocol codes: 1 baseline, 2 stress (TSST), 3 amusement, 4 meditation.
WESAD_CONDITIONS = {1: ("low", 0.15), 2: ("high", 0.80), 3: ("medium", 0.40), 4: ("low", 0.20)}


def load_wesad(root: str | Path, windowing: Optional[Windowing] = None) -> List[PhysiologyRecord]:
    """WESAD (Schmidt et al., 2018): per-subject ``S*/S*.pkl`` chest recordings.

    ECG, EDA, EMG and respiration from the chest unit. Each contiguous run of a
    protocol condition is windowed on its own, so no window spans two conditions.
    """
    import pickle

    w = windowing or Windowing()
    root = _require(Path(root), "wesad")
    records: List[PhysiologyRecord] = []
    skipped: List[str] = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pkl = subject_dir / f"{subject_dir.name}.pkl"
        if not pkl.exists():
            continue
        try:
            with open(pkl, "rb") as fh:
                data = pickle.load(fh, encoding="latin1")
            chest = data["signal"]["chest"]
            labels = np.asarray(data["label"]).ravel()
            stacked = np.concatenate(
                [np.asarray(chest[k]).reshape(len(labels), -1) for k in ("ECG", "EDA", "EMG", "Resp")], axis=1
            )
        except Exception as exc:  # one unreadable subject must not end a long run
            skipped.append(f"{pkl} ({type(exc).__name__}: {exc})")
            continue
        for code, (label, fatigue) in WESAD_CONDITIONS.items():
            for a, b in _runs(labels == code):
                records.extend(_records(stacked[a:b], WESAD_RATE_HZ, w, fatigue, label, subject_dir.name, "wesad"))
    _summary("wesad", records, skipped)
    return records


# ------------------------------------------------------------------- MAMEM -- #
# Each MAMEM .mat holds the streams of one recorded session, as exported from
# Lab Streaming Layer: a cell array of structs with ``info`` (type, nominal
# rate), ``time_series`` (channels x samples) and ``time_stamps``. Stream types
# seen in the archive: EEG (Emotiv, 14 ch at 128 Hz), BIO (Shimmer GSR/heart
# rate, 2 ch at 256 Hz), Gaze, Markers and irregular VALUE streams. x_phys
# carries EEG, HRV and EDA (Sec. 3.2), so EEG and BIO are kept, in that order;
# gaze belongs to x_beh, which pre-training synthesises.
MAMEM_PHYS_TYPES = ("EEG", "BIO")


def _mamem_streams(mat: dict):
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        for s in np.atleast_1d(value).ravel():
            if hasattr(s, "info") and hasattr(s, "time_series"):
                yield s


def load_mamem(root: str | Path, windowing: Optional[Windowing] = None) -> List[PhysiologyRecord]:
    """MAMEM Phase I (Nikolopoulos et al., 2017): one LSL session per ``.mat`` file.

    The kept streams are resampled to a common rate and aligned on their shared
    time span. Every MAMEM session carries the same proxy target (Table 4).
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("loading MAMEM needs scipy") from exc

    w = windowing or Windowing()
    root = _require(Path(root), "mamem")
    records: List[PhysiologyRecord] = []
    skipped: List[str] = []
    for mat_path in sorted(root.rglob("*.mat")):
        if "__MACOSX" in mat_path.parts or mat_path.name.startswith("._"):
            continue
        try:
            mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            streams = []
            for s in _mamem_streams(mat):
                kind = str(getattr(s.info, "type", ""))
                rate = float(getattr(s.info, "nominal_srate", 0) or 0)
                series = np.asarray(s.time_series)
                if kind not in MAMEM_PHYS_TYPES or rate <= 0 or series.dtype == object or series.ndim != 2:
                    continue
                t0 = float(np.asarray(s.time_stamps, dtype=float).ravel()[0])
                streams.append((MAMEM_PHYS_TYPES.index(kind), t0, _resample(series.T, rate, w.rate_hz)))
        except Exception as exc:
            skipped.append(f"{mat_path} ({type(exc).__name__}: {exc})")
            continue
        if not streams:
            skipped.append(f"{mat_path} (no EEG or BIO stream)")
            continue
        streams.sort(key=lambda s: (s[0], s[1]))
        start = max(t0 for _, t0, _ in streams)
        offsets = [int(round((start - t0) * w.rate_hz)) for _, t0, _ in streams]
        length = min(len(x) - o for (_, _, x), o in zip(streams, offsets))
        if length < w.window:
            skipped.append(f"{mat_path} (streams overlap for under one window)")
            continue
        aligned = np.concatenate([x[o : o + length] for (_, _, x), o in zip(streams, offsets)], axis=1)
        subject = "/".join(mat_path.parts[-3:-1])  # site/participant, e.g. SHEBA/SH5
        records.extend(
            PhysiologyRecord(seg, 0.50, "medium", subject, "mamem") for seg in _windows(aligned, w)
        )
    _summary("mamem", records, skipped)
    return records


# -------------------------------------------------------------------- CLAS -- #
# Layout of the released archive, as found in it:
#   Participants/Part<N>/.../by_block/<block>_ecg_.csv       ECG
#   Participants/Part<N>/.../by_block/<block>_gsr_ppg_.csv   GSR, PPG, accelerometer
#   Block_details/Part<N>_Block_Details.csv
#       Block, Block Type, ECG File, EDA&PPG File, Length(s), EDA/ECG/PPG Quality
# Files are matched to blocks by block number: Block_Details names the ECG file
# "<block>_ecg.csv", while on disk it is "<block>_ecg_.csv". The per-stimulus
# copies of the same signals carry an index after the final underscore and are
# not read, so no sample is counted twice. A folder with no participant number
# (the archive's "Sample") is skipped, and so is a participant with no
# Block_Details, rather than guessing its block types.
CLAS_RATE_HZ = 256  # Shimmer3 ECG and GSR+ units
# Table 4: arithmetic and Stroop blocks are high load, every other block low --
# including the Logic task, which Block_Details calls "IQ Test".
CLAS_HIGH_LOAD = ("math", "arithmetic", "stroop")
_CLAS_BLOCK_FILE = re.compile(r"^(?P<block>.+?)_(?P<kind>ecg|gsr_ppg)_\.csv$", re.IGNORECASE)
_PARTICIPANT = re.compile(r"part(?:icipant)?[\s_-]*(\d+)", re.IGNORECASE)


def _participant(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        m = _PARTICIPANT.search(part)
        if m:
            return str(int(m.group(1)))
    return None


def _block_id(text: str) -> str:
    text = text.strip()
    return str(int(text)) if text.isdigit() else text.lower()


def _read_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    """Header (lower-cased, or empty) and the numeric rows of a CSV."""
    import csv

    with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if not rows:
        return [], np.empty((0, 0))
    header: List[str] = []
    if any(_to_float(c) is None for c in rows[0]):
        header, rows = [c.strip().lower() for c in rows[0]], rows[1:]
    width = max((len(r) for r in rows), default=0)
    values = np.array(
        [[(_to_float(c) if i < len(r) else None) for i, c in enumerate(r + [""] * (width - len(r)))] for r in rows],
        dtype=float,
    )
    return header, values


def _clas_block_types(root: Path) -> Dict[Tuple[str, str], str]:
    """``(participant, block number) -> block type`` from every ``Part<N>_Block_Details.csv``."""
    import csv

    mapping: Dict[Tuple[str, str], str] = {}
    for details in sorted(root.rglob("*_Block_Details.csv")):
        participant = _participant(details)
        with open(details, newline="", encoding="utf-8", errors="ignore") as fh:
            rows = [[c.strip() for c in r] for r in csv.reader(fh) if r]
        if participant is None or not rows:
            continue
        header = [h.lower() for h in rows[0]]
        block_col = next((i for i, h in enumerate(header) if h == "block"), None)
        type_col = next((i for i, h in enumerate(header) if "type" in h), None)
        if block_col is None or type_col is None:
            logger.warning("clas: %s has no Block / Block Type columns: %s", details, rows[0])
            continue
        for row in rows[1:]:
            if len(row) > max(block_col, type_col) and row[block_col] and row[type_col]:
                mapping[(participant, _block_id(row[block_col]))] = row[type_col].lower()
    return mapping


def _clas_blocks(root: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, Path]], List[str], List[str]]:
    """Per-block files grouped by ``(participant, block)``, the folders skipped for
    having no participant number, and the duplicate files left out."""
    blocks: Dict[Tuple[str, str], Dict[str, Path]] = {}
    no_participant: set = set()
    duplicates: List[str] = []
    for path in sorted(root.rglob("*.csv")):
        m = _CLAS_BLOCK_FILE.match(path.name)
        if not m:
            continue
        participant = _participant(path)
        if participant is None:
            no_participant.add(str(path.parent))
            continue
        files = blocks.setdefault((participant, _block_id(m.group("block"))), {})
        kind = m.group("kind").lower()
        if kind in files:  # a second copy of the same participant's block
            duplicates.append(f"{path} (kept {files[kind]})")
            continue
        files[kind] = path
    return blocks, sorted(no_participant), duplicates


def _clas_high(block_type: str) -> bool:
    return any(t in block_type for t in CLAS_HIGH_LOAD)


def clas_inventory(root: str | Path) -> Dict[str, object]:
    """What :func:`load_clas` would read, found without reading any signal.

    For a quick check before a long pre-training run.
    """
    root = Path(root)
    blocks, no_participant, duplicates = _clas_blocks(root)
    types = _clas_block_types(root)
    typed = {k: types[k] for k in blocks if k in types}
    participants = {p for p, _ in blocks}
    return {
        "participants": len(participants),
        "blocks": len(blocks),
        "typed_blocks": len(typed),
        "high_load_blocks": sum(_clas_high(t) for t in typed.values()),
        "block_types": dict(sorted(collections.Counter(typed.values()).items())),
        "participants_without_block_details": sorted(
            participants - {p for p, _ in typed}, key=lambda p: int(p) if p.isdigit() else -1
        ),
        "folders_without_participant": no_participant,
        "duplicate_files": len(duplicates),
    }


def _clas_signals(path: Path, kind: str) -> np.ndarray:
    """ECG, or GSR then PPG, from one per-block file; timestamps and the
    uncalibrated accelerometer are dropped."""
    header, values = _read_csv(path)
    if values.size == 0:
        return np.empty((0, 0))
    wanted = ("ecg",) if kind == "ecg" else ("gsr", "ppg")
    found = [next((i for i, h in enumerate(header) if name in h and "time" not in h), None) for name in wanted]
    cols = [i for i in found if i is not None]
    if len(cols) < len(wanted):
        cols = list(range(1, 1 + len(wanted)))  # documented order: timestamp, then the signals
    cols = [c for c in cols if c < values.shape[1]]
    x = values[:, cols]
    return x[np.isfinite(x).any(axis=1)]


def load_clas(root: str | Path, windowing: Optional[Windowing] = None) -> List[PhysiologyRecord]:
    """CLAS (Markova et al., 2019): per-block ECG and GSR/PPG files.

    Each block's type comes from ``Block_details``. The ECG and GSR/PPG files of
    a block are resampled and joined channel-wise over their common length.
    """
    w = windowing or Windowing()
    root = _require(Path(root), "clas")
    types = _clas_block_types(root)
    blocks, no_participant, duplicates = _clas_blocks(root)
    records: List[PhysiologyRecord] = []
    skipped: List[str] = [f"{d} (no participant number)" for d in no_participant] + duplicates
    for (participant, block), files in sorted(blocks.items()):
        block_type = types.get((participant, block))
        if block_type is None:
            skipped.append(f"participant {participant} block {block} (not in Block_Details)")
            continue
        try:
            parts = [_resample(_clas_signals(files[k], k), CLAS_RATE_HZ, w.rate_hz) for k in ("ecg", "gsr_ppg") if k in files]
        except Exception as exc:
            skipped.append(f"participant {participant} block {block} ({type(exc).__name__}: {exc})")
            continue
        parts = [p for p in parts if p.size]
        if not parts:
            skipped.append(f"participant {participant} block {block} (no signal columns)")
            continue
        length = min(len(p) for p in parts)
        x = np.concatenate([p[:length] for p in parts], axis=1)
        label, fatigue = ("high", 0.75) if _clas_high(block_type) else ("low", 0.20)
        records.extend(
            PhysiologyRecord(seg, fatigue, label, f"Part{participant}", "clas") for seg in _windows(x, w)
        )
    _summary("clas", records, skipped)
    return records


def _to_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except ValueError:
        return None


def load_public_corpora(
    root: str | Path = "data/raw", strict: bool = False, windowing: Optional[Windowing] = None
) -> List[PhysiologyRecord]:
    """Load whichever of the three corpora are present (Sec. 3.7 stage 1)."""
    root = Path(root)
    w = windowing or Windowing()
    loaders = {"mamem": load_mamem, "clas": load_clas, "wesad": load_wesad}
    out: List[PhysiologyRecord] = []
    for name, fn in loaders.items():
        try:
            out.extend(fn(root / name, w))
        except (FileNotFoundError, ImportError) as exc:
            if strict:
                raise
            logger.info("%s not loaded: %s", name, exc)
    return out


# --------------------------------------------------------------------------- #
def synthetic_pretraining_corpus(
    dims: InputDims,
    n_subjects: int = 100,
    windows_per_subject: int = 40,
    seed: int = 0,
) -> List[PhysiologyRecord]:
    """SYNTHETIC stand-in for the >100-participant pre-training pool (Sec. 3.3).

    Clearly labelled: ``source='synthetic'``. Any model pre-trained on this has
    NOT been pre-trained on MAMEM/CLAS/WESAD, and its fatigue estimates carry no
    claim about real physiology.
    """
    from .physiology import PhysiologySynthesiser

    rng = np.random.default_rng(seed)
    synth = PhysiologySynthesiser(dims)
    records: List[PhysiologyRecord] = []
    for s in range(n_subjects):
        # Per-subject offset stands in for inter-individual variability.
        bias = rng.normal(0.0, 0.05)
        for _ in range(windows_per_subject):
            fatigue = float(np.clip(rng.uniform(0.0, 1.0) + bias, 0.0, 1.0))
            arousal = float(np.clip(rng.beta(2, 4) + 0.3 * fatigue, 0.0, 1.0))
            signals = synth.physiology(fatigue, arousal, rng)
            label = LOAD_LABELS[min(int(fatigue * 3), 2)]
            records.append(
                PhysiologyRecord(signals, fatigue, label, f"synthetic-{s:03d}", "synthetic")
            )
    return records

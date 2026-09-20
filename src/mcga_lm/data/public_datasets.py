

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from ..config import InputDims

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


def _require(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found at {path}. It is not redistributed with this repository; "
            f"download it from DOI {DATASET_DOIS[name]} into {path}. See data/README.md."
        )
    return path


def load_wesad(root: str | Path) -> List[PhysiologyRecord]:  # pragma: no cover - UNVERIFIED
    """WESAD (Schmidt et al., 2018): per-subject ``S*/S*.pkl`` chest/wrist data."""
    import pickle

    root = _require(Path(root), "wesad")
    records: List[PhysiologyRecord] = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pkl = subject_dir / f"{subject_dir.name}.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as fh:
            data = pickle.load(fh, encoding="latin1")
        chest = data["signal"]["chest"]
        labels = np.asarray(data["label"])
        stacked = np.concatenate(
            [np.asarray(chest[k]).reshape(len(labels), -1) for k in ("ECG", "EDA", "EMG", "Resp")], axis=1
        )
        # WESAD protocol codes: 1 baseline, 2 stress, 3 amusement, 4 meditation.
        mapping = {1: ("low", 0.15), 2: ("high", 0.80), 3: ("medium", 0.40), 4: ("low", 0.20)}
        for code, (label, fatigue) in mapping.items():
            mask = labels == code
            if mask.sum() == 0:
                continue
            records.append(
                PhysiologyRecord(stacked[mask], fatigue, label, subject_dir.name, "wesad")
            )
    return records


def load_clas(root: str | Path) -> List[PhysiologyRecord]:  # pragma: no cover - UNVERIFIED
    """CLAS (Markova et al., 2019): per-participant CSV blocks."""
    import csv

    root = _require(Path(root), "clas")
    records: List[PhysiologyRecord] = []
    for csv_path in sorted(root.rglob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8", errors="ignore") as fh:
            rows = list(csv.reader(fh))
        if len(rows) < 2:
            continue
        values = np.asarray(
            [[_to_float(c) for c in row] for row in rows[1:]], dtype=float
        )
        task = csv_path.stem.lower()
        label, fatigue = ("high", 0.75) if "stroop" in task or "math" in task else ("low", 0.20)
        records.append(PhysiologyRecord(values, fatigue, label, csv_path.parent.name, "clas"))
    return records


def load_mamem(root: str | Path) -> List[PhysiologyRecord]:  # pragma: no cover - UNVERIFIED
    """MAMEM (Nikolopoulos et al., 2017): MATLAB session files with biosignals."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("loading MAMEM needs scipy") from exc

    root = _require(Path(root), "mamem")
    records: List[PhysiologyRecord] = []
    for mat_path in sorted(root.rglob("*.mat")):
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        key = next((k for k in mat if not k.startswith("__")), None)
        if key is None:
            continue
        signals = np.atleast_2d(np.asarray(mat[key], dtype=float))
        if signals.shape[0] < signals.shape[1]:
            signals = signals.T
        records.append(PhysiologyRecord(signals, 0.5, "medium", mat_path.stem, "mamem"))
    return records


def _to_float(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return 0.0


def load_public_corpora(root: str | Path = "data/raw", strict: bool = False) -> List[PhysiologyRecord]:
    """Load whichever of the three corpora are present (Sec. 3.7 stage 1)."""
    root = Path(root)
    loaders = {"mamem": load_mamem, "clas": load_clas, "wesad": load_wesad}
    out: List[PhysiologyRecord] = []
    for name, fn in loaders.items():
        try:
            out.extend(fn(root / name))
        except (FileNotFoundError, ImportError):
            if strict:
                raise
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

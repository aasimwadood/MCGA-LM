
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def get_logger(name: str = "mcga_lm") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def resolve_device(requested: str = "auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_encode)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _encode(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def mean_sd(values: Iterable[float]) -> Dict[str, float]:
    """Mean and *sample* SD. Paper Sec. 3.8: every reported ``+/-`` is an SD
    across the 20 personas, not across seeds."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "sd": sd, "n": int(arr.size)}


def median_iqr(values: Iterable[float]) -> Dict[str, float]:
    """Median and interquartile range (Table 7 reports a median [IQR] for SACT).

    Returns NaNs for an empty sample, which happens for the character-level
    baselines whose SACT is "n/r" in Table 7.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    return {
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
    }


def flatten(nested: Iterable[Iterable[Any]]) -> List[Any]:
    return [x for sub in nested for x in sub]

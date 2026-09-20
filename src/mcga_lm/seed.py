

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed ``random``, ``numpy`` and (if installed) ``torch``."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return an independent NumPy generator (preferred over global state)."""
    return np.random.default_rng(seed)


@contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Run a block under ``seed`` and restore the previous global RNG state."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = None
    try:
        import torch

        torch_state = torch.random.get_rng_state()
    except ImportError:  # pragma: no cover
        torch = None  # type: ignore[assignment]
    set_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch_state is not None:
            import torch as _torch

            _torch.random.set_rng_state(torch_state)

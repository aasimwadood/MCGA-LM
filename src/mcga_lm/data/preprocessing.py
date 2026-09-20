

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

EEG_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


# ------------------------------------------------------------------ EEG --- #
def bandpass(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass, 0.5-40 Hz (Sec. 4.1)."""
    from scipy.signal import butter, filtfilt

    nyq = fs / 2.0
    b, a = butter(order, [max(low / nyq, 1e-6), min(high / nyq, 0.999)], btype="band")
    axis = 0 if signal.ndim > 1 else -1
    return filtfilt(b, a, signal, axis=axis)


def window_signal(signal: np.ndarray, fs: float, seconds: float = 2.0, overlap: float = 0.5) -> np.ndarray:
    """Segment into ``seconds``-long windows with fractional ``overlap``."""
    size = int(round(seconds * fs))
    step = max(1, int(round(size * (1.0 - overlap))))
    n = signal.shape[0]
    starts = range(0, max(n - size + 1, 1), step)
    return np.stack([signal[s : s + size] for s in starts if s + size <= n]) if n >= size else np.empty((0, size, *signal.shape[1:]))


def log_band_power(window: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
    """Log band power for delta/theta/alpha/beta (Sec. 4.1)."""
    from scipy.signal import welch

    freqs, psd = welch(window, fs=fs, axis=0, nperseg=min(window.shape[0], int(fs)))
    out: Dict[str, np.ndarray] = {}
    for band, (lo, hi) in EEG_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        power = psd[mask].mean(axis=0) if mask.any() else np.zeros(psd.shape[1:])
        out[band] = np.log(np.asarray(power) + 1e-12)
    return out


def artifact_fraction(window: np.ndarray, z_threshold: float = 5.0) -> float:
    """Fraction of samples exceeding ``z_threshold`` -- the contamination proxy.

    TODO: replace with Artifact Subspace Reconstruction + ICA as in Sec. 4.1.
    """
    if window.size == 0:
        return 1.0
    mu, sd = window.mean(), window.std() + 1e-9
    return float((np.abs(window - mu) / sd > z_threshold).mean())


def reject_windows(windows: np.ndarray, max_contamination: float = 0.30) -> np.ndarray:
    """Drop windows with > 30% artifact contamination (Sec. 4.1, applied uniformly)."""
    if windows.size == 0:
        return windows
    keep = [i for i in range(windows.shape[0]) if artifact_fraction(windows[i]) <= max_contamination]
    return windows[keep]


# ------------------------------------------------------------------ EDA --- #
def decompose_eda(eda: np.ndarray, fs: float, cutoff: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Tonic / phasic decomposition (Sec. 4.1, continuous decomposition analysis).

    ASSUMPTION A-20: a low-pass split at 0.05 Hz stands in for cvxEDA-style CDA.
    """
    from scipy.signal import butter, filtfilt

    b, a = butter(2, min(cutoff / (fs / 2.0), 0.999), btype="low")
    tonic = filtfilt(b, a, eda, axis=0)
    return tonic, eda - tonic


def scr_rate(phasic: np.ndarray, fs: float, threshold: float = 0.01) -> float:
    """Skin-conductance-response rate per minute (Sec. 4.1 "SCR rate")."""
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(np.asarray(phasic).reshape(-1), height=threshold)
    seconds = len(phasic) / fs
    return float(len(peaks) / seconds * 60.0) if seconds > 0 else 0.0


# ------------------------------------------------------------------ HRV --- #
def correct_ectopic(rr_ms: np.ndarray, tolerance: float = 0.2) -> np.ndarray:
    """Remove R-R intervals deviating > 20% from the running median (Sec. 4.1)."""
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size < 3:
        return rr
    median = np.median(rr)
    keep = np.abs(rr - median) <= tolerance * median
    return rr[keep] if keep.any() else rr


def hrv_features(rr_ms: Sequence[float], fs_interp: float = 4.0) -> Dict[str, float]:
    """SDNN, RMSSD, pNN50, LF/HF over a 60-s window (Sec. 3.2, 4.1)."""
    rr = correct_ectopic(np.asarray(rr_ms, dtype=float))
    if rr.size < 2:
        return {"sdnn": 0.0, "rmssd": 0.0, "pnn50": 0.0, "lf_hf": 0.0}
    diff = np.diff(rr)
    features = {
        "sdnn": float(rr.std(ddof=1)),
        "rmssd": float(np.sqrt((diff**2).mean())),
        "pnn50": float((np.abs(diff) > 50).mean()),
        "lf_hf": _lf_hf(rr, fs_interp),
    }
    return features


def _lf_hf(rr_ms: np.ndarray, fs_interp: float) -> float:
    from scipy.signal import welch

    t = np.cumsum(rr_ms) / 1000.0
    if t[-1] <= 0 or rr_ms.size < 8:
        return 0.0
    grid = np.arange(0, t[-1], 1.0 / fs_interp)
    series = np.interp(grid, t, rr_ms)
    freqs, psd = welch(series - series.mean(), fs=fs_interp, nperseg=min(len(series), 256))
    lf = psd[(freqs >= 0.04) & (freqs < 0.15)].sum()
    hf = psd[(freqs >= 0.15) & (freqs < 0.40)].sum()
    return float(lf / hf) if hf > 0 else 0.0


# ----------------------------------------------------------------- gaze --- #
@dataclass
class GazeFeatures:
    fixation_rate: float
    saccade_rate: float
    blink_rate: float
    mean_pupil: float


def ivt_filter(
    gaze_xy: np.ndarray, fs: float, velocity_threshold: float = 30.0, pupil: Optional[np.ndarray] = None
) -> GazeFeatures:
    """IV-T velocity filter -> fixation / saccade / blink features (Sec. 4.1).

    ``velocity_threshold`` is in degrees/second; gaze is assumed to be in degrees
    of visual angle (ASSUMPTION A-21 -- the paper gives coordinates, not units).
    """
    xy = np.asarray(gaze_xy, dtype=float)
    if xy.shape[0] < 2:
        return GazeFeatures(0.0, 0.0, 0.0, float(np.mean(pupil)) if pupil is not None else 0.0)
    velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fs
    saccade = velocity > velocity_threshold
    seconds = xy.shape[0] / fs
    transitions = int(np.sum(np.diff(saccade.astype(int)) == 1))
    fixations = int(np.sum(np.diff((~saccade).astype(int)) == 1))
    blinks = int(np.sum(np.isnan(xy).any(axis=1))) if np.isnan(xy).any() else 0
    return GazeFeatures(
        fixation_rate=fixations / seconds * 60.0,
        saccade_rate=transitions / seconds * 60.0,
        blink_rate=blinks / seconds * 60.0,
        mean_pupil=float(np.nanmean(pupil)) if pupil is not None else 0.0,
    )


# ------------------------------------------------------------ normalise --- #
def zscore_to_baseline(signal: np.ndarray, baseline: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-persona z-scoring against a session-start resting baseline (Sec. 4.1)."""
    mu = baseline.mean(axis=0, keepdims=True)
    sd = baseline.std(axis=0, keepdims=True) + eps
    return (signal - mu) / sd


def tft_windows(features: np.ndarray, fs: float, seconds: float = 30.0, stride_s: float = 5.0) -> np.ndarray:
    """T = 30 s sliding windows with a 5-s stride, as fed to the TFT (Sec. 4.1)."""
    size = int(round(seconds * fs))
    step = max(1, int(round(stride_s * fs)))
    n = features.shape[0]
    if n < size:
        return np.empty((0, size, features.shape[1]))
    return np.stack([features[s : s + size] for s in range(0, n - size + 1, step)])

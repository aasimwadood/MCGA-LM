
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import InputDims


@dataclass
class FatigueModel:
    """Exponential rise in cognitive load over a session (Sec. 4.5)."""

    half_life_min: float = 30.0
    peak: float = 0.8
    baseline: float = 0.05
    noise: float = 0.02

    def value(self, minutes: float | np.ndarray, rng: Optional[np.random.Generator] = None):
        """f(t) = baseline + (peak - baseline) * (1 - 2^{-t / half_life})."""
        t = np.asarray(minutes, dtype=float)
        risen = 1.0 - np.power(0.5, t / max(self.half_life_min, 1e-6))
        f = self.baseline + (self.peak - self.baseline) * risen
        if rng is not None and self.noise > 0:
            f = f + rng.normal(0.0, self.noise, size=f.shape)
        return np.clip(f, 0.0, 1.0)

    def adapted_value(self, minutes, rng=None, relief: float = 0.25, onset_min: float = 20.0):
        """Fatigue trajectory under TFT-driven adaptation (Fig. 5).

        Once adaptation activates (the paper's Fig. 5 marks an activation point),
        the effort per turn falls, so the accumulated fatigue rises more slowly.
        ASSUMPTION A-15: modelled as a multiplicative relief on the increment
        past ``onset_min``.
        """
        t = np.asarray(minutes, dtype=float)
        base = self.value(t, rng=None)
        before = self.value(np.minimum(t, onset_min), rng=None)
        adapted = np.where(t <= onset_min, base, before + (base - before) * (1.0 - relief))
        if rng is not None and self.noise > 0:
            adapted = adapted + rng.normal(0.0, self.noise, size=adapted.shape)
        return np.clip(adapted, 0.0, 1.0)


class PhysiologySynthesiser:
    """Generates the ``x_phys`` / ``x_beh`` arrays of Eq. (2) from a fatigue level.

    Channel layout matches Sec. 3.2 exactly:
      x_phys = [EEG_1..EEG_4 | SDNN, RMSSD, LF/HF | EDA_tonic, EDA_phasic]
      x_beh  = [gaze_x, gaze_y, pupil_diameter, blink_rate]
    All values are already z-scored per user against a session-start resting
    baseline (Sec. 4.1 preprocessing), so they are dimensionless.
    """

    def __init__(
        self,
        dims: InputDims,
        reduced_sensor_set: bool = False,
        switchboard: Optional["SensorSwitchboard"] = None,
    ) -> None:
        from ..privacy import SensorSwitchboard

        self.dims = dims
        # Sec. 4.6 / 5.6: consumer configuration is EDA + PPG + monocular gaze,
        # i.e. EEG channels are unavailable. That is the same thing as Sec. 3.9's
        # worked example of a user switching EEG off, so both go through one
        # mechanism: the switchboard. ``reduced_sensor_set=True`` is shorthand
        # for the EEG-off switchboard, kept for backwards compatibility.
        self.reduced_sensor_set = reduced_sensor_set
        if switchboard is None:
            switchboard = (
                SensorSwitchboard.reduced_consumer_set() if reduced_sensor_set else SensorSwitchboard()
            )
        self.switchboard = switchboard
        self.monocular_gaze = reduced_sensor_set  # Table 5: webcam gaze is monocular

    def physiology(self, fatigue: float, arousal: float, rng: np.random.Generator) -> np.ndarray:
        d = self.dims
        t = np.arange(d.phys_window, dtype=float) / d.phys_rate_hz
        out = np.zeros((d.phys_window, d.phys_dim), dtype=np.float32)

        # --- EEG (4 channels): frontal theta power rises with fatigue -------
        for ch in range(d.eeg_channels):
            theta = np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi)) * (0.4 + 1.2 * fatigue)
            alpha = np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi)) * (0.8 - 0.3 * fatigue)
            beta = np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, 2 * np.pi)) * (0.5 * (1.0 - fatigue) + 0.3 * arousal)
            out[:, ch] = theta + alpha + beta + rng.normal(0, 0.25, d.phys_window)

        # --- HRV: SDNN and RMSSD fall, LF/HF rises with fatigue -------------
        base = d.eeg_channels
        out[:, base + 0] = -1.4 * fatigue + rng.normal(0, 0.15, d.phys_window)  # SDNN
        out[:, base + 1] = -1.1 * fatigue - 0.3 * arousal + rng.normal(0, 0.15, d.phys_window)  # RMSSD
        out[:, base + 2] = 1.3 * fatigue + 0.5 * arousal + rng.normal(0, 0.2, d.phys_window)  # LF/HF

        # --- EDA: tonic level rises with arousal and (weakly) fatigue -------
        base = d.eeg_channels + d.hrv_features
        out[:, base + 0] = 1.1 * arousal + 0.45 * fatigue + rng.normal(0, 0.12, d.phys_window)
        phasic = np.maximum(0.0, rng.normal(0.0, 0.35, d.phys_window)) * (0.6 + 1.0 * arousal)
        out[:, base + 1] = phasic
        # Sec. 3.9: a disabled sensor contributes nothing. Zeroing the channels
        # rather than reshaping keeps ``X_t`` the shape the encoder was trained
        # on, which is what Sec. 3.2 relies on the Perceiver to tolerate.
        return out * self.switchboard.phys_mask(d)

    def behaviour(self, fatigue: float, arousal: float, rng: np.random.Generator) -> np.ndarray:
        d = self.dims
        out = np.zeros((d.beh_window, d.beh_features), dtype=np.float32)
        drift = np.cumsum(rng.normal(0, 0.25 + 0.4 * fatigue, d.beh_window))
        out[:, 0] = drift  # gaze x
        out[:, 1] = np.cumsum(rng.normal(0, 0.25 + 0.4 * fatigue, d.beh_window))  # gaze y
        out[:, 2] = -0.9 * fatigue + 0.7 * arousal + rng.normal(0, 0.12, d.beh_window)  # pupil
        out[:, 3] = 1.5 * fatigue + rng.normal(0, 0.15, d.beh_window)  # blink rate
        return out * self.switchboard.beh_mask(d, monocular=self.monocular_gaze)


def environment_vector(
    dims: InputDims,
    location_id: int,
    partner_id: int,
    noise_db: float,
    time_of_day: float,
) -> np.ndarray:
    """``x_env``: location one-hot | partner one-hot | noise, sin/cos time (Sec. 3.2, 4.1)."""
    vec = np.zeros(dims.env_dim, dtype=np.float32)
    vec[location_id % dims.env_location_classes] = 1.0
    vec[dims.env_location_classes + (partner_id % dims.env_partner_classes)] = 1.0
    tail = dims.env_location_classes + dims.env_partner_classes
    vec[tail + 0] = (noise_db - 45.0) / 15.0  # z-ish scaling of dBA
    vec[tail + 1] = np.sin(2 * np.pi * time_of_day)  # cyclic time encoding, Sec. 4.1
    vec[tail + 2] = np.cos(2 * np.pi * time_of_day)
    return vec

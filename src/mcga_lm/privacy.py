

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

# The sensors Sec. 3.9 says a user may independently disable. "environment"
# covers the GPS/noise/time/partner vector of Sec. 3.2, which is as personal as
# the physiological streams and is listed in Sec. 3.9's opening sentence.
SENSORS = ("eeg", "hrv", "eda", "gaze", "environment")

# Sec. 3.9's own worked example, and the only sensor-disable effect the paper
# quantifies. Reported as a post-hoc observation, not a pre-registered endpoint.
PAPER_EEG_DISABLED_SACT = (3.1, 4.2)

MEDICAL_DEVICE_DISCLAIMER = (
    "MCGA-LM is not a medical device. It does not perform health monitoring, "
    "diagnosis, or alerting for any condition, including seizures, arrhythmias "
    "and sleep apnoea. Physiological sensing is used only to estimate "
    "communicative effort and adapt the interface. Do not rely on this system "
    "to detect or report a medical event."
)

# Capabilities the disclaimer rules out. assert_not_medical_device() exists so a
# future contributor cannot add one of these without the test suite objecting.
PROHIBITED_CAPABILITIES = (
    "seizure_detection",
    "arrhythmia_detection",
    "sleep_apnoea_detection",
    "diagnosis",
    "health_alerting",
    "clinical_monitoring",
)


# --------------------------------------------------------------- sensing --- #
@dataclass
class SensorWarning:
    """The persistent, dismissible, session-recurring warning of Sec. 3.9.

    "Persistent" and "session-recurring" are the paper's words: dismissing the
    warning silences it for the rest of the session and it returns at the next
    one. :meth:`dismiss` models that, and :attr:`dismissed` is reset by
    :meth:`SensorSwitchboard.begin_session`.
    """

    sensor: str
    message: str
    dismissed: bool = False

    def dismiss(self) -> None:
        self.dismissed = True

    @property
    def visible(self) -> bool:
        return not self.dismissed


@dataclass
class SensorSwitchboard:
    """Per-sensor enable/disable, as Sec. 3.9's settings interface (Sec. 3.9).

    The paper's example is EEG, but it says *each* sensor, so every stream in
    Eq. (2) gets a switch. Disabling a sensor zeroes its channels rather than
    changing the tensor layout: the encoder is trained on a fixed ``X_t`` shape
    and Sec. 3.2 already relies on the Perceiver to "implicitly weight
    informative modalities and ignore noisy or missing channels".

    >>> sw = SensorSwitchboard()
    >>> w = sw.disable("eeg")
    >>> w.visible and "adaptation accuracy" in w.message
    True
    >>> sw.enabled("eeg")
    False
    """

    disabled: set = field(default_factory=set)
    warnings: Dict[str, SensorWarning] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        unknown = self.disabled - set(SENSORS)
        if unknown:
            raise ValueError(f"unknown sensor(s) {sorted(unknown)}; expected {list(SENSORS)}")

    @classmethod
    def reduced_consumer_set(cls) -> "SensorSwitchboard":
        """Table 5's consumer configuration: EDA + PPG(HRV) + monocular gaze.

        Sec. 4.6 / 5.6. EEG is the sensor that is "clinical/research only", so
        the consumer configuration is exactly this switchboard with EEG off --
        which is also Sec. 3.9's worked example, and why both report the same
        SACT of 4.2.
        """
        return cls(disabled={"eeg"})

    # ------------------------------------------------------------------ #
    def enabled(self, sensor: str) -> bool:
        self._check(sensor)
        return sensor not in self.disabled

    def disable(self, sensor: str) -> SensorWarning:
        """Disable a sensor and raise its warning (Sec. 3.9)."""
        self._check(sensor)
        self.disabled.add(sensor)
        warning = SensorWarning(sensor=sensor, message=self._warning_text(sensor))
        self.warnings[sensor] = warning
        return warning

    def enable(self, sensor: str) -> None:
        self._check(sensor)
        self.disabled.discard(sensor)
        self.warnings.pop(sensor, None)

    def begin_session(self) -> List[SensorWarning]:
        """Start a session: every still-disabled sensor re-raises its warning.

        This is what "session-recurring" means -- a warning dismissed yesterday
        does not stay dismissed today.
        """
        for sensor in self.disabled:
            self.warnings[sensor] = SensorWarning(sensor=sensor, message=self._warning_text(sensor))
        return self.active_warnings()

    def active_warnings(self) -> List[SensorWarning]:
        return [w for w in self.warnings.values() if w.visible]

    # ------------------------------------------------------------------ #
    def _warning_text(self, sensor: str) -> str:
        if sensor == "eeg":
            low, high = PAPER_EEG_DISABLED_SACT
            return (
                "EEG sensing is off. Fatigue estimation now uses EDA and HRV only, "
                "and adaptation accuracy may decrease: the paper reports switch "
                f"activations per turn rising from {low} to {high} in post-hoc "
                "analysis (Sec. 3.9). Communication still works; it may take more "
                "switch presses."
            )
        return (
            f"{sensor.upper()} sensing is off. Fatigue and context estimation lose this "
            "input, and adaptation accuracy may decrease (Sec. 3.9). Communication "
            "still works; it may take more switch presses."
        )

    @staticmethod
    def _check(sensor: str) -> None:
        if sensor not in SENSORS:
            raise ValueError(f"unknown sensor {sensor!r}; expected one of {list(SENSORS)}")

    # ------------------------------------------------------------------ #
    def phys_mask(self, dims) -> np.ndarray:
        """Channel mask over ``x_phys`` = [EEG | SDNN, RMSSD, LF/HF | tonic, phasic]."""
        mask = np.ones(dims.phys_dim, dtype=np.float32)
        base = 0
        if not self.enabled("eeg"):
            mask[base : base + dims.eeg_channels] = 0.0
        base += dims.eeg_channels
        if not self.enabled("hrv"):
            mask[base : base + dims.hrv_features] = 0.0
        base += dims.hrv_features
        if not self.enabled("eda"):
            mask[base : base + dims.eda_features] = 0.0
        return mask

    def beh_mask(self, dims, monocular: bool = False) -> np.ndarray:
        """Channel mask over ``x_beh`` = [gaze_x, gaze_y, pupil, blink].

        ``monocular=True`` is Table 5's consumer gaze: a webcam gives one eye, so
        the binocular-only channel (pupil diameter, which the paper's Table 5
        pairs with a 60 Hz binocular tracker) is dropped while the gaze point and
        blink rate survive.
        """
        mask = np.ones(dims.beh_features, dtype=np.float32)
        if not self.enabled("gaze"):
            mask[:] = 0.0
        elif monocular and dims.beh_features > 2:
            mask[2] = 0.0
        return mask

    def env_mask(self, dims) -> np.ndarray:
        """Channel mask over ``x_env`` (location, partner, noise, time-of-day)."""
        return np.full(dims.env_dim, 1.0 if self.enabled("environment") else 0.0, dtype=np.float32)

    def as_dict(self) -> Dict[str, bool]:
        return {s: self.enabled(s) for s in SENSORS}

    def summary(self) -> str:
        off = sorted(self.disabled)
        return "all sensors enabled" if not off else f"disabled: {', '.join(off)}"


# ---------------------------------------------------------------- erasure -- #
class ErasedStoreError(RuntimeError):
    """Raised when a user store is read after its key has been destroyed."""


@dataclass
class EncryptedUserStore:
    """Key-wrapped on-device store with an irreversible erase (Sec. 3.9).

    Holds the three artefacts Sec. 3.9 names -- the personal intent graph, the
    LoRA adapter weights, and stored physiological features -- as ciphertext
    under a single per-device key. :meth:`crypto_erase` destroys that key, after
    which no read can succeed: the ciphertext may survive on the medium, but
    nothing can turn it back into data. That is what "cryptographically and
    irreversibly erases" buys over deleting files.

    NOT A SECURITY IMPLEMENTATION. The cipher is a keyed-hash stream built from
    ``hashlib``, chosen so the repository takes no cryptographic dependency; it
    is adequate to demonstrate the erase *semantics* and nothing more. A
    deployment needs an authenticated cipher and a key held in a secure element.
    See ASSUMPTION A-36.
    """

    _key: Optional[bytes] = field(default=None, repr=False)
    _blobs: Dict[str, bytes] = field(default_factory=dict, repr=False)
    erased: bool = False
    require_hardware_confirmation: bool = True

    def __post_init__(self) -> None:
        if self._key is None:
            self._key = os.urandom(32)

    # ------------------------------------------------------------------ #
    def put(self, name: str, payload: object) -> None:
        """Store an artefact (graph, adapter weights, physiological features)."""
        self._require_live()
        raw = json.dumps(payload, sort_keys=True, default=_jsonable).encode("utf-8")
        self._blobs[name] = _xor_stream(raw, self._key, name)

    def get(self, name: str) -> object:
        self._require_live()
        if name not in self._blobs:
            raise KeyError(name)
        return json.loads(_xor_stream(self._blobs[name], self._key, name).decode("utf-8"))

    def names(self) -> List[str]:
        self._require_live()
        return sorted(self._blobs)

    def ciphertext_size(self) -> int:
        """Bytes still on the medium -- non-zero even after erase, by design."""
        return sum(len(b) for b in self._blobs.values())

    # ------------------------------------------------------------------ #
    def crypto_erase(self, hardware_confirmed: bool = False) -> Dict[str, object]:
        """Destroy the key, rendering every stored artefact unrecoverable.

        Sec. 3.9 specifies "a physical switch (or software button with hardware
        confirmation)", so the confirmation is required rather than implied;
        pass ``hardware_confirmed=True`` to stand in for the switch. Set
        ``require_hardware_confirmation=False`` only in tests.
        """
        if self.require_hardware_confirmation and not hardware_confirmed:
            raise PermissionError(
                "Sec. 3.9 requires a physical switch or hardware-confirmed button "
                "before erasure; call with hardware_confirmed=True"
            )
        residual = self.ciphertext_size()
        n = len(self._blobs)
        self._key = None
        self._blobs.clear()
        self.erased = True
        return {
            "erased": True,
            "artefacts_destroyed": n,
            "key_destroyed": True,
            "residual_ciphertext_bytes": residual,
            "recoverable": False,
            "note": (
                "The key is gone, so the ciphertext is unrecoverable. Bytes that "
                "were already copied off-device by a backup are outside this "
                "guarantee -- Sec. 3.9's claim is about the device."
            ),
        }

    def _require_live(self) -> None:
        if self.erased or self._key is None:
            raise ErasedStoreError(
                "this user store was cryptographically erased (Sec. 3.9); its key no longer exists"
            )


def _jsonable(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "summary"):
        return obj.summary()
    raise TypeError(f"cannot serialise {type(obj)!r} into the user store")


def _xor_stream(data: bytes, key: bytes, label: str) -> bytes:
    """Keyed-hash stream cipher -- demonstration only, see the class docstring."""
    out = bytearray()
    counter = 0
    while len(out) < len(data):
        block = hmac.new(key, f"{label}:{counter}".encode("utf-8"), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(a ^ b for a, b in zip(data, out[: len(data)]))


def erase_user_data(
    store: EncryptedUserStore,
    graph=None,
    adapters: Optional[Mapping[str, object]] = None,
    hardware_confirmed: bool = False,
) -> Dict[str, object]:
    """Erase everything Sec. 3.9 names, in-memory copies included.

    The store's key is destroyed, and any live in-memory ``graph`` and
    ``adapters`` handed in are cleared too -- a crypto-erase that leaves the
    graph sitting in RAM would not be the guarantee the paper describes.
    """
    report = store.crypto_erase(hardware_confirmed=hardware_confirmed)
    cleared: List[str] = []
    if graph is not None:
        for attr in ("nodes", "edges"):
            container = getattr(graph, attr, None)
            if container is not None and hasattr(container, "clear"):
                container.clear()
                cleared.append(f"graph.{attr}")
        index = getattr(graph, "_index", None)
        if index is not None and hasattr(index, "clear"):
            index.clear()
            cleared.append("graph._index")
    if adapters is not None:
        for name, tensor in adapters.items():
            if hasattr(tensor, "zero_"):
                tensor.zero_()
                cleared.append(f"adapter.{name}")
    report["in_memory_cleared"] = cleared
    return report


# -------------------------------------------------- not a medical device --- #
def assert_not_medical_device(capabilities: Iterable[str] = ()) -> None:
    """Fail if the system claims any capability the Sec. 3.9 disclaimer denies.

    Sec. 3.9 tells users MCGA-LM does no health monitoring, diagnosis or
    alerting. That is a promise about the product, so it belongs in the test
    suite: ``tests/test_privacy.py`` calls this with the pipeline's advertised
    capabilities, and adding seizure detection later will break the build rather
    than silently contradict the setup screen.
    """
    offending = sorted(set(c.lower() for c in capabilities) & set(PROHIBITED_CAPABILITIES))
    if offending:
        raise AssertionError(
            f"Sec. 3.9 states MCGA-LM is not a medical device, but these "
            f"capabilities are claimed: {offending}. Either remove them or "
            f"remove the disclaimer -- they cannot both stand."
        )


def setup_disclosures(switchboard: Optional[SensorSwitchboard] = None) -> Sequence[str]:
    """The text shown during setup (Sec. 3.9: "users are informed of this")."""
    lines = [MEDICAL_DEVICE_DISCLAIMER, ON_DEVICE_PROCESSING_NOTICE, DATA_DELETION_NOTICE]
    if switchboard is not None:
        lines.append(f"Sensor status: {switchboard.summary()}.")
        lines.extend(w.message for w in switchboard.active_warnings())
    return tuple(lines)


ON_DEVICE_PROCESSING_NOTICE = (
    "All physiological, gaze and environmental features are processed on-device. "
    "Raw signals are never transmitted off-device; only anonymised summary "
    "metrics (the fatigue index and utterance embeddings) are stored, locally "
    "(Sec. 3.9)."
)

DATA_DELETION_NOTICE = (
    "You can erase everything this device knows about you at any time. The "
    "deletion switch destroys the encryption key for your intent graph, your "
    "personalised adapter weights and your stored physiological features, which "
    "makes them permanently unrecoverable (Sec. 3.9)."
)

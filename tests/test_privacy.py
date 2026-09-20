"""Privacy, safety and user control (paper Sec. 3.9).

Sec. 3.9 is a set of promises to the user rather than a model component, so the
tests here check that the promises hold and, where the paper quantifies one,
that the quantity is the one the paper names.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcga_lm.config import InputDims
from mcga_lm.data.physiology import PhysiologySynthesiser
from mcga_lm.privacy import (
    MEDICAL_DEVICE_DISCLAIMER,
    PAPER_EEG_DISABLED_SACT,
    PROHIBITED_CAPABILITIES,
    SENSORS,
    EncryptedUserStore,
    ErasedStoreError,
    SensorSwitchboard,
    assert_not_medical_device,
    erase_user_data,
    setup_disclosures,
)


# ------------------------------------------------------- sensor control --- #
def test_every_sensor_can_be_independently_disabled() -> None:
    """Sec. 3.9: "Each sensor can be independently disabled"."""
    for sensor in SENSORS:
        sw = SensorSwitchboard()
        sw.disable(sensor)
        assert not sw.enabled(sensor)
        assert all(sw.enabled(other) for other in SENSORS if other != sensor)


def test_unknown_sensor_is_rejected() -> None:
    with pytest.raises(ValueError):
        SensorSwitchboard().disable("telepathy")


def test_disabling_eeg_reverts_the_tft_to_eda_plus_hrv() -> None:
    """Sec. 3.9: "Disabling EEG ... reverts the TFT to EDA+HRV only"."""
    dims = InputDims()
    sw = SensorSwitchboard()
    sw.disable("eeg")
    mask = sw.phys_mask(dims)
    assert mask[: dims.eeg_channels].sum() == 0.0, "EEG channels must be silenced"
    assert mask[dims.eeg_channels :].all(), "HRV and EDA must survive"


def test_disabling_eeg_zeroes_the_eeg_channels_of_x_phys() -> None:
    dims = InputDims()
    rng = np.random.default_rng(0)
    on = PhysiologySynthesiser(dims).physiology(0.5, 0.5, np.random.default_rng(0))
    off = PhysiologySynthesiser(
        dims, switchboard=SensorSwitchboard(disabled={"eeg"})
    ).physiology(0.5, 0.5, rng)
    assert np.any(on[:, : dims.eeg_channels] != 0.0)
    assert np.all(off[:, : dims.eeg_channels] == 0.0)
    assert np.any(off[:, dims.eeg_channels :] != 0.0), "non-EEG channels still carry signal"


def test_the_eeg_warning_names_the_papers_own_sact_figures() -> None:
    """Sec. 3.9 quantifies exactly one cost: "SACT increase from 3.1 to 4.2"."""
    warning = SensorSwitchboard().disable("eeg")
    low, high = PAPER_EEG_DISABLED_SACT
    assert str(low) in warning.message and str(high) in warning.message
    assert "adaptation accuracy" in warning.message


def test_warning_is_dismissible_and_recurs_next_session() -> None:
    """Sec. 3.9: "a persistent (dismissible, session-recurring) warning"."""
    sw = SensorSwitchboard()
    warning = sw.disable("eeg")
    assert warning.visible
    warning.dismiss()
    assert not warning.visible and sw.active_warnings() == []

    reraised = sw.begin_session()
    assert len(reraised) == 1 and reraised[0].visible, "the warning must return next session"


def test_re_enabling_a_sensor_clears_its_warning() -> None:
    sw = SensorSwitchboard()
    sw.disable("eda")
    sw.enable("eda")
    assert sw.enabled("eda") and sw.active_warnings() == []


def test_reduced_consumer_set_is_the_eeg_off_switchboard() -> None:
    """Table 5's consumer config and Sec. 3.9's example are the same state."""
    reduced = SensorSwitchboard.reduced_consumer_set()
    assert not reduced.enabled("eeg")
    assert all(reduced.enabled(s) for s in SENSORS if s != "eeg")


def test_reduced_sensor_set_flag_still_silences_eeg() -> None:
    """Backwards compatibility: the old boolean routes through the switchboard."""
    dims = InputDims()
    phys = PhysiologySynthesiser(dims, reduced_sensor_set=True).physiology(
        0.5, 0.5, np.random.default_rng(1)
    )
    assert np.all(phys[:, : dims.eeg_channels] == 0.0)


def test_disabling_environment_blanks_the_env_mask() -> None:
    dims = InputDims()
    sw = SensorSwitchboard(disabled={"environment"})
    assert np.all(sw.env_mask(dims) == 0.0)
    assert np.all(SensorSwitchboard().env_mask(dims) == 1.0)


def test_turn_encoder_applies_the_switchboard_to_x_env(torch_mod, small_cfg) -> None:
    """Sec. 3.9 has to reach the tensors, not just the settings object."""
    from mcga_lm.data.dataset import TurnEncoder
    from mcga_lm.data.personas import build_persona_suite

    persona = build_persona_suite(small_cfg.simulation, small_cfg.graph, seed=0)[0]
    turn = persona.simulate_session(seed=0, n_turns=1)[0]
    rng = np.random.default_rng(0)

    on = TurnEncoder(small_cfg.inputs).encode_turn(turn, rng)
    off = TurnEncoder(
        small_cfg.inputs, switchboard=SensorSwitchboard(disabled={"environment"})
    ).encode_turn(turn, np.random.default_rng(0))

    assert np.any(on["env"] != 0.0)
    assert np.all(off["env"] == 0.0)
    assert np.any(off["phys"] != 0.0), "disabling one sensor must not silence the others"


def test_disabling_gaze_blanks_x_beh() -> None:
    dims = InputDims()
    sw = SensorSwitchboard(disabled={"gaze"})
    beh = PhysiologySynthesiser(dims, switchboard=sw).behaviour(0.5, 0.5, np.random.default_rng(2))
    assert np.all(beh == 0.0)


# --------------------------------------------------------- data deletion --- #
def test_store_round_trips_before_erasure() -> None:
    store = EncryptedUserStore()
    store.put("graph", {"nodes": ["User", "pain"]})
    store.put("lora", {"weights": [0.1, 0.2]})
    assert store.get("graph") == {"nodes": ["User", "pain"]}
    assert store.names() == ["graph", "lora"]


def test_stored_bytes_are_not_plaintext() -> None:
    store = EncryptedUserStore()
    store.put("graph", {"nodes": ["Nurse_Smith"]})
    assert b"Nurse_Smith" not in store._blobs["graph"]


def test_erasure_requires_hardware_confirmation() -> None:
    """Sec. 3.9: "a physical switch (or software button with hardware confirmation)"."""
    store = EncryptedUserStore()
    store.put("graph", {"a": 1})
    with pytest.raises(PermissionError):
        store.crypto_erase()
    assert store.get("graph") == {"a": 1}, "a refused erase must not destroy anything"


def test_crypto_erase_is_irreversible() -> None:
    """Sec. 3.9: "cryptographically and irreversibly erases all user data"."""
    store = EncryptedUserStore()
    store.put("graph", {"nodes": ["pain"]})
    store.put("lora", {"weights": [1.0]})
    store.put("physiology", {"fatigue": [0.1, 0.2]})

    report = store.crypto_erase(hardware_confirmed=True)
    assert report["erased"] and report["key_destroyed"] and not report["recoverable"]
    assert report["artefacts_destroyed"] == 3

    for name in ("graph", "lora", "physiology"):
        with pytest.raises(ErasedStoreError):
            store.get(name)
    with pytest.raises(ErasedStoreError):
        store.put("graph", {"nodes": []})


def test_erase_user_data_also_clears_the_live_graph() -> None:
    """A crypto-erase that left the graph in RAM would not be the guarantee."""
    from mcga_lm.memory.graph import IntentMemoryGraph, SemanticFrame

    graph = IntentMemoryGraph()
    graph.update_from_frame(SemanticFrame(obj="water", obj_type="Object", partner="Nurse"))
    assert len(graph.nodes) > 0

    store = EncryptedUserStore()
    store.put("graph", graph.to_dict())
    report = erase_user_data(store, graph=graph, hardware_confirmed=True)

    assert len(graph.nodes) == 0 and len(graph.edges) == 0
    assert "graph.nodes" in report["in_memory_cleared"]


# --------------------------------------------------- not a medical device --- #
def test_the_pipeline_claims_no_medical_capability() -> None:
    """Sec. 3.9 tells users the system does no monitoring, diagnosis or alerting.

    MCGA-LM's advertised outputs are an utterance, a fatigue index and a
    confidence. None of them is a clinical claim, and this test exists so that
    adding one later breaks the build rather than silently contradicting the
    setup screen.
    """
    assert_not_medical_device(["utterance_generation", "fatigue_estimation", "uncertainty_gating"])


@pytest.mark.parametrize("capability", PROHIBITED_CAPABILITIES)
def test_prohibited_capabilities_are_rejected(capability: str) -> None:
    with pytest.raises(AssertionError):
        assert_not_medical_device([capability])


def test_setup_discloses_all_three_commitments() -> None:
    """Sec. 3.9: "users are informed of this during setup"."""
    text = "\n".join(setup_disclosures(SensorSwitchboard()))
    assert MEDICAL_DEVICE_DISCLAIMER in text
    assert "on-device" in text.lower()
    assert "erase" in text.lower()

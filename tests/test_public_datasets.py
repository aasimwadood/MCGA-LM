"""Public pre-training corpora (paper Sec. 4.1, Table 4): loaders on fixture files.

Each fixture mirrors its dataset's real layout: MAMEM's Lab Streaming Layer
export (a cell array of stream structs per session), WESAD's per-subject pickle,
and CLAS's ``Data`` / ``Block_details`` / ``Answers`` folders.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from mcga_lm.data import public_datasets as P

W = P.Windowing(rate_hz=64, window=128, stride=64)


def _n_windows(seconds: float) -> int:
    return int((seconds * 64 - 128) // 64 + 1)


# ------------------------------------------------------------------ MAMEM -- #
def _stream(kind: str, rate: float, channels: int, seconds: float, t0: float, numeric: bool = True) -> dict:
    n = int(rate * seconds) if rate else 50
    series = np.random.default_rng(0).normal(size=(channels, n)) if numeric else np.array(["m"] * n, dtype=object)
    return {
        "info": {"type": kind, "nominal_srate": str(rate), "name": kind},
        "time_series": series,
        "time_stamps": t0 + np.arange(n) / (rate or 1.0),
    }


def _write_mamem(path, streams) -> None:
    from scipy.io import savemat

    cell = np.empty(len(streams), dtype=object)
    cell[:] = streams
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {"streams": cell})


def test_mamem_keeps_eeg_and_bio_aligned_and_windowed(tmp_path) -> None:
    root = tmp_path / "mamem"
    _write_mamem(root / "SHEBA" / "SH5" / "SHEBA_SH5_Light_ERRP.mat", [
        _stream("Gaze", 30, 2, 12, t0=100.0),
        _stream("EEG", 128, 14, 12, t0=100.0),
        _stream("BIO", 256, 2, 11, t0=101.0),  # starts 1 s later: only the shared 11 s is kept
        _stream("Markers", 0, 1, 0, t0=100.0, numeric=False),
    ])
    (root / "__MACOSX").mkdir()
    (root / "__MACOSX" / "._SHEBA_SH5_Light_ERRP.mat").write_bytes(b"not a mat file")

    records = P.load_mamem(root, W)
    assert len(records) == _n_windows(11)
    assert {r.signals.shape for r in records} == {(128, 16)}  # 14 EEG + 2 BIO; gaze dropped
    assert {(r.source, r.load_label, r.fatigue, r.subject) for r in records} == {("mamem", "medium", 0.5, "SHEBA/SH5")}


def test_mamem_reads_the_session_whatever_its_variable_is_called(tmp_path) -> None:
    from scipy.io import savemat

    cell = np.empty(1, dtype=object)
    cell[:] = [_stream("BIO", 256, 2, 6, t0=0.0)]
    (tmp_path / "AUTH" / "PH1").mkdir(parents=True)
    savemat(tmp_path / "AUTH" / "PH1" / "AUTH_PH1_GTW.mat", {"gtw": cell})
    assert len(P.load_mamem(tmp_path, W)) == _n_windows(6)


def test_an_unreadable_session_is_skipped_not_fatal(tmp_path) -> None:
    _write_mamem(tmp_path / "A" / "P1" / "good.mat", [_stream("EEG", 128, 14, 4, t0=0.0)])
    (tmp_path / "A" / "P1" / "broken.mat").write_bytes(b"\x00" * 64)
    assert len(P.load_mamem(tmp_path, W)) == _n_windows(4)


# ------------------------------------------------------------------ WESAD -- #
def test_wesad_windows_each_condition_run_separately(tmp_path) -> None:
    rate, seconds = 700, {1: 10, 4: 5, 2: 6}
    # meditation (4) occurs twice, split by stress (2): two runs, never merged
    labels = np.concatenate([np.full(rate * 10, 1), np.full(rate * 5, 4), np.full(rate * 6, 2), np.full(rate * 5, 4)])
    chest = {k: np.random.default_rng(1).normal(size=(len(labels), 1)) for k in ("ECG", "EDA", "EMG", "Resp", "Temp")}
    (tmp_path / "S2").mkdir()
    with open(tmp_path / "S2" / "S2.pkl", "wb") as fh:
        pickle.dump({"signal": {"chest": chest}, "label": labels, "subject": "S2"}, fh)

    records = P.load_wesad(tmp_path, W)
    by_label = {lab: sum(r.load_label == lab for r in records) for lab in ("low", "medium", "high")}
    assert by_label == {"low": _n_windows(seconds[1]) + 2 * _n_windows(5), "medium": 0, "high": _n_windows(seconds[2])}
    assert {r.signals.shape for r in records} == {(128, 4)}
    assert {r.fatigue for r in records if r.load_label == "high"} == {0.80}  # Table 4: stress (TSST)


# ------------------------------------------------------------------- CLAS -- #
# The released archive: Participants/Part<N>/by_block/<block>_ecg_.csv and
# _gsr_ppg_.csv, and Block_details/Part<N>_Block_Details.csv, which names the
# ECG file "<block>_ecg.csv" -- without the underscore the file on disk has.
DETAILS_HEADER = "Block,Block Type, ECG File,EDA&PPG File,Length(s),EDA Quality,ECG Quality,PPG Quality,\n"


def _write_clas(root, folder: str, block: int, seconds: float) -> None:
    rng = np.random.default_rng(block)
    n = int(256 * seconds)
    d = root / "Participants" / folder / "by_block"
    d.mkdir(parents=True, exist_ok=True)
    t = np.arange(n) / 256
    np.savetxt(d / f"{block}_ecg_.csv", np.column_stack([t, rng.normal(size=n)]),
               delimiter=",", header="Timestamp,ecg2", comments="")
    np.savetxt(d / f"{block}_gsr_ppg_.csv", np.column_stack([t, rng.normal(size=(n, 5))]),
               delimiter=",", header="Timestamp,GSR,PPG,AccX,AccY,AccZ", comments="")
    # a per-stimulus copy of the same block, which must not be counted twice
    np.savetxt(d / f"{block}_ecg_1.csv", np.column_stack([t, rng.normal(size=n)]), delimiter=",")


def _write_details(root, participant: int, blocks) -> None:
    (root / "Block_details").mkdir(exist_ok=True)
    rows = "".join(f"{b},{kind},{b}_ecg.csv,{b}_gsr_ppg_.csv,   {s:.2f},    1.00,    1.00,    1.00,\n"
                   for b, kind, s in blocks)
    (root / "Block_details" / f"Part{participant}_Block_Details.csv").write_text(DETAILS_HEADER + rows)


@pytest.fixture()
def clas_root(tmp_path):
    root = tmp_path / "CLAS"
    blocks = [(1, "Baseline", 10), (2, "Math Test", 6), (8, "IQ Test", 4), (9, "Video clip", 4)]
    for b, _, s in blocks:
        _write_clas(root, "Part1", b, s)
        _write_clas(root, "Sample/Sample", b, s)     # the archive's example copy: no participant number
    _write_clas(root, "Part1_copy", 2, 6)            # a second copy of participant 1's block 2
    _write_clas(root, "Part4", 1, 10)                # participant 4 has no Block_Details
    _write_details(root, 1, blocks)
    (root / "Answers").mkdir()
    (root / "Answers" / "Part1_c_i_answers.csv").write_text("q,correct\n1,1\n2,0\n")
    return root


def test_clas_matches_block_details_by_block_number_and_labels_by_table_4(clas_root) -> None:
    records = P.load_clas(clas_root, W)
    assert sum(r.load_label == "high" for r in records) == _n_windows(6)  # Math, read once
    # Baseline, the Logic task ("IQ Test") and video are all "other blocks": low
    assert sum(r.load_label == "low" for r in records) == _n_windows(10) + _n_windows(4) + _n_windows(4)
    assert {r.signals.shape for r in records} == {(128, 3)}  # ECG, GSR, PPG
    assert {r.subject for r in records} == {"Part1"}


def test_clas_inventory_reports_what_will_and_will_not_be_read(clas_root) -> None:
    inv = P.clas_inventory(clas_root)
    assert inv["participants"] == 2 and inv["blocks"] == 5 and inv["typed_blocks"] == 4
    assert inv["high_load_blocks"] == 1
    assert inv["block_types"] == {"baseline": 1, "iq test": 1, "math test": 1, "video clip": 1}
    assert inv["participants_without_block_details"] == ["4"]
    assert len(inv["folders_without_participant"]) == 1 and inv["duplicate_files"] == 2


# ------------------------------------------------------------ windowing -- #
def test_windows_are_resampled_to_64_hz_with_half_overlap() -> None:
    x = np.sin(np.arange(700 * 10) / 700 * 2 * np.pi)[:, None]  # 10 s of a 1 Hz sine at 700 Hz
    windows = P._windows(P._resample(x, 700, 64), W)
    assert len(windows) == _n_windows(10)
    assert np.allclose(windows[1][:64], windows[0][64:])  # 50% overlap
    assert abs(windows[0][16, 0] - 1.0) < 0.05  # the peak lands at 0.25 s = sample 16


def test_max_windows_keeps_evenly_spaced_windows() -> None:
    x = np.arange(64 * 100, dtype=float)[:, None]
    capped = P._windows(x, P.Windowing(max_windows=5))
    assert len(capped) == 5
    assert capped[0][0, 0] == 0.0 and capped[-1][-1, 0] == x[-1, 0]


def test_load_public_corpora_skips_missing_corpora_unless_strict(tmp_path) -> None:
    (tmp_path / "wesad").mkdir()
    assert P.load_public_corpora(tmp_path, windowing=W) == []
    with pytest.raises(FileNotFoundError):
        P.load_public_corpora(tmp_path, strict=True, windowing=W)

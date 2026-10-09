"""scripts/merge_results.py: recombining evaluate.py runs made in separate sessions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mcga_lm.config import Config  # noqa: E402

PERSONAS = ["P00", "P01", "P02", "P03"]


def _chunk(root: Path, name: str, seeds, results: dict, held_out: dict | None = None, quick: bool = False) -> None:
    d = root / name / "evaluate"
    d.mkdir(parents=True)
    payload = {
        "config": {"simulation": {"seeds": list(seeds)}},
        "quick": quick,
        "results": {k: {"aggregate": {}, "per_persona": v} for k, v in results.items()},
        "held_out": {k: {"aggregate": {}, "per_persona": v} for k, v in (held_out or {}).items()},
    }
    (d / "results.json").write_text(json.dumps(payload))


def _scores(sact: float) -> dict:
    return {p: {"sact": sact + 0.1 * i, "wpm": 20.0 - sact} for i, p in enumerate(PERSONAS)}


@pytest.fixture()
def cfg() -> Config:
    cfg = Config()
    cfg.simulation.seeds = [0, 1, 2]
    return cfg


def test_seed_chunks_recombine_and_each_system_comes_from_its_own_chunk(tmp_path, cfg) -> None:
    import merge_results as M

    _chunk(tmp_path, "eval_MCGA-LM_seeds0-1", [0, 1],
           {"MCGA-LM": _scores(3.0), "TouchChat": _scores(12.0)}, held_out={"MCGA-LM": _scores(4.0)})
    _chunk(tmp_path, "eval_MCGA-LM_seeds2", [2],
           {"MCGA-LM": _scores(6.0), "TouchChat": _scores(15.0)}, held_out={"MCGA-LM": _scores(7.0)})
    # this chunk's copy of the baseline differs, and must be ignored
    _chunk(tmp_path, "eval_RAG-LLM", [0, 1, 2], {"RAG-LLM": _scores(5.0), "TouchChat": _scores(99.0)})

    m = M.merge(tmp_path, cfg)
    assert set(m["results"]) == {"MCGA-LM", "RAG-LLM", "TouchChat"}
    # seed-weighted: (2 * 3.0 + 1 * 6.0) / 3 = 4.0 for P00
    assert m["results"]["MCGA-LM"].per_persona["P00"]["sact"] == pytest.approx(4.0)
    assert m["results"]["TouchChat"].per_persona["P00"]["sact"] == pytest.approx(13.0)
    assert m["held_out"]["MCGA-LM"].per_persona["P00"]["sact"] == pytest.approx(5.0)
    assert {c["comparison"] for c in m["statistics"]["sact_effect_sizes_vs_MCGA-LM"]} == {
        "MCGA-LM vs RAG-LLM", "MCGA-LM vs TouchChat",
    }


def test_overlapping_or_missing_seeds_are_refused(tmp_path, cfg) -> None:
    import merge_results as M

    _chunk(tmp_path, "eval_MCGA-LM_seeds0-1", [0, 1], {"MCGA-LM": _scores(3.0)})
    _chunk(tmp_path, "eval_MCGA-LM_seeds1-2", [1, 2], {"MCGA-LM": _scores(3.0)})
    with pytest.raises(SystemExit, match="overlapping seeds"):
        M.merge(tmp_path, cfg)

    other = tmp_path / "other"
    _chunk(other, "eval_MCGA-LM_seeds0-1", [0, 1], {"MCGA-LM": _scores(3.0)})
    with pytest.raises(SystemExit, match="do not match"):
        M.merge(other, cfg)


def test_a_quick_wiring_check_is_never_merged(tmp_path, cfg) -> None:
    import merge_results as M

    _chunk(tmp_path, "eval_MCGA-LM", [0, 1, 2], {"MCGA-LM": _scores(3.0)}, quick=True)
    with pytest.raises(SystemExit, match="wiring check"):
        M.merge(tmp_path, cfg)

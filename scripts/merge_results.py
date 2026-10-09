#!/usr/bin/env python3
"""Merge evaluate.py runs made in separate sessions into one result.

    python scripts/merge_results.py --chunks runs/evaluate_chunks \\
        --config configs/llm_hf.yaml --out runs/evaluate_merged

A long main comparison can be run one system per session, and a system can be
split further by seed. Each run writes ``<chunks>/<chunk>/evaluate/results.json``.
A chunk named ``eval_<system>`` or ``eval_<system>_seeds...`` contributes that
system only. Every evaluate.py run also scores the four non-LLM baselines, so
those are taken from the MCGA-LM chunks and ignored elsewhere. A chunk with any
other name contributes every system it holds.

A system split by seed is recombined per persona as the seed-count-weighted
mean, which is the mean over all its seeds. Seeds must not overlap and must
together equal the config's seeds, and every system must cover the same
personas. The statistics are computed by evaluate.py's own ``_statistics``, so
they are what one run over every system would have reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from _common import logger, provenance_note
from evaluate import GENERATIVE, _markdown_tables, _statistics

Chunk = Tuple[Tuple[int, ...], Dict[str, Dict[str, float]], str]


def chunk_owner(name: str) -> Optional[str]:
    """``eval_MCGA-LM_seeds0-1`` -> ``MCGA-LM``; ``None`` for a chunk of any other name."""
    if not name.startswith("eval_"):
        return None
    return name[len("eval_"):].split("_seeds")[0]


def takes(owner: Optional[str], system: str) -> bool:
    """Whether a chunk owned by ``owner`` is the source of ``system``'s scores."""
    return owner is None or system == owner or (owner == "MCGA-LM" and system not in GENERATIVE)


def merge_system(name: str, chunks: List[Chunk], expected: Tuple[int, ...]) -> Dict[str, Dict[str, float]]:
    seeds = [s for chunk_seeds, _, _ in chunks for s in chunk_seeds]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"{name}: overlapping seeds across {[c[2] for c in chunks]}")
    if sorted(seeds) != sorted(expected):
        raise SystemExit(f"{name}: seeds {sorted(seeds)} do not match the config's {sorted(expected)}")
    personas = {tuple(sorted(pp)) for _, pp, _ in chunks}
    if len(personas) != 1:
        raise SystemExit(f"{name}: chunks cover different personas")
    merged: Dict[str, Dict[str, float]] = {}
    for pid in next(iter(personas)):
        metrics = set().union(*(pp[pid] for _, pp, _ in chunks))
        merged[pid] = {
            m: sum(len(s) * pp[pid][m] for s, pp, _ in chunks if m in pp[pid])
            / sum(len(s) for s, pp, _ in chunks if m in pp[pid])
            for m in metrics
        }
    return merged


def merge(chunks_dir: Path, cfg) -> Dict:
    """Collect, check and recombine every chunk under ``chunks_dir``."""
    from mcga_lm.eval import runner as R
    from mcga_lm.eval import stats as S

    files = sorted(chunks_dir.rglob("results.json"))
    if not files:
        raise SystemExit(f"no results.json under {chunks_dir}")
    found: Dict[str, Dict[str, List[Chunk]]] = {"results": {}, "held_out": {}}
    for f in files:
        d = json.loads(f.read_text())
        if d.get("quick") or d.get("smoke_test_not_a_result"):
            raise SystemExit(f"{f} is a --quick wiring check, not a result")
        owner = chunk_owner(f.relative_to(chunks_dir).parts[0])
        seeds = tuple(d["config"]["simulation"]["seeds"])
        for split in found:
            for name, r in d.get(split, {}).items():
                if not takes(owner, name):
                    continue
                if "per_persona" not in r:
                    logger.warning("%s: %s %s has no per-persona scores; leaving it out", f, split, name)
                    continue
                found[split].setdefault(name, []).append((seeds, r["per_persona"], str(f)))

    expected = tuple(cfg.simulation.seeds)
    split_name = {"results": R.IN_DISTRIBUTION, "held_out": R.HELD_OUT}
    merged = {
        split: {
            name: R.SystemResult(name=name, split=split_name[split], per_persona=merge_system(name, chunks, expected)).finalise()
            for name, chunks in systems.items()
        }
        for split, systems in found.items()
    }
    personas = {tuple(sorted(r.per_persona)) for r in merged["results"].values()}
    if len(personas) > 1:
        raise SystemExit("systems were evaluated on different persona sets")
    return {
        "results": merged["results"],
        "held_out": merged["held_out"],
        "sources": {s: {n: [c[2] for c in cs] for n, cs in systems.items()} for s, systems in found.items()},
        "statistics": _statistics(merged["results"], cfg, S),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", required=True, help="directory searched for results.json files")
    ap.add_argument("--config", required=True, help="the config every chunk was run with")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mcga_lm.config import Config
    from mcga_lm.utils import save_json

    cfg = Config.load(args.config)
    m = merge(Path(args.chunks), cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "provenance": provenance_note(cfg),
            "merged_from": m["sources"],
            "seeds": list(cfg.simulation.seeds),
            "results": {k: {"aggregate": v.aggregate, "per_persona": v.per_persona} for k, v in m["results"].items()},
            "held_out": {k: {"aggregate": v.aggregate, "per_persona": v.per_persona} for k, v in m["held_out"].items()},
            "statistics": m["statistics"],
        },
        out / "results.json",
    )
    table = _markdown_tables(m["results"], m["held_out"], m["statistics"], provenance=provenance_note(cfg))
    (out / "results.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"merged {len(m['results'])} systems ({len(m['held_out'])} with a held-out split) -> {out}")


if __name__ == "__main__":
    main()

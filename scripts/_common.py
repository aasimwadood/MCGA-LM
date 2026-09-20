"""Shared CLI plumbing for the scripts in this directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Allow running the scripts straight from a clone without installing.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mcga_lm.config import Config, merge_overrides  # noqa: E402
from mcga_lm.seed import set_seed  # noqa: E402
from mcga_lm.utils import get_logger  # noqa: E402

logger = get_logger("mcga_lm.scripts")


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config")
    parser.add_argument("--seed", type=int, default=None, help="override the global seed")
    parser.add_argument("--out", type=str, default=None, help="output directory")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config overrides, e.g. --set training.epochs=5 llm.backend=hf",
    )
    return parser


def load_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config) if args.config else Config()
    if args.overrides:
        cfg = merge_overrides(cfg, args.overrides)
    if args.seed is not None:
        cfg.seed = args.seed
    if getattr(args, "out", None):
        cfg.output_dir = args.out
    set_seed(cfg.seed)
    return cfg


def out_dir(cfg: Config, subdir: str) -> Path:
    path = Path(cfg.output_dir) / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_personas(cfg: Config):
    from mcga_lm.data.personas import build_persona_suite

    return build_persona_suite(cfg.simulation, cfg.graph, seed=cfg.seed)


def build_backend(cfg: Config):
    """Build the language backend with d_w tied to ``inputs.ling_dim`` (Sec. 3.2)."""
    from mcga_lm.llm import build_backend as _build

    return _build(cfg.llm, device="cpu", embedding_dim=cfg.inputs.ling_dim)


def corpus_sentences(cfg: Config, personas) -> List[str]:
    """Training-split sentences of the synthetic AAC-Intent-Corpus (Sec. 4.1)."""
    from mcga_lm.data.corpus import build_corpus

    pool = {"Person": [], "Object": [], "Activity": [], "AbstractState": []}
    for persona in personas:
        for node_type in pool:
            pool[node_type].extend(persona.nodes_of_type(node_type))
    pool = {k: sorted(set(v)) or ["thing"] for k, v in pool.items()}
    corpus = build_corpus(pool, seed=cfg.seed)
    return [ex.text for ex in corpus.train]


def banner(title: str) -> None:
    line = "=" * max(len(title), 60)
    print(f"\n{line}\n{title}\n{line}")


def provenance_note() -> str:
    return (
        "Produced by the MCGA-LM reference implementation on synthetic personas. "
        "Unless a model backend and pre-training corpora were configured, these "
        "numbers come from the weight-free template backend and are not the "
        "paper's reported figures. See docs/STATUS.md."
    )

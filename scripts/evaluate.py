#!/usr/bin/env python3
"""The main experiment: Tables 6-7 and Fig. 2 (paper Sec. 4.3, 4.4, 5.1).

    python scripts/evaluate.py --out runs/ --pretrained runs/pretrain/encoder_pretrained.pt

Runs MCGA-LM against every baseline of Sec. 4.4 over the persona suite, on the
in-distribution and held-out splits, and runs the Sec. 4.8 statistical analysis
plan: repeated-measures ANOVA on SACT with Holm-corrected paired tests, ART and
Wilcoxon tests for hallucination rate and IHR@3, KL and bootstrapped ECE
differences for calibration, Table 7's significance markers, and Cohen's d_s.
``--descriptive-only`` reports means, SDs and d_s without any test (not in the
paper).

``--instruction-data`` runs Sec. 3.7's instruction-tuning step before any system
is trained; each system then fine-tunes per-persona LoRA adapters (LLaMA only).

``--quick`` is a WIRING CHECK, NOT A RESULT: it shrinks the encoder eightfold and
trains the retrieval head for one epoch, which drives IHR, SACT, WPM and FAR to
chance. Its output goes to runs/evaluate/quick/ and is stamped accordingly.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from _common import (
    banner,
    base_parser,
    build_backend,
    build_personas,
    corpus_sentences,
    instruction_tune_from_args,
    load_config,
    logger,
    out_dir,
    provenance_note,
)

GENERATIVE = ["MCGA-LM", "RAG-LLM", "M-LLM", "LLM-Only"]
KEYSTROKE = ["TouchChat", "Static-WP-bigram", "Adaptive-grid"]
# Sec. 5.1: the SACT ANOVA covers the five utterance-level systems of the
# original design; the Non-LLM Intent baseline is compared descriptively only.
ANOVA_SYSTEMS = ["TouchChat", "LLM-Only", "M-LLM", "RAG-LLM", "MCGA-LM"]


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="representation epochs per system")
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="small model, few seeds -- for smoke tests")
    parser.add_argument(
        "--per-persona",
        action="store_true",
        help="fit a separate encoder/GAT per persona (not in the paper; tau and LoRA adapters are "
        "per persona either way). Slower, but the shared-model path under-fits retrieval badly "
        "(see the Reproduction status section of README.md)",
    )
    parser.add_argument(
        "--descriptive-only",
        action="store_true",
        help="skip the Sec. 4.8 tests and report means, SDs and Cohen's d_s only (not in the paper)",
    )
    parser.add_argument(
        "--instruction-data",
        type=str,
        default=None,
        help="JSON-lines {prompt, response} file for Sec. 3.7's instruction-tuning step",
    )
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--skip-heldout", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args)
    if args.quick:
        cfg = _quick(cfg)

    import torch

    from mcga_lm.baselines.intent_classifier import IntentClassifierBaseline
    from mcga_lm.baselines.variants import get_variant
    from mcga_lm.eval import runner as R
    from mcga_lm.eval import stats as S
    from mcga_lm.training.personalise import train
    from mcga_lm.utils import resolve_device, save_json

    device = resolve_device(cfg.training.device)
    personas = build_personas(cfg)
    backend = build_backend(cfg)
    seeds = list(cfg.simulation.seeds)
    # Quick runs write somewhere else entirely, so a smoke test can never
    # overwrite -- or be mistaken for -- the canonical results.
    target = out_dir(cfg, "evaluate/quick" if args.quick else "evaluate")
    systems = args.systems or GENERATIVE

    banner("Evaluating systems on the synthetic persona suite (paper Sec. 5.1)")
    print(f"personas={len(personas)} seeds={seeds} device={device} backend={cfg.llm.backend}")
    print("model fitting: " + ("one per persona" if args.per_persona else "one shared across personas"))
    instruction = instruction_tune_from_args(backend, cfg, args.instruction_data)
    print(f"Latin-square counterbalancing order (Sec. 3.8): {R.latin_square(len(systems))[0]} ...")

    results: Dict[str, R.SystemResult] = {}
    heldout: Dict[str, R.SystemResult] = {}

    for name in systems:
        variant = get_variant(name)
        vcfg = variant.apply(cfg)
        logger.info("training %s", name)
        model, report = train(
            vcfg,
            personas,
            backend,
            epochs=args.epochs,
            head_epochs=args.head_epochs,
            pretrained=args.pretrained,
            per_persona=args.per_persona,
        )
        logger.info("evaluating %s", name)
        res, _ = R.run_generative_system(
            name, vcfg, model, backend, personas, seeds,
            taus=report.tau_by_persona, floors=report.floor_by_persona, device=device, split=R.IN_DISTRIBUTION,
        )
        results[name] = res
        if not args.skip_heldout:
            ho, _ = R.run_generative_system(
                name, vcfg, model, backend, personas, seeds,
                taus=report.tau_by_persona, floors=report.floor_by_persona, device=device, split=R.HELD_OUT,
            )
            heldout[name] = ho

    banner("Keystroke-level baselines (paper Sec. 4.4 items 1, 5, 6)")
    sentences = corpus_sentences(cfg, personas)
    for name in KEYSTROKE:
        results[name] = R.run_keystroke_baseline(name, cfg, personas, seeds, sentences)

    banner("Non-LLM intent classifier (paper Sec. 4.4 item 7)")
    classifier = IntentClassifierBaseline(cfg).to(device)
    _train_classifier(classifier, cfg, personas, device, epochs=max(1, args.head_epochs))
    results["Non-LLM-Intent"] = R.run_intent_classifier(cfg, classifier, personas, seeds, device)

    banner("Effect sizes only (--descriptive-only)" if args.descriptive_only else "Statistical analysis (paper Sec. 4.8)")
    stats = _statistics(results, cfg, S, descriptive_only=args.descriptive_only)

    payload = {
        "config": cfg.to_dict(),
        "quick": bool(args.quick),
        "smoke_test_not_a_result": bool(args.quick),
        "per_persona": args.per_persona,
        "provenance": provenance_note(),
        "results": {k: {"aggregate": v.aggregate, "per_persona": v.per_persona} for k, v in results.items()},
        "held_out": {k: {"aggregate": v.aggregate} for k, v in heldout.items()},
        "statistics": stats,
        "instruction_tuning": instruction,
        # Sec. 3.6 safety: personas whose FAR the gate could not bound at all.
        "calibration": {
            "summary": report.summary(),
            "tau_by_persona": report.tau_by_persona,
            "floor_by_persona": report.floor_by_persona,
            "far_budget_missed": sorted(report.far_budget_missed),
        },
    }
    save_json(payload, target / "results.json")
    table = _markdown_tables(results, heldout, stats, report)
    if args.quick:
        table = (
            "> ⚠️ **SMOKE TEST, NOT A RESULT.** Reduced encoder, one training epoch,\n"
            "> 2 seeds, 20 turns per session. Retrieval-dependent metrics (IHR, SACT,\n"
            "> WPM, FAR) collapse towards chance by construction. Do not cite, compare\n"
            "> against the paper, or commit these numbers.\n\n"
        ) + table
    (target / "results.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"\nwritten to {target}")


# --------------------------------------------------------------------------- #
def _quick(cfg):
    """Smoke-test configuration: same code path, much smaller everything.

    This is a wiring check, NOT a result. It shrinks the encoder ~8x and trains
    the retrieval head for a single epoch, so every retrieval-dependent metric
    (IHR, SACT, WPM, FAR) collapses towards chance. Output is written to a
    ``quick/`` subdirectory and stamped ``"quick": true`` so it can never be
    mistaken for, or committed as, a measurement.
    """
    cfg.perceiver.num_latents = 32
    cfg.perceiver.latent_dim = 128
    cfg.perceiver.depth = 1
    cfg.perceiver.self_attn_per_block = 1
    cfg.inputs.phys_window = 32
    cfg.tft.window = 8
    cfg.tft.state_dim = 32
    cfg.graph.gat_hidden = 32
    cfg.simulation.turns_per_session = 20
    cfg.simulation.seeds = [0, 1]
    cfg.training.representation_epochs = 1
    cfg.safety.mc_passes = 8
    # A wiring check has to run anywhere, so it uses the CPU backend.
    cfg.llm.backend = "template"
    cfg.inputs.ling_dim = 256
    cfg.safety.mc_dropout_site = "scoring_head"
    return cfg


def _train_classifier(classifier, cfg, personas, device, epochs: int) -> None:
    """Train baseline 7 on the same training sessions as the main system."""
    import torch
    import torch.nn.functional as F

    from mcga_lm.data.dataset import TurnEncoder
    from mcga_lm.data.taxonomy import FUNCTION_INDEX
    from mcga_lm.training.personalise import TRAIN_SEEDS

    encoder = TurnEncoder(cfg.inputs, reduced_sensor_set=cfg.simulation.reduced_sensor_set)
    optimiser = torch.optim.AdamW(classifier.parameters(), lr=cfg.training.lr * 5)
    for _ in range(epochs):
        for persona in personas:
            for seed in TRAIN_SEEDS:
                turns = persona.simulate_session(seed=seed)
                session = encoder.encode_session(persona, turns, persona.graph, seed=seed).to(device)
                targets = torch.tensor(
                    [FUNCTION_INDEX[t.intent.function] for t in turns], device=device
                )
                logits = classifier(session.batch)
                loss = F.cross_entropy(logits, targets)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()


def _statistics(results, cfg, S, descriptive_only: bool = False) -> Dict:
    """Sec. 4.8's analysis plan, and the SACT effect sizes of Sec. 5.1.

    Cohen's d_s of each system against MCGA-LM, from the per-condition means and
    SDs -- the pooled-SD form that can be recomputed from Table 6 -- signed so a
    positive value favours MCGA-LM. Unless ``descriptive_only``, the tests of
    :func:`mcga_lm.eval.stats.analysis_plan` follow: the SACT ANOVA over
    ``ANOVA_SYSTEMS``, and every other test over the generative systems, which
    are the rows of Table 7.
    """
    names = [n for n in results if np.isfinite(results[n].column("sact")).all()]
    if "MCGA-LM" not in names or len(names) < 2:
        return {"note": "insufficient systems with SACT for effect sizes"}
    matrix = np.stack([results[n].column("sact") for n in names], axis=1)
    ref = names.index("MCGA-LM")
    effect_sizes = []
    for j, name in enumerate(names):
        if j == ref:
            continue
        d_s = S.cohens_ds(matrix[:, j], matrix[:, ref])
        effect_sizes.append(
            {"comparison": f"MCGA-LM vs {name}", "d_s": d_s, "large_effect": bool(abs(d_s) > 0.8)}
        )
    out: Dict = {"sact_systems": names, "sact_effect_sizes_vs_MCGA-LM": effect_sizes}
    if not descriptive_only:
        out["analysis_plan"] = S.analysis_plan(
            sact={n: results[n].column("sact") for n in ANOVA_SYSTEMS if n in results},
            metrics={
                n: {m: results[n].column(m) for m in results[n].aggregate}
                for n in GENERATIVE
                if n in results
            },
            n_iter=cfg.evaluation.bootstrap_iters,
            alpha=cfg.evaluation.holm_alpha,
            seed=cfg.seed,
        )
    return out


def _fmt(agg: Dict[str, Dict[str, float]], key: str, scale: float = 1.0, nd: int = 1) -> str:
    if key not in agg or not np.isfinite(agg[key]["mean"]):
        return "n/r"
    return f"{agg[key]['mean']*scale:.{nd}f} ± {agg[key]['sd']*scale:.{nd}f}"


def _fmt_sact(agg: Dict[str, Dict[str, float]]) -> str:
    """SACT as Table 6 prints it: ``mean ± SD (median M [Q1-Q3])``."""
    if "sact" not in agg or not np.isfinite(agg["sact"]["mean"]):
        return "n/r"
    a = agg["sact"]
    base = f"{a['mean']:.1f} ± {a['sd']:.1f}"
    if np.isfinite(a.get("median", float("nan"))):
        base += f" (median {a['median']:.1f} [{a['q1']:.1f}-{a['q3']:.1f}])"
    return base


def _markdown_tables(results, heldout, stats, report=None) -> str:
    lines: List[str] = []
    lines.append("# MCGA-LM results (synthetic personas)\n")
    lines.append("> " + provenance_note().replace("\n", " ") + "\n")
    lines.append("## Table 6 analogue - primary comparison\n")
    lines.append("| System | WPM ↑ | SACT ↓ | IHR@3 ↑ | Hard halluc. ↓ | Soft halluc. | FAR ↓ | Abstain |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, res in results.items():
        a = res.aggregate
        lines.append(
            f"| {name} | {_fmt(a,'wpm')} | {_fmt_sact(a)} | {_fmt(a,'ihr@3',100,0)} | "
            f"{_fmt(a,'hallucination_hard',100,1)} | {_fmt(a,'hallucination_soft',100,1)} | "
            f"{_fmt(a,'far',100,1)} | {_fmt(a,'abstention',100,1)} |"
        )
    plan = stats.get("analysis_plan", {})
    # Table 7's superscripts: post-hoc paired t-test vs MCGA-LM, Holm-corrected.
    marks = {
        (metric, t["system"]): t["stars"]
        for metric, tests in plan.get("table_7", {}).items()
        for t in tests
        if t["significant"]
    }
    lines.append("\n## Table 7 analogue - retrieval, fluency and calibration\n")
    lines.append("| System | IHR@1 ↑ | IHR@5 ↑ | BLEU-4 ↑ | ROUGE-L ↑ | ECE ↓ | KSPC ↓ |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, res in results.items():
        a = res.aggregate
        cell = {m: marks.get((m, name), "") for m in ("ihr@1", "ihr@5", "bleu4", "rouge_l", "ece")}
        lines.append(
            f"| {name} | {_fmt(a,'ihr@1',100,0)}{cell['ihr@1']} | {_fmt(a,'ihr@5',100,0)}{cell['ihr@5']} | "
            f"{_fmt(a,'bleu4',1,2)}{cell['bleu4']} | {_fmt(a,'rouge_l',1,2)}{cell['rouge_l']} | "
            f"{_fmt(a,'ece',1,3)}{cell['ece']} | {_fmt(a,'kspc',1,2)} |"
        )
    if plan.get("table_7"):
        lines.append(
            "\nSignificance vs. MCGA-LM (post-hoc paired t-test, Holm-corrected): "
            "\\*\\*\\* p<0.001, \\*\\* p<0.01, \\* p<0.05."
        )
    if heldout:
        lines.append("\n## Held-out split (context/interlocutor/topic absent from the graph, Sec. 4.3)\n")
        lines.append("| System | Hard halluc. ↓ | Soft halluc. | IHR@3 ↑ |")
        lines.append("|---|---|---|---|")
        for name, res in heldout.items():
            a = res.aggregate
            lines.append(
                f"| {name} | {_fmt(a,'hallucination_hard',100,1)} | {_fmt(a,'hallucination_soft',100,1)} | "
                f"{_fmt(a,'ihr@3',100,0)} |"
            )
    if "sact_effect_sizes_vs_MCGA-LM" in stats:
        lines.append("\n## Effect sizes on SACT (Sec. 5.1)\n")
        lines.append("| Contrast | Cohen's d_s (positive favours MCGA-LM) |")
        lines.append("|---|---|")
        for c in stats["sact_effect_sizes_vs_MCGA-LM"]:
            lines.append(f"| {c['comparison']} | {c['d_s']:.2f} |")
        lines.append("\nd_s uses the pooled within-condition SD, so it can be recomputed from the table above.")
    if plan:
        lines.extend(_plan_markdown(plan))
    if report is not None:
        missed = sorted(getattr(report, "far_budget_missed", ()) or ())
        lines.append("\n## Safety calibration (paper Sec. 3.6)\n")
        if missed:
            lines.append(
                f"⚠️ **The FAR budget was unreachable for {len(missed)} persona(s):** "
                f"{', '.join(missed)}. For these users no gate setting kept false "
                f"acceptance within {getattr(report, 'far_budget', 0.05):.0%}, so the FAR "
                f"reported above is **not bounded by the gate**. This is the signature of a "
                f"scoring head that is confidently wrong rather than uncertain, which no "
                f"variance threshold can detect."
            )
        else:
            lines.append(
                "The FAR budget was met for every persona; each τ was paired with a "
                "calibrated confidence floor."
            )
    return "\n".join(lines) + "\n"


def _anova_line(label: str, a: Dict) -> str:
    return (
        f"{label}: F({a['df1']:.1f}, {a['df2']:.1f}) = {a['F']:.2f}, p = {a['p']:.2e}, "
        f"partial eta^2 = {a['partial_eta_sq']:.3f}"
        + (" (Greenhouse-Geisser corrected)" if a["gg_corrected"] else "")
    )


def _plan_markdown(plan: Dict) -> List[str]:
    """The Sec. 4.8 tests, in the order the paper describes them."""
    lines = ["\n## Statistical analysis (Sec. 4.8)\n"]
    lines.append(
        "> The personas are simulated, so these p-values describe how consistently the "
        "systems differ across the simulator's persona space, not a population of users "
        "(Sec. 4.8). Holm-Bonferroni correction is within each metric family.\n"
    )
    if "sact_anova" in plan:
        a = plan["sact_anova"]
        lines.append(_anova_line(f"Repeated-measures ANOVA on SACT ({', '.join(a['systems'])})", a))
        lines.append("\n| Contrast | t | p (Holm) | |")
        lines.append("|---|---|---|---|")
        for c in plan["sact_pairwise"]:
            lines.append(f"| {c['comparison']} | {c['statistic']:.2f} | {c['p_holm']:.2e} | {c['stars']} |")
    for metric, res in plan.get("nonparametric", {}).items():
        lines.append(f"\n### {metric}: aligned rank transform + Wilcoxon\n")
        lines.append(_anova_line("ANOVA on aligned ranks", res["art_anova"]))
        lines.append("\n| Contrast | W | p (Holm) | |")
        lines.append("|---|---|---|---|")
        for c in res["pairwise_wilcoxon"]:
            lines.append(f"| {c['comparison']} | {c['statistic']:.1f} | {c['p_holm']:.2e} | {c['stars']} |")
    calibration = plan.get("calibration", {})
    if calibration:
        lines.append("\n### Calibration\n")
    if "kl_from_diagonal" in calibration:
        lines.append("| System | KL from the diagonal ↓ |")
        lines.append("|---|---|")
        for name, kl in calibration["kl_from_diagonal"].items():
            lines.append(f"| {name} | {kl['mean']:.3f} ± {kl['sd']:.3f} |")
    if "ece_bootstrap" in calibration:
        lines.append("\n| Contrast | ECE difference [95% bootstrap CI] | p (Holm) | |")
        lines.append("|---|---|---|---|")
        for c in calibration["ece_bootstrap"]:
            lines.append(
                f"| {c['comparison']} | {c['ece_difference']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}] | "
                f"{c['p_holm']:.2e} | {c['stars']} |"
            )
        lines.append("\nPersonas are resampled in pairs, so the repeated-measures pairing is preserved.")
    return lines


if __name__ == "__main__":
    main()

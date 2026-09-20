#!/usr/bin/env python3


from __future__ import annotations

from typing import Dict, List

import numpy as np

from _common import (
    banner,
    base_parser,
    build_backend,
    build_personas,
    corpus_sentences,
    load_config,
    logger,
    out_dir,
    provenance_note,
)

GENERATIVE = ["MCGA-LM", "RAG-LLM", "M-LLM", "LLM-Only"]
KEYSTROKE = ["TouchChat", "Static-WP-bigram", "Adaptive-grid"]


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="representation epochs per system")
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="small model, few seeds -- for smoke tests")
    parser.add_argument(
        "--per-persona",
        action="store_true",
        help="fit one model per persona (Sec. 3.7's on-device configuration). Slower, but it is "
        "what the paper describes, and the shared-model path under-fits retrieval badly "
        "(docs/RESULTS.md section 4)",
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
    target = out_dir(cfg, "evaluate")
    systems = args.systems or GENERATIVE

    banner("Evaluating systems on the synthetic persona suite (paper Sec. 5.1)")
    print(f"personas={len(personas)} seeds={seeds} device={device} backend={cfg.llm.backend}")
    print("model fitting: " + ("one per persona (Sec. 3.7)" if args.per_persona else "one shared across personas"))
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
            taus=report.tau_by_persona, device=device, split=R.IN_DISTRIBUTION,
        )
        results[name] = res
        if not args.skip_heldout:
            ho, _ = R.run_generative_system(
                name, vcfg, model, backend, personas, seeds,
                taus=report.tau_by_persona, device=device, split=R.HELD_OUT,
            )
            heldout[name] = ho

    banner("Keystroke-level baselines (paper Sec. 4.3 items 1, 5, 6)")
    sentences = corpus_sentences(cfg, personas)
    for name in KEYSTROKE:
        results[name] = R.run_keystroke_baseline(name, cfg, personas, seeds, sentences)

    banner("Non-LLM intent classifier (paper Sec. 4.3 item 7)")
    classifier = IntentClassifierBaseline(cfg).to(device)
    _train_classifier(classifier, cfg, personas, device, epochs=max(1, args.head_epochs))
    results["Non-LLM-Intent"] = R.run_intent_classifier(cfg, classifier, personas, seeds, device)

    banner("Statistics (paper Sec. 4.7)")
    stats = _statistics(results, cfg, S)

    payload = {
        "config": cfg.to_dict(),
        "per_persona": args.per_persona,
        "provenance": provenance_note(),
        "results": {k: {"aggregate": v.aggregate, "per_persona": v.per_persona} for k, v in results.items()},
        "held_out": {k: {"aggregate": v.aggregate} for k, v in heldout.items()},
        "statistics": stats,
    }
    save_json(payload, target / "results.json")
    table = _markdown_tables(results, heldout, stats)
    (target / "results.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"\nwritten to {target}")


# --------------------------------------------------------------------------- #
def _quick(cfg):
    """Smoke-test configuration: same code path, much smaller everything."""
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
    cfg.training.lora_epochs = 1
    cfg.safety.mc_passes = 8
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


def _statistics(results, cfg, S) -> Dict:
    """RM-ANOVA on SACT plus Cohen's d_s and Holm-Bonferroni (Sec. 4.7, 5.1)."""
    names = [n for n in results if np.isfinite(results[n].column("sact")).all()]
    if "MCGA-LM" not in names or len(names) < 2:
        return {"note": "insufficient systems with SACT for the ANOVA"}
    matrix = np.stack([results[n].column("sact") for n in names], axis=1)
    anova = S.repeated_measures_anova(matrix)
    ref = names.index("MCGA-LM")
    pairwise = S.paired_t_tests(matrix, names, reference=ref)
    holm = S.holm_bonferroni([c["p"] for c in pairwise], alpha=cfg.evaluation.holm_alpha)
    for c, p_adj, rej in zip(pairwise, holm["p_adjusted"], holm["reject"]):
        c["p_holm"] = p_adj
        c["significant"] = bool(rej)
        c["stars"] = S.stars(p_adj)
    # Sec. 5.1 reports every effect size with a 95% bootstrap percentile CI
    # (10,000 resamples), resampling personas so the pairing is preserved.
    for j, c in zip([i for i in range(len(names)) if i != ref], pairwise):
        ci = S.bootstrap_effect_size(
            matrix[:, ref], matrix[:, j], n_iter=cfg.evaluation.bootstrap_iters, seed=cfg.seed
        )
        c["d_s_ci_low"] = ci["ci_low"]
        c["d_s_ci_high"] = ci["ci_high"]
        c["d_s_formatted"] = S.format_effect_size(ci)
        c["large_effect"] = bool(abs(ci["d_s"]) > 0.8)  # Sec. 5.1: "all large effects (d > 0.8)"

    nonparam: Dict[str, list] = {}
    for metric in ("hallucination_hard", "ihr@3"):
        usable = [n for n in names if np.isfinite(results[n].column(metric)).all()]
        if len(usable) < 2 or "MCGA-LM" not in usable:
            continue
        mat = np.stack([results[n].column(metric) for n in usable], axis=1)
        ranks = S.aligned_rank_transform(mat)
        tests = S.pairwise_wilcoxon(mat, usable, reference=usable.index("MCGA-LM"))
        adj = S.holm_bonferroni([t["p"] for t in tests])
        for t, p_adj in zip(tests, adj["p_adjusted"]):
            t["p_holm"] = p_adj
            t["stars"] = S.stars(p_adj)
        nonparam[metric] = tests
        nonparam[f"{metric}_art_rank_means"] = ranks.mean(axis=0).tolist()

    return {
        "sact_anova": anova.as_dict(),
        "sact_systems": names,
        "sact_pairwise_vs_MCGA-LM": pairwise,
        "nonparametric": nonparam,
        "post_hoc_power_f0.40": S.post_hoc_power(0.40, matrix.shape[0], matrix.shape[1]),
    }


def _fmt(agg: Dict[str, Dict[str, float]], key: str, scale: float = 1.0, nd: int = 1) -> str:
    if key not in agg or not np.isfinite(agg[key]["mean"]):
        return "n/r"
    return f"{agg[key]['mean']*scale:.{nd}f} ± {agg[key]['sd']*scale:.{nd}f}"


def _fmt_sact(agg: Dict[str, Dict[str, float]]) -> str:
    """SACT as Table 7 prints it: ``mean ± SD (median M [Q1-Q3])``."""
    if "sact" not in agg or not np.isfinite(agg["sact"]["mean"]):
        return "n/r"
    a = agg["sact"]
    base = f"{a['mean']:.1f} ± {a['sd']:.1f}"
    if np.isfinite(a.get("median", float("nan"))):
        base += f" (median {a['median']:.1f} [{a['q1']:.1f}-{a['q3']:.1f}])"
    return base


def _markdown_tables(results, heldout, stats) -> str:
    lines: List[str] = []
    lines.append("# MCGA-LM results (synthetic personas)\n")
    lines.append("> " + provenance_note().replace("\n", " ") + "\n")
    lines.append("## Table 7 analogue - primary comparison\n")
    lines.append("| System | WPM ↑ | SACT ↓ | IHR@3 ↑ | Hard halluc. ↓ | Soft halluc. | FAR ↓ | Abstain |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, res in results.items():
        a = res.aggregate
        lines.append(
            f"| {name} | {_fmt(a,'wpm')} | {_fmt_sact(a)} | {_fmt(a,'ihr@3',100,0)} | "
            f"{_fmt(a,'hallucination_hard',100,1)} | {_fmt(a,'hallucination_soft',100,1)} | "
            f"{_fmt(a,'far',100,1)} | {_fmt(a,'abstention',100,1)} |"
        )
    lines.append("\n## Table 8 analogue - retrieval, fluency and calibration\n")
    lines.append("| System | IHR@1 ↑ | IHR@5 ↑ | BLEU-4 ↑ | ROUGE-L ↑ | ECE ↓ | KSPC ↓ |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, res in results.items():
        a = res.aggregate
        lines.append(
            f"| {name} | {_fmt(a,'ihr@1',100,0)} | {_fmt(a,'ihr@5',100,0)} | {_fmt(a,'bleu4',1,2)} | "
            f"{_fmt(a,'rouge_l',1,2)} | {_fmt(a,'ece',1,3)} | {_fmt(a,'kspc',1,2)} |"
        )
    if heldout:
        lines.append("\n## Held-out split (context/interlocutor/topic absent from the graph, Sec. 4.2)\n")
        lines.append("| System | Hard halluc. ↓ | Soft halluc. | IHR@3 ↑ |")
        lines.append("|---|---|---|---|")
        for name, res in heldout.items():
            a = res.aggregate
            lines.append(
                f"| {name} | {_fmt(a,'hallucination_hard',100,1)} | {_fmt(a,'hallucination_soft',100,1)} | "
                f"{_fmt(a,'ihr@3',100,0)} |"
            )
    if "sact_anova" in stats:
        a = stats["sact_anova"]
        lines.append("\n## Statistics (paper Sec. 4.7)\n")
        lines.append(
            f"Repeated-measures ANOVA on SACT: F({a['df1']:.1f}, {a['df2']:.1f}) = {a['F']:.2f}, "
            f"p = {a['p']:.2e}, partial eta^2 = {a['partial_eta_sq']:.3f}"
            + (" (Greenhouse-Geisser corrected)" if a["gg_corrected"] else "")
        )
        lines.append("\n| Contrast | Cohen's d_s [95% bootstrap CI] | p (Holm) | |")
        lines.append("|---|---|---|---|")
        for c in stats["sact_pairwise_vs_MCGA-LM"]:
            d = c.get("d_s_formatted", f"{c['d_s']:.2f}")
            lines.append(f"| {c['comparison']} | {d} | {c['p_holm']:.2e} | {c['stars']} |")
        lines.append(
            "\nIntervals are 95% bootstrap percentile intervals over personas "
            "(Sec. 5.1; 10,000 resamples), resampled pairwise so the "
            "repeated-measures pairing is preserved."
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

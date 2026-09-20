#!/usr/bin/env python3


from __future__ import annotations

from typing import Dict

from _common import banner, base_parser, build_backend, build_personas, load_config, logger, out_dir, provenance_note

ABLATION_ORDER = ["MCGA-LM", "\\PerceiverIO", "\\TFT", "\\GAT", "\\BayesianGate", "\\cross-attention"]


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--per-persona", action="store_true",
                        help="fit one model per persona (Sec. 3.7)")
    parser.add_argument("--fatigued", action="store_true", help="evaluate under induced fatigue (Sec. 4.5)")
    parser.add_argument("--full-factorial", action="store_true",
                        help="run all 2^k cells rather than one-at-a-time (D-10)")
    args = parser.parse_args()
    cfg = load_config(args)
    if args.quick:
        from evaluate import _quick

        cfg = _quick(cfg)

    from mcga_lm.baselines.variants import full_factorial_design, get_variant
    from mcga_lm.eval import runner as R
    from mcga_lm.training.personalise import train
    from mcga_lm.utils import resolve_device, save_json
    from mcga_lm.viz import plot_ablation

    device = resolve_device(cfg.training.device)
    personas = build_personas(cfg)
    backend = build_backend(cfg)
    seeds = list(cfg.simulation.seeds)
    target = out_dir(cfg, "ablations")

    if args.full_factorial:
        banner("Full-factorial ablation, all 2^k cells (paper Sec. 4.4, first sentence)")
        variants = full_factorial_design()
        print(f"{len(variants)} conditions; Fig. 4 reports {len(ABLATION_ORDER)} of them\n")
    else:
        banner("Ablation study, one component at a time (paper Sec. 4.4, Fig. 4)")
        variants = [get_variant(n) for n in ABLATION_ORDER]

    table: Dict[str, dict] = {}
    for variant in variants:
        name = variant.name
        vcfg = variant.apply(cfg)
        logger.info("ablation %s", name)
        model, report = train(
            vcfg, personas, backend, epochs=args.epochs,
            head_epochs=args.head_epochs, pretrained=args.pretrained,
            per_persona=args.per_persona,
        )
        res, _ = R.run_generative_system(
            name, vcfg, model, backend, personas, seeds,
            taus=report.tau_by_persona, floors=report.floor_by_persona, device=device,
            # Sec. 4.4: the TFT ablation is the one that must be probed under
            # induced fatigue, since it "has little effect when rested".
            adapted_fatigue=not args.fatigued,
        )
        table[name] = res.aggregate
        print(
            f"  {name:18s} SACT {res.aggregate['sact']['mean']:.2f} ± {res.aggregate['sact']['sd']:.2f}"
            f"   hard-halluc {res.aggregate['hallucination_hard']['mean']*100:.1f}%"
            f"   IHR@3 {res.aggregate['ihr@3']['mean']*100:.0f}%"
            f"   FAR {res.aggregate['far']['mean']*100:.1f}%"
        )

    save_json({"fatigued": args.fatigued, "full_factorial": args.full_factorial,
               "design": "2^k full factorial" if args.full_factorial else "one component at a time",
               "provenance": provenance_note(), "ablations": table},
              target / ("ablations_factorial.json" if args.full_factorial else "ablations.json"))

    # Fig. 4 plots the one-at-a-time conditions; under --full-factorial those are
    # the single-factor cells, which are a subset of what was just run.
    plotted = [n for n in ABLATION_ORDER if n in table]
    if len(plotted) == len(ABLATION_ORDER):
        plot_ablation(
            [n.replace("\\", "\\\\") for n in plotted],
            [table[n]["sact"]["mean"] for n in plotted],
            [table[n]["sact"]["sd"] for n in plotted],
            [table[n]["hallucination_hard"]["mean"] for n in plotted],
            [table[n]["hallucination_hard"]["sd"] for n in plotted],
            target / "fig4_ablation.png",
        )
    print(f"\nwritten to {target}")


if __name__ == "__main__":
    main()

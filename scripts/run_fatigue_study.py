#!/usr/bin/env python3


from __future__ import annotations

from typing import Dict, List

import numpy as np

from _common import banner, base_parser, build_backend, build_personas, load_config, out_dir, provenance_note


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--per-persona", action="store_true",
                        help="fit one model per persona (Sec. 3.7)")
    args = parser.parse_args()
    cfg = load_config(args)
    if args.quick:
        from evaluate import _quick

        cfg = _quick(cfg)

    from mcga_lm.baselines.variants import get_variant
    from mcga_lm.data.physiology import FatigueModel
    from mcga_lm.eval import runner as R
    from mcga_lm.training.personalise import train
    from mcga_lm.utils import resolve_device, save_json
    from mcga_lm.viz import plot_fatigue_trajectory

    device = resolve_device(cfg.training.device)
    personas = build_personas(cfg)
    backend = build_backend(cfg)
    seeds = list(cfg.simulation.seeds)
    target = out_dir(cfg, "fatigue")

    banner("Fatigue adaptation (paper Sec. 5.4) and sensitivity (Sec. 5.5)")

    systems = {}
    for name in ("MCGA-LM", "RAG-LLM"):
        vcfg = get_variant(name).apply(cfg)
        model, report = train(vcfg, personas, backend, epochs=args.epochs, head_epochs=2,
                              pretrained=args.pretrained, per_persona=args.per_persona)
        systems[name] = (vcfg, model, report.tau_by_persona)

    # --- Sec. 5.4: rested vs fatigued ------------------------------------- #
    rows: List[Dict] = []
    for name, (vcfg, model, taus) in systems.items():
        rested, _ = R.run_generative_system(
            name, vcfg, model, backend, personas, seeds, taus=taus, device=device,
            fatigue_half_life=1e6, fatigue_peak=0.05, adapted_fatigue=False,
        )
        fatigued, _ = R.run_generative_system(
            name, vcfg, model, backend, personas, seeds, taus=taus, device=device,
            adapted_fatigue=(name == "MCGA-LM"),
        )
        retained = fatigued.aggregate["wpm"]["mean"] / rested.aggregate["wpm"]["mean"]
        rows.append({
            "system": name,
            "wpm_rested": rested.aggregate["wpm"]["mean"],
            "wpm_fatigued": fatigued.aggregate["wpm"]["mean"],
            "retained_fraction": retained,
            "sact_rested": rested.aggregate["sact"]["mean"],
            "sact_fatigued": fatigued.aggregate["sact"]["mean"],
        })
        print(f"  {name:10s} WPM {rested.aggregate['wpm']['mean']:.1f} -> "
              f"{fatigued.aggregate['wpm']['mean']:.1f}  ({retained*100:.0f}% retained)")

    # --- Sec. 5.5: sensitivity to the fatigue-model parameters -------------- #
    banner("Sensitivity of the fatigue model (paper Sec. 5.5)")
    grid: List[Dict] = []
    for half_life in (15.0, 30.0, 45.0, 60.0):
        for peak in (0.5, 0.7, 0.9):
            row = {"half_life_min": half_life, "peak": peak}
            for name, (vcfg, model, taus) in systems.items():
                res, _ = R.run_generative_system(
                    name, vcfg, model, backend, personas, seeds, taus=taus, device=device,
                    fatigue_half_life=half_life, fatigue_peak=peak,
                    adapted_fatigue=(name == "MCGA-LM"),
                )
                row[f"{name}_wpm"] = res.aggregate["wpm"]["mean"]
                row[f"{name}_sact"] = res.aggregate["sact"]["mean"]
            grid.append(row)
            print(f"  half-life {half_life:4.0f} min, peak {peak:.1f}: "
                  + "  ".join(f"{n} WPM {row[f'{n}_wpm']:.1f}" for n in systems))

    # --- Fig. 5 ------------------------------------------------------------ #
    model_f = FatigueModel(
        half_life_min=cfg.simulation.fatigue_half_life_min, peak=cfg.simulation.fatigue_peak, noise=0.0
    )
    minutes = np.linspace(0, cfg.simulation.session_minutes, 121)
    plot_fatigue_trajectory(
        minutes, model_f.value(minutes), model_f.adapted_value(minutes), target / "fig5_fatigue.png"
    )
    save_json({"provenance": provenance_note(), "rested_vs_fatigued": rows, "sensitivity_grid": grid},
              target / "fatigue_study.json")
    print(f"\nwritten to {target}")


if __name__ == "__main__":
    main()

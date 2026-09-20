#!/usr/bin/env python3


from __future__ import annotations

from _common import banner, base_parser, build_backend, build_personas, load_config, out_dir, provenance_note
from mcga_lm.training.personalise import train
from mcga_lm.utils import save_json


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--pretrained", type=str, default=None, help="encoder checkpoint from stage 1-2")
    parser.add_argument("--epochs", type=int, default=None, help="representation-training epochs")
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--variant", type=str, default="MCGA-LM", help="see mcga_lm.baselines.variants")
    parser.add_argument(
        "--per-persona",
        action="store_true",
        help="fit one model per persona -- the on-device configuration of Sec. 3.7 "
        "(cost is linear in the number of personas)",
    )
    parser.add_argument(
        "--tau-rule",
        choices=["largest", "smallest"],
        default="largest",
        help="threshold-selection rule; see D-01 in src/mcga_lm/safety/gate.py",
    )
    args = parser.parse_args()
    cfg = load_config(args)

    from mcga_lm.baselines.variants import get_variant

    variant = get_variant(args.variant)
    cfg = variant.apply(cfg)

    banner(f"Stage 3: training variant '{variant.name}' (paper Sec. 3.7)")
    personas = build_personas(cfg)
    backend = build_backend(cfg)
    model, report = train(
        cfg,
        personas,
        backend,
        epochs=args.epochs,
        head_epochs=args.head_epochs,
        pretrained=args.pretrained,
        out_dir=str(out_dir(cfg, f"train/{variant.name.replace(chr(92), '')}")),
        per_persona=args.per_persona,
    )
    summary = report.summary()
    save_json(
        {"variant": variant.name, "summary": summary, "tau": report.tau_by_persona,
         "history": report.history, "head_history": report.head_history},
        out_dir(cfg, f"train/{variant.name.replace(chr(92), '')}") / "training_report.json",
    )
    if args.per_persona:
        first = next(iter(model.values()))
        print(f"models          : {len(model)} (one per persona, Sec. 3.7)")
        print(f"parameters each : {first.n_parameters():,}")
    else:
        print(f"models          : 1 (shared across all personas)")
        print(f"parameters      : {model.n_parameters():,}")
    print(f"final loss      : {summary['final_loss']:.4f}")
    print(f"scoring-head BCE: {summary['final_head_loss']:.4f}")
    print(
        f"calibrated tau  : mean {summary['tau_mean']:.3f}, SD {summary['tau_sd']:.3f}, "
        f"range [{summary['tau_min']:.3f}, {summary['tau_max']:.3f}]"
    )
    print(f"  (paper Sec. 3.6 reports mean 0.13, SD 0.04, range [0.07, 0.21] for its own personas)")
    print("\n" + provenance_note())


if __name__ == "__main__":
    main()

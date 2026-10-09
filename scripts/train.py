#!/usr/bin/env python3
"""Stage 3: train MCGA-LM and calibrate tau per persona (paper Sec. 3.7, 3.6).

    python scripts/train.py --out runs/ --pretrained runs/pretrain/encoder_pretrained.pt

Fits the Perceiver/TFT/GAT against Eq. (9), trains the intent-scoring head,
calibrates the uncertainty threshold for each persona against the 5%
false-acceptance budget, then fine-tunes each persona's LoRA adapter on the
utterances it accepted (Sec. 3.5; LLaMA backend only).

``--instruction-data`` instruction-tunes the shared adapter first (Sec. 3.7
stage 3). The paper's "generic AAC prompt dataset" is not published, so the
step is skipped without it.

``--per-persona`` fits a separate encoder/GAT per persona, which the paper does
not describe; tau and the LoRA adapters are per persona either way.
"""

from __future__ import annotations

from _common import (
    banner,
    base_parser,
    build_backend,
    build_personas,
    instruction_tune_from_args,
    load_config,
    out_dir,
    provenance_note,
)
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
        help="fit a separate encoder/GAT per persona (not in the paper; cost is linear in personas)",
    )
    parser.add_argument(
        "--tau-rule",
        choices=["smallest", "largest"],
        default=None,
        help="threshold-selection rule; default 'smallest' as in Sec. 3.6",
    )
    parser.add_argument(
        "--instruction-data",
        type=str,
        default=None,
        help="JSON-lines {prompt, response} file for Sec. 3.7's instruction-tuning step",
    )
    args = parser.parse_args()
    cfg = load_config(args)
    if args.tau_rule:
        cfg.safety.tau_rule = args.tau_rule

    from mcga_lm.baselines.variants import get_variant

    variant = get_variant(args.variant)
    cfg = variant.apply(cfg)

    banner(f"Stage 3: training variant '{variant.name}' (paper Sec. 3.7)")
    personas = build_personas(cfg)
    backend = build_backend(cfg)
    instruction = instruction_tune_from_args(backend, cfg, args.instruction_data)
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
         "history": report.history, "head_history": report.head_history,
         "instruction_tuning": instruction, "lora": report.lora_by_persona},
        out_dir(cfg, f"train/{variant.name.replace(chr(92), '')}") / "training_report.json",
    )
    if args.per_persona:
        first = next(iter(model.values()))
        print(f"models          : {len(model)} (one per persona)")
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
    trained = [r for r in report.lora_by_persona.values() if "history" in r]
    if trained:
        print(f"LoRA adapters   : {len(trained)} personas fine-tuned, {cfg.training.lora_epochs} epochs each")
    else:
        reason = next(iter(report.lora_by_persona.values()), {}).get("skipped", "not run")
        print(f"LoRA adapters   : none trained ({reason})")
    print("\n" + provenance_note(cfg))


if __name__ == "__main__":
    main()

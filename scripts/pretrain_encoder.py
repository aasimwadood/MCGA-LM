#!/usr/bin/env python3
"""Stages 1-2: pre-train the Perceiver IO encoder and the TFT (paper Sec. 3.7, 4.9).

    python scripts/pretrain_encoder.py --out runs/

Stage 1 trains Perceiver IO and the TFT jointly; stage 2 trains the TFT alone on
the same data. Fits on MAMEM/CLAS/WESAD if they are present under data/raw/, and
on clearly-labelled synthetic physiology otherwise. Sec. 4.9 specifies 100
epochs, batch 128, AdamW at 1e-4 with cosine decay.

Run this before train.py or evaluate.py, and with the same config you intend to
evaluate with -- a checkpoint from one encoder size will not load into another.
"""

from __future__ import annotations

from _common import banner, base_parser, build_backend, load_config, out_dir, provenance_note
from mcga_lm.training.pretrain import pretrain
from mcga_lm.utils import save_json


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=None, help="epochs for each of the two stages")
    parser.add_argument(
        "--require-public-data",
        action="store_true",
        help="fail instead of falling back to synthetic physiology",
    )
    parser.add_argument(
        "--max-windows-per-recording",
        type=int,
        default=None,
        help="keep at most this many evenly spaced 2-s windows per recording (a compute "
        "budget, not in the paper; default keeps all)",
    )
    args = parser.parse_args()
    cfg = load_config(args)

    banner("Stage 1-2: encoder pre-training (paper Sec. 3.7)")
    target = out_dir(cfg, "pretrain")
    report = pretrain(
        cfg,
        data_root=args.data_root,
        epochs=args.epochs,
        out_dir=str(target),
        allow_synthetic=not args.require_public_data,
        backend_factory=lambda: build_backend(cfg),
        max_windows_per_recording=args.max_windows_per_recording,
    )
    save_json(report, target / "pretrain_report.json")
    print(f"pre-training source : {report['source']}")
    print(f"windows             : {report['n_records']}  {report.get('records_by_source', {})}")
    print(f"final loss          : {report['final_loss']:.5f}")
    print(f"stage 1 / 2 epochs  : {report['epochs']} / {report['tft_epochs']}")
    print(f"fatigue MAE         : {report['final_fatigue_mae']:.4f}  (index scale 0-1, after stage 2)")
    print(f"reconstruction MSE  : {report['final_recon_mse']:.4f}  (z-scored signals)")
    print(f"contrastive term    : {'active' if report['contrastive_active'] else 'inactive (no transcripts)'}")
    print(f"checkpoint          : {report.get('checkpoint')}")
    if report["source"] == "synthetic":
        print("\nNOTE: pre-trained on SYNTHETIC physiology. The fatigue head carries no")
        print("      claim about real HRV/EDA/EEG. " + provenance_note(cfg))


if __name__ == "__main__":
    main()

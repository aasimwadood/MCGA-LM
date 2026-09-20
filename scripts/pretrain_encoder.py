#!/usr/bin/env python3

from __future__ import annotations

from _common import banner, base_parser, load_config, out_dir, provenance_note
from mcga_lm.training.pretrain import pretrain
from mcga_lm.utils import save_json


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--require-public-data",
        action="store_true",
        help="fail instead of falling back to synthetic physiology",
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
    )
    save_json(report, target / "pretrain_report.json")
    print(f"pre-training source : {report['source']}")
    print(f"records             : {report['n_records']}")
    print(f"final loss          : {report['final_loss']:.5f}")
    print(f"fatigue MAE         : {report['final_fatigue_mae']:.4f}  (index scale 0-1)")
    print(f"reconstruction MSE  : {report['final_recon_mse']:.4f}  (z-scored signals)")
    print(f"checkpoint          : {report.get('checkpoint')}")
    if report["source"] == "synthetic":
        print("\nNOTE: pre-trained on SYNTHETIC physiology. The fatigue head carries no")
        print("      claim about real HRV/EDA/EEG. " + provenance_note())


if __name__ == "__main__":
    main()

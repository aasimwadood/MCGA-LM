# runs/pretrain/

Empty by design. No trained weights are committed to this repository.

A checkpoint that was here previously was produced by `--quick` (a 32x128
encoder, depth 1, 10 epochs on synthetic physiology) and could not be loaded by
the default 256x512 configuration, so it has been removed rather than left as a
trap.

Generate one before training:

```bash
python scripts/pretrain_encoder.py --out runs/
```

`pretrain_report.json` from the earlier run is kept for reference. Its
`source: synthetic` field records that MAMEM/CLAS/WESAD were not present, so the
fatigue head it fitted carries no physiological grounding.

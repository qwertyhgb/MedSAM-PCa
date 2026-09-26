# MedSAM-PCa

2D automatic csPCa lesion segmentation on PI-CAI.

- **Task**: axial slice-level segmentation of clinically significant prostate cancer (csPCa) lesions on prostate MRI.
- **Data**: PI-CAI (1500 cases / 1476 patients); this project currently uses the **T2W** sequence only, replicated to 3 channels.
- **Method**: **frozen MedSAM ViT-B image encoder** + task-specific lightweight decoders (E0–E3, 0.26 K – 1.52 M trainable parameters). The encoder is never trained; only the decoder is optimized.
- **Split**: `StratifiedGroupKFold` by patient, seed 42. Phase 1 uses fold0 as validation (300 cases / 6614 slices) and fold1–4 as training (1200 cases / 27090 slices).

## Repository layout

```
configs/            experiment configs (e0_linear, e1_simple_pyramid, e2_unetr, e3_multilevel_fpn)
src/
  datasets/         PI-CAI 2D slice dataset, mask alignment and preprocessing
  losses/           Dice + BCE loss
  metrics/          slice-level metrics, multi-threshold scan, lesion-level metrics
  models/           MedSAM encoder wrapper, decoders, segmentation heads
scripts/            data audit / manifest / split / mask cache building, benchmarks, evaluation
tests/              unit tests (forward-only, no training)
train.py            training entry point (must be launched by the project owner)
evaluate.py         single-checkpoint evaluation entry point
docs/experiments/   experiment protocol and per-experiment reports
```

## Status and results

Phase 1 (four decoder baselines, 40 epochs each) is complete; Phase 2A adds evaluation audit,
threshold analysis and volume/lesion-level evaluation. See:

- [`docs/experiments/protocol.md`](docs/experiments/protocol.md) — data, training and evaluation protocol
- [`docs/experiments/README.md`](docs/experiments/README.md) — cross-experiment comparison
- [`docs/experiments/phase2a_evaluation.md`](docs/experiments/phase2a_evaluation.md) — Phase 2A evaluation report

Performance is reported as **positive-slice Dice on the fold0 validation split** and is still far
from clinically usable: the Phase 1 baselines reach 0.33–0.48 positive-slice Dice with very low
precision (0.06–0.15) and high false-positive rates. Numbers, caveats and failure modes are
documented in full in the reports above; do not quote them out of context.

## Constraints

See [`PROJECT_CONSTRAINTS.md`](PROJECT_CONSTRAINTS.md). In short: original data and produced
checkpoints are read-only, encoder stays frozen, metrics require explicit input types, and long
running operations must show progress bars.

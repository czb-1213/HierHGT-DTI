# DO-GMA for HierHGT-DTI Benchmark

This directory contains a fixed-split adaptation of DO-GMA for the HierHGT-DTI
benchmark protocol.

Unlike the official DO-GMA entrypoint, the adapted runner reads the existing
`train/val/test.csv` files under `data/<dataset>/<split>/`, builds a shared
cache once per dataset/split, and reuses that cache across seeds so graph
construction does not dominate runtime.

## Usage

Prepare cache once per dataset/split:

```bash
python baselines/DO-GMA/prepare_adapted.py --dataset BioSnap --split random
```

Train and evaluate:

```bash
python baselines/DO-GMA/run_adapted.py --dataset BioSnap --split random --seed 42
```

The adapted defaults are tuned for the main benchmark workflow:
`batch_size=64`, `epochs=60`, `patience=8`.

Outputs are written to:

- `baselines/DO-GMA/data_adapted/<dataset>_<split>/`
- `baselines/DO-GMA/output_adapted/<dataset>_<split>/seed<seed>/`

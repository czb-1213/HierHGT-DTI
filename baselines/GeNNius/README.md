# GeNNius for HierHGT-DTI Benchmark

This directory contains a split-aligned GeNNius adaptation for the HierHGT-DTI
benchmark protocol.

Unlike the official GeNNius entrypoint, the adapted runner does not use
`RandomLinkSplit`. It builds the message-passing graph from training positive
edges only, while keeping the original fixed `train/val/test` labels for
evaluation.

## Usage

Prepare cache once per dataset/split:

```bash
python baselines/GeNNius/prepare_adapted.py --dataset BioSnap --split random
```

Train and evaluate:

```bash
python baselines/GeNNius/run_adapted.py --dataset BioSnap --split random --seed 42
```

Outputs are written to:

- `baselines/GeNNius/data_adapted/<dataset>_<split>/`
- `baselines/GeNNius/output_adapted/<dataset>_<split>/seed<seed>/`

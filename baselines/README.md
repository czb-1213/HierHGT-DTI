# Retained Baselines

This directory keeps only the baseline implementations that are used in the manuscript comparisons.

## Included models

- `MolTrans`
- `TransformerCPI`
- `DrugBAN`
- `DO-GMA`
- `GeNNius`

## Shared protocol

- fixed `train/val/test` CSV splits under `data/<Dataset>/<Split>/`
- manuscript datasets: `BioSnap` and `DrugBank`
- manuscript splits: `random`, `cold_drug`, `cold_protein`
- fixed candidate negatives shared across methods
- validation-only checkpoint selection
- shared metric-reporting pipeline

The main manuscript tables report AUROC and AUPR as mean +/- population standard deviation over seeds 42, 256 and 3407. The corresponding reviewer-facing summary files are `results/main_benchmark_summary.csv` and `results/main_benchmark_summary.md`.

## Run all retained baselines

```bash
bash baselines/run_selected_baselines.sh
```

## Run a subset

```bash
MODELS=moltrans,drugban DATASETS=BioSnap SPLITS=random,cold_protein SEEDS=42 bash baselines/run_selected_baselines.sh
```

Protocol details are summarized in `BENCHMARK_PROTOCOL.md`.

# Baseline Benchmark Protocol

All adapted baselines in `baselines/` should follow the same evaluation protocol.

- data split: fixed `train/val/test` CSV under `data/<Dataset>/<Split>/`
- selection metric: `val_aupr`
- threshold policy: choose the F1-optimal threshold on the validation split, then reuse it on test
- reported metrics: `auc`, `aupr`, `f1`, `acc`
- result metadata: `seed`, `best_epoch`, `early_stop_epoch`, `selection_metric`, `threshold`, `threshold_policy`

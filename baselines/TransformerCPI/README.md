# TransformerCPI for HierHGT-DTI Benchmark

This directory vendors the minimum official TransformerCPI assets needed for the
HierHGT-DTI baseline benchmark.

Source repository:

- Paper: <https://academic.oup.com/bioinformatics/article/36/16/4406/5840724>
- Code: <https://github.com/lifanchen-simm/transformerCPI>

Included upstream files:

- `model.py` adapted from the official `Train_details/BindingDB/model_v2.py`
- `mol_featurizer.py`
- `word2vec.py`
- `RAdam.py`
- `lookahead.py`
- `word2vec_30.model`

Benchmark entry points:

```bash
python baselines/TransformerCPI/convert_generate_data_splitssformercpi.py --dataset BioSnap --split random
python baselines/TransformerCPI/run_adapted.py --dataset BioSnap --split random --seed 42
```

The adapted runner follows the shared benchmark protocol:

- fixed `train/val/test` CSV input under `data/<Dataset>/<Split>/`
- model selection by `val_aupr`
- threshold chosen on validation by F1 and reused on test
- reported metrics: `auc`, `aupr`, `f1`, `acc`

Compatibility note:

- the official repository uses an RAdam+Lookahead training wrapper
- the packaged Lookahead implementation is not compatible with the torch 2.1 baseline environment here
- the adapted runner therefore keeps the official model structure but trains it with `AdamW`

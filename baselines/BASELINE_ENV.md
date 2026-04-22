# Baseline Environment

Keep the main HierHGT-DTI environment separate from the baseline environment.

## Recommended virtual environment path

```bash
/home/czb/.venv/hierhgt-dti-baseline
```

## Activate

```bash
source /home/czb/.venv/hierhgt-dti-baseline/bin/activate
```

## Install or refresh

```bash
uv pip install --native-tls   --python /home/czb/.venv/hierhgt-dti-baseline/bin/python   -r baselines/requirements.baseline.txt
```

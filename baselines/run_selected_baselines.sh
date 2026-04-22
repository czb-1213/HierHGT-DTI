#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-moltrans,transformercpi,drugban,do_gma,gennius}"
DATASETS="${DATASETS:-BioSnap,DrugBank}"
SPLITS="${SPLITS:-random,cold_drug,cold_protein}"
SEEDS="${SEEDS:-42}"
DATA_ROOT="${DATA_ROOT:-data}"
DRUGBAN_OUTPUT_DIR="${DRUGBAN_OUTPUT_DIR:-baselines/DrugBAN/result}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

contains_model() {
  local name="$1"
  [[ ",${MODELS}," == *",${name},"* ]]
}

validate_model_selection() {
  local model
  IFS=',' read -r -a MODEL_ARR <<< "$MODELS"
  for model in "${MODEL_ARR[@]}"; do
    case "$model" in
      moltrans|transformercpi|drugban|do_gma|gennius)
        ;;
      *)
        echo "[ERROR] unsupported model name: ${model}" >&2
        exit 1
        ;;
    esac
  done
}

IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"
IFS=',' read -r -a SPLIT_ARR <<< "$SPLITS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

validate_model_selection

if contains_model "moltrans"; then
  echo "[MolTrans] running..."
  for ds in "${DATASET_ARR[@]}"; do
    ds_lower="$(echo "$ds" | tr '[:upper:]' '[:lower:]')"
    for sp in "${SPLIT_ARR[@]}"; do
      task="${ds_lower}_${sp}"
      for seed in "${SEED_ARR[@]}"; do
        result_file="baselines/MolTrans/output_moltrans/${task}/seed${seed}/test_results.json"
        if [[ "$SKIP_EXISTING" == "true" && -f "$result_file" ]]; then
          echo "  -> task=${task}, seed=${seed} [SKIP: already exists]"
          continue
        fi
        echo "  -> task=${task}, seed=${seed}"
        "$PYTHON_BIN" baselines/MolTrans/train.py           --task "$task"           --data_root "$DATA_ROOT"           --output_dir baselines/MolTrans/output_moltrans           --batch-size 16           --epochs 60           --patience 8           --lr 1e-4           --seed "$seed"
      done
    done
  done
fi

if contains_model "drugban"; then
  echo "[DrugBAN] running..."
  for ds in "${DATASET_ARR[@]}"; do
    for sp in "${SPLIT_ARR[@]}"; do
      for seed in "${SEED_ARR[@]}"; do
        result_file="${DRUGBAN_OUTPUT_DIR}/${ds}_${sp}/seed${seed}/test_results.json"
        if [[ "$SKIP_EXISTING" == "true" && -f "$result_file" ]]; then
          echo "  -> dataset=${ds}, split=${sp}, seed=${seed} [SKIP: already exists]"
          continue
        fi
        echo "  -> dataset=${ds}, split=${sp}, seed=${seed}"
        "$PYTHON_BIN" baselines/DrugBAN/main.py           --cfg baselines/DrugBAN/configs/DrugBAN.yaml           --data "$ds"           --split "$sp"           --data_root "$DATA_ROOT"           --output_dir "$DRUGBAN_OUTPUT_DIR"           --seed "$seed"           --epochs 100           --patience 12           --lr 5e-5
      done
    done
  done
fi

if contains_model "transformercpi"; then
  echo "[TransformerCPI] running..."
  for ds in "${DATASET_ARR[@]}"; do
    for sp in "${SPLIT_ARR[@]}"; do
      echo "  -> dataset=${ds}, split=${sp}"
      "$PYTHON_BIN" baselines/TransformerCPI/convert_data_transformercpi.py         --dataset "$ds" --split "$sp" --data_root "$DATA_ROOT"
      for seed in "${SEED_ARR[@]}"; do
        result_file="baselines/TransformerCPI/output_adapted/${ds}_${sp}/seed${seed}/test_results.json"
        if [[ "$SKIP_EXISTING" == "true" && -f "$result_file" ]]; then
          echo "  -> dataset=${ds}, split=${sp}, seed=${seed} [SKIP: already exists]"
          continue
        fi
        "$PYTHON_BIN" baselines/TransformerCPI/run_adapted.py           --dataset "$ds" --split "$sp" --seed "$seed"
      done
    done
  done
fi

if contains_model "do_gma"; then
  echo "[DO-GMA] running..."
  for ds in "${DATASET_ARR[@]}"; do
    for sp in "${SPLIT_ARR[@]}"; do
      echo "  -> dataset=${ds}, split=${sp}"
      "$PYTHON_BIN" baselines/DO-GMA/prepare_adapted.py         --dataset "$ds" --split "$sp" --data_root "$DATA_ROOT"
      for seed in "${SEED_ARR[@]}"; do
        result_file="baselines/DO-GMA/output_adapted/${ds}_${sp}/seed${seed}/test_results.json"
        if [[ "$SKIP_EXISTING" == "true" && -f "$result_file" ]]; then
          echo "  -> dataset=${ds}, split=${sp}, seed=${seed} [SKIP: already exists]"
          continue
        fi
        "$PYTHON_BIN" baselines/DO-GMA/run_adapted.py           --dataset "$ds" --split "$sp" --seed "$seed" --epochs 60 --patience 8 --batch_size 64
      done
    done
  done
fi

if contains_model "gennius"; then
  echo "[GeNNius] running..."
  for ds in "${DATASET_ARR[@]}"; do
    for sp in "${SPLIT_ARR[@]}"; do
      echo "  -> dataset=${ds}, split=${sp}"
      "$PYTHON_BIN" baselines/GeNNius/prepare_adapted.py         --dataset "$ds" --split "$sp" --data_root "$DATA_ROOT"
      for seed in "${SEED_ARR[@]}"; do
        result_file="baselines/GeNNius/output_adapted/${ds}_${sp}/seed${seed}/test_results.json"
        if [[ "$SKIP_EXISTING" == "true" && -f "$result_file" ]]; then
          echo "  -> dataset=${ds}, split=${sp}, seed=${seed} [SKIP: already exists]"
          continue
        fi
        "$PYTHON_BIN" baselines/GeNNius/run_adapted.py           --dataset "$ds" --split "$sp" --seed "$seed"
      done
    done
  done
fi

echo "All retained baseline runs finished."

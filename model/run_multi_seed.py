"""
多 seed 训练自动化脚本（固定数据集划分）

说明：
1. 直接使用 base_config 中 data.train_file/val_file/test_file
2. 仅循环训练 seed，不再按 seed 重新划分数据集
3. 每个 seed 的结果保存到 output_seed{seed}/<dataset_tag>
"""

import os
import sys
import copy
import argparse
import yaml
from datetime import datetime
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "model")


def _resolve_path_from_config(path_value, base_config_path):
    """将 config 内路径解析为绝对路径（相对路径按 base_config 所在目录解析）。"""
    if not path_value:
        raise ValueError("Missing dataset path in base config")
    if os.path.isabs(path_value):
        return path_value
    cfg_dir = os.path.dirname(os.path.abspath(base_config_path))
    return os.path.abspath(os.path.join(cfg_dir, path_value))


def _infer_dataset_tag(train_file):
    """
    从 train_file 推断输出目录标签，优先识别:
      .../<Dataset>/<split>/train.csv -> <dataset>_<split>
    """
    split_modes = {"random", "cold_drug", "cold_protein"}
    parent = os.path.basename(os.path.dirname(train_file))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(train_file)))
    if parent in split_modes and grandparent:
        return f"{grandparent.lower()}_{parent}"
    return "from_config"


def create_config_for_seed(base_config, seed: int, data_files, output_dir: str):
    config = copy.deepcopy(base_config)

    config["seed"]["seed"] = seed
    config["output"]["output_dir"] = output_dir

    config["data"]["train_file"] = data_files["train_file"]
    config["data"]["val_file"] = data_files["val_file"]
    config["data"]["test_file"] = data_files["test_file"]

    config["data"]["drug_cache_dirs"] = [os.path.join(DATA_DIR, "drug_cache")]
    config["data"]["protein_cache_dirs"] = [os.path.join(DATA_DIR, "esm_cache")]

    config_path = os.path.join(output_dir, "config_seed.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return config_path


def run_training(config_path: str):
    train_script = os.path.join(MODEL_DIR, "train_hierhgt_dti.py")
    cmd = [
        sys.executable, train_script,
        "--config", config_path,
        "--mode", "single",
    ]

    result = subprocess.run(cmd, cwd=MODEL_DIR)
    return result.returncode


def run_single_experiment(seed: int, base_config, data_files, dataset_tag: str):
    print(f"\n{'#'*70}")
    print(f"# Seed={seed} | Dataset={dataset_tag}")
    print(f"{'#'*70}")

    output_dir = os.path.join(MODEL_DIR, f"output_seed{seed}", dataset_tag)
    config_path = create_config_for_seed(base_config, seed, data_files, output_dir)
    return_code = run_training(config_path)

    status = "SUCCESS" if return_code == 0 else "FAILED"
    print(f"\n[Seed {seed}] {dataset_tag}: {status}")
    print(f"[Seed {seed}] Results: {output_dir}")

    return return_code == 0


def main():
    parser = argparse.ArgumentParser(description="Multi-seed experiment automation (fixed dataset from config)")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="List of random seeds")
    parser.add_argument("--base_config", type=str, default=None, help="Base config file path")
    args = parser.parse_args()

    if args.base_config is None:
        args.base_config = os.path.join(MODEL_DIR, "config_hierhgt_dti.yaml")

    if not os.path.exists(args.base_config):
        print(f"Error: Base config not found: {args.base_config}")
        sys.exit(1)

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    data_cfg = base_config.get("data", {})
    train_file = _resolve_path_from_config(data_cfg.get("train_file"), args.base_config)
    val_file = _resolve_path_from_config(data_cfg.get("val_file"), args.base_config)
    test_file = _resolve_path_from_config(data_cfg.get("test_file"), args.base_config)
    for path in (train_file, val_file, test_file):
        if not os.path.exists(path):
            print(f"Error: Dataset file not found: {path}")
            sys.exit(1)

    data_files = {
        "train_file": train_file,
        "val_file": val_file,
        "test_file": test_file,
    }
    dataset_tag = _infer_dataset_tag(train_file)

    total_experiments = len(args.seeds)

    print(f"\n{'='*70}")
    print("Multi-seed Experiment")
    print(f"{'='*70}")
    print(f"Seeds:       {args.seeds}")
    print(f"Dataset tag: {dataset_tag}")
    print(f"Train file:  {train_file}")
    print(f"Val file:    {val_file}")
    print(f"Test file:   {test_file}")
    print(f"Total experiments: {total_experiments}")
    print(f"Start time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    exp_idx = 0

    for seed in args.seeds:
        exp_idx += 1
        print(f"\n\n{'='*70}")
        print(f"Experiment {exp_idx}/{total_experiments}")
        print(f"{'='*70}")

        success = run_single_experiment(seed, base_config, data_files, dataset_tag)
        key = f"seed{seed}_{dataset_tag}"
        results[key] = "SUCCESS" if success else "FAILED"

    print(f"\n\n{'='*70}")
    print("Final Summary")
    print(f"{'='*70}")
    success_count = sum(1 for v in results.values() if v == "SUCCESS")
    failed_count = len(results) - success_count
    print(f"Total: {len(results)} | Success: {success_count} | Failed: {failed_count}")
    print("\nDetailed Results:")
    for key, status in results.items():
        print(f"  {key}: {status}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

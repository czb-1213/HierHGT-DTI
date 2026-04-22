"""
消融实验自动化脚本

设计原则：
  - 以 no_cross（无 sub↔pocket 交叉边）为主模型 baseline。
  - 通过 --dataset / --split 直接选择数据集和划分方式，数据固定在 data/ 目录下。
  - 模型超参从 --base_config（默认 config_hierhgt_dti.yaml）读取，数据路径由命令行覆盖。
  - 多 seed 只改训练随机种子，数据划分不变。
  - 输出目录：output_ablation/{dataset}_{split}/{ablation_name}/seed{seed}/

数据集布局（data/ 目录下）：
  data/
    BioSnap/{random,cold_drug,cold_protein}/{train,val,test}.csv
    DrugBank/{random,cold_drug,cold_protein}/{train,val,test}.csv
    drug_cache/
    esm_cache/

核心消融（6组 + baseline，论文正文）：
  0.  ours             — 最强基线（hierarchy + shortcut + typed edges + learned relation aggregation,
                         disable_sub_pocket_cross=True, flow gate off）
  1.  no_hierarchy     — 去掉 sub/pocket 中间层
                         验证：多尺度层级结构的必要性
  2.  no_relation_gate — 冻结 relation logits，退化为近似均匀 relation 权重
                         验证：学习的关系重要性是否优于启发式均匀分配
  3.  hier_mean        — attention 聚合→mean 聚合
                         验证：子节点重要性是否需要自适应建模
  4.  no_typed_edges   — 边类型退化为同质
                         验证：细粒度关系类型的贡献
  5.  no_shortcut      — 去掉 atom→drug / residue→protein 直连边
                         验证：shortcut 对缓解信息瓶颈的作用
  6.  hgt_1layer       — 2 层 HGT→1 层
                         验证：多层传播的必要性

论文叙事（以最强基线为出发点，逐项去除组件）：
  ours vs no_hierarchy     → "层级结构是关键创新，不是可有可无的中间层"
  ours vs no_relation_gate → "学习的关系重要性优于启发式均匀分配"
  ours vs hier_mean        → "attention 聚合优于简单平均，说明子节点贡献不均匀"
  ours vs no_typed_edges   → "细粒度边类型对关系感知消息传递至关重要"
  ours vs no_shortcut      → "shortcut 可以避免层级中继带来的信息瓶颈"
  ours vs hgt_1layer       → "单层 HGT 不足以完成有效信息传播"

用法:
  # BioSnap random split，主表消融
  python run_hierhgt_dti_ablation.py --dataset biosnap --split random --seeds 42

  # 两个数据集 × 三种划分，全跑
  python run_hierhgt_dti_ablation.py --dataset biosnap drugbank \\
      --split random cold_drug cold_protein --seeds 42 256 1213

  # 只跑指定消融 + 指定数据集
  python run_hierhgt_dti_ablation.py --dataset drugbank --split cold_protein \\
      --ablations ours no_hierarchy --seeds 42

  # 跳过已完成的实验
  python run_hierhgt_dti_ablation.py --dataset biosnap --split random --seeds 42 --skip_existing
"""

import os
import sys
import copy
import argparse
import yaml
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 消融实验定义 ──────────────────────────────────────────────
# 每个消融通过一个函数修改 config dict，返回修改后的 config。
# 基线 (ours) = 最强配置。
# 所有论文消融都从这一显式基线出发，只改变一个因素。

def _ensure_relation_gate_cfg(subpocket_cfg):
    gate_cfg = subpocket_cfg.setdefault("relation_msg_gate", {})
    gate_cfg.pop("entropy_lambda", None)
    gate_cfg["l1_lambda"] = 1.0e-4
    gate_cfg["init_bias"] = 0.0
    gate_cfg["temperature"] = 1.0
    gate_cfg["degree_scale"] = 1.0
    gate_cfg["freeze"] = False
    return gate_cfg


def _ablation_ours(cfg):
    """最强基线：hierarchy + shortcut + learned relation aggregation，关闭 typed edges / flow gate / sub↔pocket 交叉边。"""
    model_cfg = cfg["model"]
    rel_cfg = model_cfg.setdefault("relational_view", {})
    subpocket_cfg = rel_cfg.setdefault("subpocket", {})
    hgt_cfg = rel_cfg.setdefault("hgt", {})
    predictor_cfg = model_cfg.setdefault("predictor", {})

    rel_cfg["drug_use_bond_types"] = False
    rel_cfg["protein_use_seq_gap_types"] = False
    rel_cfg["protein_use_contact_bin_types"] = False
    rel_cfg["protein_edge_granularity"] = "fine"

    subpocket_cfg["disable_sub_pocket_cross"] = True
    subpocket_cfg["disable_cross_edges"] = False
    subpocket_cfg["disable_hierarchy"] = False
    subpocket_cfg["no_shortcut"] = False
    subpocket_cfg["mid_aggregation"] = False
    subpocket_cfg["aggregation"] = "attention"
    subpocket_cfg.pop("mask_edge_types", None)

    flow_gate_cfg = subpocket_cfg.setdefault("flow_gate", {})
    flow_gate_cfg["enabled"] = False
    flow_gate_cfg.setdefault("hidden_dim", 64)
    flow_gate_cfg.setdefault("init_bias", -2.0)

    _ensure_relation_gate_cfg(subpocket_cfg)

    hgt_cfg["n_layers"] = 2
    predictor_cfg["mode"] = "single_global"
    return cfg


def _ablation_no_hierarchy(cfg):
    """去掉 substructure/pocket 中间层，退化为 atom→drug / residue→protein 两级结构。"""
    cfg = _ablation_ours(cfg)
    cfg["model"]["relational_view"]["subpocket"]["disable_hierarchy"] = True
    return cfg


def _ablation_no_relation_gate(cfg):
    """冻结 relation logits，使 relation-level aggregation 退化为近似均匀权重。"""
    cfg = _ablation_ours(cfg)
    gate_cfg = cfg["model"]["relational_view"]["subpocket"]["relation_msg_gate"]
    gate_cfg["freeze"] = True
    gate_cfg["l1_lambda"] = 0.0
    gate_cfg["temperature"] = 1.0e6
    gate_cfg["degree_scale"] = 0.0
    gate_cfg["init_bias"] = 0.0
    return cfg


def _ablation_hier_mean(cfg):
    """层次聚合使用 mean（替代 attention）。"""
    cfg = _ablation_ours(cfg)
    cfg["model"]["relational_view"]["subpocket"]["aggregation"] = "mean"
    return cfg


def _ablation_with_typed_edges(cfg):
    """Drug/Protein 边做细粒度类型细分（bond_single/double/... + seq_gap/contact_bin）。"""
    cfg = _ablation_ours(cfg)
    cfg["model"]["relational_view"]["drug_use_bond_types"] = True
    cfg["model"]["relational_view"]["protein_use_seq_gap_types"] = True
    cfg["model"]["relational_view"]["protein_use_contact_bin_types"] = True
    return cfg


def _ablation_no_shortcut(cfg):
    """去掉 atom→drug / residue→protein 直连边，强制信息走层级路径。"""
    cfg = _ablation_ours(cfg)
    cfg["model"]["relational_view"]["subpocket"]["no_shortcut"] = True
    return cfg


def _ablation_hgt_1layer(cfg):
    """HGT 只用 1 层。"""
    cfg = _ablation_ours(cfg)
    cfg["model"]["relational_view"]["hgt"]["n_layers"] = 1
    return cfg


# ── 核心消融（论文正文，6组 + baseline） ──
ABLATION_REGISTRY_MAIN = {
    "ours":              _ablation_ours,
    "no_hierarchy":      _ablation_no_hierarchy,
    "no_relation_gate":  _ablation_no_relation_gate,
    "hier_mean":         _ablation_hier_mean,
    "with_typed_edges":  _ablation_with_typed_edges,
    "no_shortcut":       _ablation_no_shortcut,
    "hgt_1layer":        _ablation_hgt_1layer,
}

# ── 全部消融 ──
ABLATION_REGISTRY = {**ABLATION_REGISTRY_MAIN}


# ── 数据集 / 划分方式定义 ─────────────────────────────────────
# CLI 短名 -> data/ 下的实际目录名
DATASET_MAP = {
    "biosnap":  "BioSnap",
    "drugbank": "DrugBank",
}
SPLIT_CHOICES = ["random", "cold_drug", "cold_protein"]


def _resolve_data_files(dataset_key, split):
    """根据 dataset 短名和 split 返回 train/val/test 绝对路径。"""
    dataset_dir_name = DATASET_MAP[dataset_key]
    split_dir = os.path.join(DATA_DIR, dataset_dir_name, split)
    data_files = {
        "train_file": os.path.join(split_dir, "train.csv"),
        "val_file":   os.path.join(split_dir, "val.csv"),
        "test_file":  os.path.join(split_dir, "test.csv"),
    }
    for name, path in data_files.items():
        if not os.path.exists(path):
            print(f"Error: {name} not found: {path}")
            sys.exit(1)
    return data_files


def _dataset_tag(dataset_key, split):
    """输出目录标签，如 biosnap_random, drugbank_cold_drug。"""
    return f"{dataset_key}_{split}"


# ── 配置生成 & 训练 ───────────────────────────────────────────

def create_ablation_config(base_config, seed, data_files, output_dir, ablation_fn):
    """深拷贝 base_config，应用消融修改，写入 output_dir。"""
    cfg = copy.deepcopy(base_config)

    cfg["seed"]["seed"] = seed
    cfg["output"]["output_dir"] = output_dir
    cfg["data"]["train_file"] = data_files["train_file"]
    cfg["data"]["val_file"] = data_files["val_file"]
    cfg["data"]["test_file"] = data_files["test_file"]
    cfg["data"]["drug_cache_dirs"] = [os.path.join(DATA_DIR, "drug_cache")]
    cfg["data"]["protein_cache_dirs"] = [os.path.join(DATA_DIR, "esm_cache")]

    cfg = ablation_fn(cfg)

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config_ablation.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return config_path


def run_training(config_path):
    train_script = os.path.join(MODEL_DIR, "train_hierhgt_dti.py")
    cmd = [sys.executable, train_script, "--config", config_path, "--mode", "single"]
    result = subprocess.run(cmd, cwd=MODEL_DIR)
    return result.returncode


def has_finished(output_dir):
    """检查该实验是否已完成（存在 test_results.json）。"""
    return os.path.isfile(os.path.join(output_dir, "test_results.json"))


# ── 主流程 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablation study automation")
    parser.add_argument("--dataset", type=str, nargs="+", required=True,
                        choices=list(DATASET_MAP.keys()),
                        help="数据集 (biosnap / drugbank)")
    parser.add_argument("--split", type=str, nargs="+", required=True,
                        choices=SPLIT_CHOICES,
                        help="划分方式 (random / cold_drug / cold_protein)")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--ablations", type=str, nargs="+",
                        default=list(ABLATION_REGISTRY_MAIN.keys()),
                        choices=list(ABLATION_REGISTRY.keys()),
                        help="要运行的消融实验 (默认主表 7 组)")
    parser.add_argument("--base_config", type=str, default=None,
                        help="模型超参配置文件 (默认 config_hierhgt_dti.yaml，仅读取模型/训练超参)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="跳过已有 test_results.json 的实验")
    args = parser.parse_args()

    # ── 加载模型超参配置 ──
    if args.base_config is None:
        args.base_config = os.path.join(MODEL_DIR, "config_hierhgt_dti.yaml")
    if not os.path.exists(args.base_config):
        print(f"Error: Base config not found: {args.base_config}")
        sys.exit(1)
    with open(args.base_config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # ── 展开所有 (dataset, split) 组合 ──
    combos = [(ds, sp) for ds in args.dataset for sp in args.split]
    total = len(combos) * len(args.ablations) * len(args.seeds)

    print(f"\n{'='*70}")
    print(f"Ablation Study")
    print(f"{'='*70}")
    print(f"Datasets:   {args.dataset}")
    print(f"Splits:     {args.split}")
    print(f"Seeds:      {args.seeds}")
    print(f"Ablations:  {args.ablations}")
    print(f"Config:     {args.base_config}")
    print(f"Skip exist: {args.skip_existing}")
    print(f"Total runs: {total}")
    print(f"Start:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    results = {}
    idx = 0

    for dataset_key, split in combos:
        tag = _dataset_tag(dataset_key, split)
        data_files = _resolve_data_files(dataset_key, split)

        for ablation_name in args.ablations:
            for seed in args.seeds:
                idx += 1
                key = f"{tag}/{ablation_name}/seed{seed}"
                output_dir = os.path.join(
                    MODEL_DIR,
                    "output_ablation",
                    tag,
                    ablation_name,
                    f"seed{seed}",
                )

                print(f"\n[{idx}/{total}] {key}")

                if args.skip_existing and has_finished(output_dir):
                    print(f"  SKIP (already finished)")
                    results[key] = "SKIPPED"
                    continue

                ablation_fn = ABLATION_REGISTRY[ablation_name]
                config_path = create_ablation_config(
                    base_config, seed, data_files, output_dir, ablation_fn,
                )

                rc = run_training(config_path)
                status = "SUCCESS" if rc == 0 else "FAILED"
                results[key] = status
                print(f"  {status} -> {output_dir}")

    # ── 汇总 ──
    print(f"\n\n{'='*70}")
    print(f"Ablation Study Summary")
    print(f"{'='*70}")
    n_success = sum(1 for v in results.values() if v == "SUCCESS")
    n_skip = sum(1 for v in results.values() if v == "SKIPPED")
    n_fail = sum(1 for v in results.values() if v == "FAILED")
    print(f"Total: {len(results)} | Success: {n_success} | Skipped: {n_skip} | Failed: {n_fail}")
    print(f"\nDetails:")
    for key, status in results.items():
        print(f"  {key}: {status}")
    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

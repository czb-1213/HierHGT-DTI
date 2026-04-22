"""
Unified training entry for HierHGT-DTI experiments.
"""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm
import json
import random
import traceback
import warnings
from contextlib import nullcontext
from typing import Optional

from hierhgt_dti_model import HierHGTDTIModel
from hierhgt_dti_dataset import HierHGTDTIDataset, hierhgt_dti_collate_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HierHGTDTITrainer')


def _resolve_path_from_base(path_value, base_dir):
    if path_value is None:
        return None
    path_str = str(path_value).strip()
    if not path_str:
        return path_value
    expanded = os.path.expanduser(path_str)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(base_dir, expanded))


def _resolve_runtime_paths(config, base_config_path):
    base_dir = os.path.dirname(os.path.abspath(base_config_path))
    data_cfg = config.get("data", {})
    for key in ("train_file", "val_file", "test_file"):
        if key in data_cfg and data_cfg.get(key):
            data_cfg[key] = _resolve_path_from_base(data_cfg.get(key), base_dir)
    for key in ("drug_cache_dirs", "protein_cache_dirs"):
        dirs = data_cfg.get(key)
        if isinstance(dirs, list):
            data_cfg[key] = [_resolve_path_from_base(p, base_dir) for p in dirs]
    output_cfg = config.get("output", {})
    if output_cfg.get("output_dir"):
        output_cfg["output_dir"] = _resolve_path_from_base(output_cfg.get("output_dir"), base_dir)
    return config


def set_seed(seed, deterministic=True, benchmark=False):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = benchmark


def compute_metrics(preds, labels, threshold=0.5):
    """计算评估指标"""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        accuracy_score, precision_score, recall_score, f1_score
    )

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        auc_roc = float("nan")
        auc_pr = float("nan")
    else:
        # AUC-ROC
        auc_roc = roc_auc_score(labels, preds)
        # AUC-PR
        auc_pr = average_precision_score(labels, preds)

    # Accuracy, Precision, Recall, F1
    pred_labels = (preds >= threshold).astype(int)
    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels, zero_division=0)
    recall = recall_score(labels, pred_labels, zero_division=0)
    f1 = f1_score(labels, pred_labels, zero_division=0)

    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }



def _to_detached_scalar_tensor(value, ref_tensor):
    """
    Convert possibly-missing metric values to detached scalar tensor on ref device.
    This avoids frequent host synchronization in per-batch loops.
    """
    if value is None:
        return ref_tensor.new_zeros(())
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return ref_tensor.new_zeros(())
        return value.detach().float().mean()
    return ref_tensor.new_tensor(float(value), dtype=torch.float32)



def _parse_amp_dtype(dtype_value):
    dtype_key = str(dtype_value).lower()
    if dtype_key in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype_key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(
        f"Unsupported train.amp.dtype='{dtype_value}'. Use one of: float16, bfloat16."
    )


def _resolve_amp_settings(train_cfg, device):
    amp_cfg = train_cfg.get("amp", {})
    requested_enabled = amp_cfg.get("enabled", False)
    amp_enabled = bool(requested_enabled) and device.type == "cuda"
    amp_dtype = _parse_amp_dtype(amp_cfg.get("dtype", "float16"))
    use_scaler = amp_enabled and amp_dtype == torch.float16
    return amp_enabled, amp_dtype, use_scaler


def _build_amp_autocast(device, enabled, dtype):
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled, dtype=dtype)
    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def _build_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _suppress_known_dgl_amp_future_warnings():
    # DGL<new-amp-api internally calls torch.cuda.amp.*, which emits FutureWarning
    # on newer PyTorch. Keep this narrow to DGL sparse backend only.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"dgl\.backend\.pytorch\.sparse",
        message=r".*torch\.cuda\.amp\..*deprecated.*",
    )


def _compute_cross_edge_weight(config, epoch):
    """Compute cross-edge weight for SubPocket warmup."""
    rel_cfg = config.get("model", {}).get("relational_view", {})
    subpocket_cfg = rel_cfg.get("subpocket", {})
    warmup_epochs = int(subpocket_cfg.get("cross_edge_warmup_epochs", 5))
    warmup_start = float(subpocket_cfg.get("cross_edge_warmup_start", 0.1))
    if warmup_epochs <= 1 or epoch >= warmup_epochs:
        return 1.0
    progress = min(1.0, max(0.0, float(epoch - 1) / float(warmup_epochs - 1)))
    return warmup_start + (1.0 - warmup_start) * progress




def _build_optimizer_param_groups(model, base_lr, rel_lr, weight_decay):
    rel_module_names = {
        # Joint HGT backbone
        "hgt",
        "hgt_stage1",
        "hgt_stage2",
        # Global pair predictor
        "pair_mlp_r",
        "predictor_rel",
        # Hierarchical aggregators (SubPocket)
        "atom_to_sub_agg",
        "res_to_pocket_agg",
        "affinity_map_generator",
        # Cross-edge gate
        # Token-level predictor branch
        "token_atom_proj",
        "token_res_proj",
        "token_pair_mlp",
        "predictor_token",
        # Fusion gate
        "fusion_gate_mlp",
    }
    # Direct nn.Parameter names (no '.' prefix) that belong to the relational group.
    rel_param_prefixes = {
        "drug_super_node_emb",
        "protein_super_node_emb",
        "sub_super_node_emb",
        "pocket_super_node_emb",
    }

    rel_decay = []
    rel_no_decay = []
    main_decay = []
    main_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        root = name.split(".", 1)[0]
        is_rel_group = root in rel_module_names or name in rel_param_prefixes
        is_no_decay = (param.ndim <= 1) or name.endswith(".bias")
        if is_rel_group and is_no_decay:
            rel_no_decay.append(param)
        elif is_rel_group and (not is_no_decay):
            rel_decay.append(param)
        elif (not is_rel_group) and is_no_decay:
            main_no_decay.append(param)
        else:
            main_decay.append(param)

    groups = []
    if rel_decay:
        groups.append(
            {
                "params": rel_decay,
                "weight_decay": weight_decay,
                "lr": float(rel_lr),
                "group_name": "rel",
            }
        )
    if rel_no_decay:
        groups.append(
            {
                "params": rel_no_decay,
                "weight_decay": 0.0,
                "lr": float(rel_lr),
                "group_name": "rel",
            }
        )
    if main_decay:
        groups.append(
            {
                "params": main_decay,
                "weight_decay": weight_decay,
                "lr": float(base_lr),
                "group_name": "main",
            }
        )
    if main_no_decay:
        groups.append(
            {
                "params": main_no_decay,
                "weight_decay": 0.0,
                "lr": float(base_lr),
                "group_name": "main",
            }
        )
    return groups


def _create_scheduler(optimizer, train_config, lr_scheduler_type, t_max_epochs):
    if lr_scheduler_type == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=train_config.get('lr_factor', 0.5),
            patience=train_config.get('lr_patience', 5),
            min_lr=train_config.get('lr_min', 1e-6),
        )
    if lr_scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(t_max_epochs)),
            eta_min=train_config.get('lr_min', 1e-6),
        )
    return None


def _precompute_graph_etype_ids(model, graph, graph_kind):
    if graph_kind == "drug":
        if "drug_etype_id" in graph.edata:
            return False
        src, dst = graph.edges(order="eid")
        edge_feat = model._get_graph_edge_feat(graph, model.drug_edge_feat_dim, "drug_graph")
        graph.edata["drug_etype_id"] = model._decode_drug_homo_etype_indices(
            src, dst, edge_feat
        ).to(dtype=torch.long)
        return True

    if graph_kind == "protein":
        if model.protein_edge_granularity == "coarse":
            key = model.protein_coarse_cache_key
            decode_fn = model._decode_protein_homo_etype_indices_coarse
        else:
            key = "protein_etype_id_fine"
            decode_fn = model._decode_protein_homo_etype_indices

        if key in graph.edata:
            return False
        src, dst = graph.edges(order="eid")
        edge_feat = model._get_graph_edge_feat(graph, model.protein_edge_feat_dim, "protein_graph")
        graph.edata[key] = decode_fn(src, dst, edge_feat).to(dtype=torch.long)
        return True

    raise ValueError(f"Unsupported graph_kind: {graph_kind}")


def _maybe_precompute_dataset_etype_ids(model, dataset, split_name):
    if not bool(getattr(dataset, "cache_in_memory", False)):
        return
    drug_cache = getattr(dataset, "drug_graph_cache", None)
    protein_cache = getattr(dataset, "protein_graph_cache", None)
    if not isinstance(drug_cache, dict) or not isinstance(protein_cache, dict):
        return
    if len(drug_cache) == 0 and len(protein_cache) == 0:
        return

    added_drug = 0
    added_protein = 0
    with torch.no_grad():
        for graph in drug_cache.values():
            if _precompute_graph_etype_ids(model, graph, "drug"):
                added_drug += 1
        for graph in protein_cache.values():
            if _precompute_graph_etype_ids(model, graph, "protein"):
                added_protein += 1

    if added_drug > 0 or added_protein > 0:
        logger.info(
            "预计算 %s 边类型索引: drug=%d, protein=%d",
            split_name,
            added_drug,
            added_protein,
        )


def _resolve_protein_etype_cache_key(model):
    if str(getattr(model, "protein_edge_granularity", "coarse")).lower() == "coarse":
        return str(getattr(model, "protein_coarse_cache_key"))
    return "protein_etype_id_fine"


def _prepare_dataset_local_typed_edges(model, dataset, split_name):
    if not hasattr(dataset, "prepare_local_typed_edges"):
        raise AttributeError(
            f"{split_name} dataset does not implement prepare_local_typed_edges(...)."
        )
    dataset.prepare_local_typed_edges(
        drug_num_etypes=len(getattr(model, "drug_homo_etypes")),
        protein_num_etypes=len(getattr(model, "protein_homo_etypes")),
        protein_etype_key=_resolve_protein_etype_cache_key(model),
    )
    logger.info(
        "已准备 %s local typed edges: drug_etypes=%d, protein_etypes=%d",
        split_name,
        len(getattr(model, "drug_homo_etypes")),
        len(getattr(model, "protein_homo_etypes")),
    )


def _move_typed_edge_batch_to_device(typed_edge_batch, device):
    if not isinstance(typed_edge_batch, dict):
        raise TypeError("typed_edge_batch must be a dict.")

    required_keys = (
        "drug_edges_src",
        "drug_edges_dst",
        "drug_edges_ptr",
        "protein_edges_src",
        "protein_edges_dst",
        "protein_edges_ptr",
        "protein_other_drop_ratio",
        # SubPocket hierarchical mappings
        "atom_to_sub",
        "sub_counts",
        "total_subs",
        "res_to_pocket",
        "pocket_counts",
        "total_pockets",
    )
    missing_keys = [k for k in required_keys if k not in typed_edge_batch]
    if missing_keys:
        raise ValueError(f"typed_edge_batch is missing required packed keys: {missing_keys}")

    out = {}
    for key in (
        "drug_edges_src",
        "drug_edges_dst",
        "drug_edges_ptr",
        "protein_edges_src",
        "protein_edges_dst",
        "protein_edges_ptr",
    ):
        tensor = typed_edge_batch[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"typed_edge_batch['{key}'] must be a Tensor.")
        if tensor.ndim != 1:
            raise ValueError(f"typed_edge_batch['{key}'] must be 1D, got shape={tuple(tensor.shape)}.")
        out[key] = tensor.to(
            device=device,
            dtype=torch.int64,
            non_blocking=True,
        ).contiguous()

    protein_other_drop_ratio = typed_edge_batch["protein_other_drop_ratio"]
    if not isinstance(protein_other_drop_ratio, torch.Tensor):
        raise TypeError("typed_edge_batch['protein_other_drop_ratio'] must be a Tensor.")
    if protein_other_drop_ratio.ndim != 1:
        raise ValueError(
            "typed_edge_batch['protein_other_drop_ratio'] must be 1D, "
            f"got shape={tuple(protein_other_drop_ratio.shape)}."
        )
    out["protein_other_drop_ratio"] = protein_other_drop_ratio.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).contiguous()

    for key in ("atom_to_sub", "sub_counts", "res_to_pocket", "pocket_counts"):
        tensor = typed_edge_batch[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"typed_edge_batch['{key}'] must be a Tensor.")
        if tensor.ndim != 1:
            raise ValueError(f"typed_edge_batch['{key}'] must be 1D, got shape={tuple(tensor.shape)}.")
        out[key] = tensor.to(
            device=device,
            dtype=torch.int64,
            non_blocking=True,
        ).contiguous()

    for key in ("total_subs", "total_pockets"):
        scalar = typed_edge_batch[key]
        if isinstance(scalar, torch.Tensor):
            if scalar.numel() != 1:
                raise ValueError(f"typed_edge_batch['{key}'] tensor must contain exactly one element.")
            scalar = int(scalar.item())
        else:
            scalar = int(scalar)
        out[key] = scalar

    protein_edges_ew = typed_edge_batch.get("protein_edges_ew", None)
    if isinstance(protein_edges_ew, torch.Tensor):
        out["protein_edges_ew"] = protein_edges_ew.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ).contiguous()

    return out


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    config,
    epoch,
    amp_enabled=False,
    amp_dtype=torch.float16,
    scaler=None,
):
    """Train one epoch for the atom-residue cross-edge pathway."""
    model.train()

    train_cfg = config.get('train', {})
    cross_edge_weight = _compute_cross_edge_weight(config, epoch)
    log_interval = max(1, int(train_cfg.get('log_interval', 20)))

    sums = {
        'total_loss': torch.zeros((), device=device, dtype=torch.float32),
        'pred_loss': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_l1': torch.zeros((), device=device, dtype=torch.float32),
        'protein_other_drop_ratio': torch.zeros((), device=device, dtype=torch.float32),
        'cross_edge_weight': torch.zeros((), device=device, dtype=torch.float32),
        'token_valid_ratio': torch.zeros((), device=device, dtype=torch.float32),
        'token_fallback_ratio': torch.zeros((), device=device, dtype=torch.float32),
        'fusion_alpha_mean': torch.zeros((), device=device, dtype=torch.float32),
        'hier_flow_gate_drug_mean': torch.zeros((), device=device, dtype=torch.float32),
        'hier_flow_gate_protein_mean': torch.zeros((), device=device, dtype=torch.float32),
        'hier_flow_gate_drug_min': torch.zeros((), device=device, dtype=torch.float32),
        'hier_flow_gate_protein_min': torch.zeros((), device=device, dtype=torch.float32),
        'hier_flow_gate_drug_max': torch.zeros((), device=device, dtype=torch.float32),
        'hier_flow_gate_protein_max': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_mean': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_min': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_max': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_hierarchy_mean': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_hierarchy_min': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_hierarchy_max': torch.zeros((), device=device, dtype=torch.float32),
    }

    relation_msg_gate_cfg = config.get('model', {}).get('relational_view', {}).get('subpocket', {}).get('relation_msg_gate', {})
    relation_msg_gate_l1_lambda = float(
        relation_msg_gate_cfg.get('l1_lambda', relation_msg_gate_cfg.get('entropy_lambda', 0.0))
    )
    all_preds = []
    all_labels = []
    relation_msg_gate_value_sums = {}
    relation_msg_gate_value_counts = {}
    relation_msg_gate_hierarchy_mass_sum = 0.0
    relation_msg_gate_hierarchy_mass_count = 0
    relation_msg_gate_hierarchy_mass_batch_means = []
    n_batches = 0
    pin_memory_enabled = bool(getattr(dataloader, "pin_memory", False))

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (drug_graphs, protein_graphs, labels, _, _, typed_edge_batch) in enumerate(pbar):
        if drug_graphs is None or labels is None:
            continue

        n_batches += 1

        drug_graphs = drug_graphs.to(device, non_blocking=pin_memory_enabled)
        protein_graphs = protein_graphs.to(device, non_blocking=pin_memory_enabled)
        labels = labels.to(device, non_blocking=pin_memory_enabled)
        typed_edge_batch = _move_typed_edge_batch_to_device(
            typed_edge_batch,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        amp_context = _build_amp_autocast(device, enabled=amp_enabled, dtype=amp_dtype)
        with amp_context:
            outputs = model(
                drug_graphs,
                protein_graphs,
                typed_edge_batch=typed_edge_batch,
                cross_edge_weight=cross_edge_weight,
            )
            logits = outputs['logits']
            pred_loss = criterion(logits, labels)

            total_loss = pred_loss

            relation_msg_gate_l1 = outputs.get('relation_msg_gate_l1')
            if relation_msg_gate_l1 is None:
                relation_msg_gate_l1 = pred_loss.new_zeros(())
            if relation_msg_gate_l1_lambda > 0.0:
                total_loss = total_loss + relation_msg_gate_l1_lambda * relation_msg_gate_l1

        grad_clip_norm = train_cfg.get('grad_clip_norm', None)
        scaler_enabled = bool(scaler is not None and scaler.is_enabled())
        if scaler_enabled:
            scaler.scale(total_loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        ref_scalar = total_loss.detach().float()
        sums['total_loss'] += ref_scalar
        sums['pred_loss'] += pred_loss.detach().float()
        sums['relation_msg_gate_l1'] += relation_msg_gate_l1.detach().float()
        sums['cross_edge_weight'] += _to_detached_scalar_tensor(outputs.get('cross_edge_weight'), ref_scalar)
        sums['token_valid_ratio'] += _to_detached_scalar_tensor(outputs.get('token_valid_ratio'), ref_scalar)
        sums['token_fallback_ratio'] += _to_detached_scalar_tensor(outputs.get('token_fallback_ratio'), ref_scalar)
        sums['fusion_alpha_mean'] += _to_detached_scalar_tensor(outputs.get('fusion_alpha_mean'), ref_scalar)
        sums['hier_flow_gate_drug_mean'] += _to_detached_scalar_tensor(outputs.get('hier_flow_gate_drug_mean'), ref_scalar)
        sums['hier_flow_gate_protein_mean'] += _to_detached_scalar_tensor(outputs.get('hier_flow_gate_protein_mean'), ref_scalar)
        sums['hier_flow_gate_drug_min'] += _to_detached_scalar_tensor(outputs.get('hier_flow_gate_drug_min'), ref_scalar)
        sums['hier_flow_gate_protein_min'] += _to_detached_scalar_tensor(outputs.get('hier_flow_gate_protein_min'), ref_scalar)
        sums['hier_flow_gate_drug_max'] += _to_detached_scalar_tensor(outputs.get('hier_flow_gate_drug_max'), ref_scalar)
        sums['hier_flow_gate_protein_max'] += _to_detached_scalar_tensor(outputs.get('hier_flow_gate_protein_max'), ref_scalar)
        sums['relation_msg_gate_mean'] += _to_detached_scalar_tensor(outputs.get('relation_msg_gate_mean'), ref_scalar)
        sums['relation_msg_gate_min'] += _to_detached_scalar_tensor(outputs.get('relation_msg_gate_min'), ref_scalar)
        sums['relation_msg_gate_max'] += _to_detached_scalar_tensor(outputs.get('relation_msg_gate_max'), ref_scalar)
        sums['relation_msg_gate_hierarchy_mean'] += _to_detached_scalar_tensor(outputs.get('relation_msg_gate_hierarchy_mean'), ref_scalar)
        sums['relation_msg_gate_hierarchy_min'] += _to_detached_scalar_tensor(outputs.get('relation_msg_gate_hierarchy_min'), ref_scalar)
        sums['relation_msg_gate_hierarchy_max'] += _to_detached_scalar_tensor(outputs.get('relation_msg_gate_hierarchy_max'), ref_scalar)
        sums['protein_other_drop_ratio'] += _to_detached_scalar_tensor(outputs.get('protein_other_drop_ratio'), ref_scalar)
        if hasattr(model, 'get_relation_msg_gate_value_stat_map'):
            stat_map = model.get_relation_msg_gate_value_stat_map()
            for edge_type, (budget_sum, count) in stat_map.items():
                relation_msg_gate_value_sums[edge_type] = relation_msg_gate_value_sums.get(edge_type, 0.0) + float(budget_sum)
                relation_msg_gate_value_counts[edge_type] = relation_msg_gate_value_counts.get(edge_type, 0) + int(count)
        if hasattr(model, 'get_relation_msg_gate_hierarchy_mass_stat'):
            hier_sum, hier_count = model.get_relation_msg_gate_hierarchy_mass_stat()
            relation_msg_gate_hierarchy_mass_sum += float(hier_sum)
            relation_msg_gate_hierarchy_mass_count += int(hier_count)
            if hier_count > 0:
                relation_msg_gate_hierarchy_mass_batch_means.append(float(hier_sum) / float(hier_count))

        all_preds.append(torch.sigmoid(logits).detach())
        all_labels.append(labels.detach())

        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{float(total_loss.detach().item()):.4f}',
                'cew': f'{cross_edge_weight:.3f}',
            })

    denom = max(1, n_batches)
    preds_np = torch.cat(all_preds, dim=0).float().cpu().numpy() if all_preds else np.array([])
    labels_np = torch.cat(all_labels, dim=0).float().cpu().numpy() if all_labels else np.array([])
    inv_denom = 1.0 / float(denom)
    avg = lambda key: float((sums[key] * inv_denom).item())
    relation_msg_gate_value_map = {}
    for edge_type, total_budget in relation_msg_gate_value_sums.items():
        total_count = relation_msg_gate_value_counts.get(edge_type, 0)
        if total_count > 0:
            relation_msg_gate_value_map[edge_type] = total_budget / float(total_count)
    if relation_msg_gate_value_map:
        relation_value_list = list(relation_msg_gate_value_map.values())
        relation_msg_gate_mean = float(sum(relation_value_list) / len(relation_value_list))
        relation_msg_gate_min = float(min(relation_value_list))
        relation_msg_gate_max = float(max(relation_value_list))
    else:
        relation_msg_gate_mean = avg('relation_msg_gate_mean')
        relation_msg_gate_min = avg('relation_msg_gate_min')
        relation_msg_gate_max = avg('relation_msg_gate_max')
    if relation_msg_gate_hierarchy_mass_count > 0:
        relation_msg_gate_hierarchy_mean = (
            relation_msg_gate_hierarchy_mass_sum / float(relation_msg_gate_hierarchy_mass_count)
        )
        relation_msg_gate_hierarchy_min = float(min(relation_msg_gate_hierarchy_mass_batch_means))
        relation_msg_gate_hierarchy_max = float(max(relation_msg_gate_hierarchy_mass_batch_means))
    else:
        relation_msg_gate_hierarchy_mean = avg('relation_msg_gate_hierarchy_mean')
        relation_msg_gate_hierarchy_min = avg('relation_msg_gate_hierarchy_min')
        relation_msg_gate_hierarchy_max = avg('relation_msg_gate_hierarchy_max')
    result = {
        'total_loss': avg('total_loss'),
        'pred_loss': avg('pred_loss'),
        'relation_msg_gate_l1': avg('relation_msg_gate_l1'),
        'protein_other_drop_ratio': avg('protein_other_drop_ratio'),
        'cross_edge_weight': avg('cross_edge_weight'),
        'token_valid_ratio': avg('token_valid_ratio'),
        'token_fallback_ratio': avg('token_fallback_ratio'),
        'fusion_alpha_mean': avg('fusion_alpha_mean'),
        'hier_flow_gate_drug_mean': avg('hier_flow_gate_drug_mean'),
        'hier_flow_gate_protein_mean': avg('hier_flow_gate_protein_mean'),
        'hier_flow_gate_drug_min': avg('hier_flow_gate_drug_min'),
        'hier_flow_gate_protein_min': avg('hier_flow_gate_protein_min'),
        'hier_flow_gate_drug_max': avg('hier_flow_gate_drug_max'),
        'hier_flow_gate_protein_max': avg('hier_flow_gate_protein_max'),
        'relation_msg_gate_mean': relation_msg_gate_mean,
        'relation_msg_gate_min': relation_msg_gate_min,
        'relation_msg_gate_max': relation_msg_gate_max,
        'relation_msg_gate_hierarchy_mean': relation_msg_gate_hierarchy_mean,
        'relation_msg_gate_hierarchy_min': relation_msg_gate_hierarchy_min,
        'relation_msg_gate_hierarchy_max': relation_msg_gate_hierarchy_max,
        'relation_msg_gate_value_map': relation_msg_gate_value_map,
        'preds': preds_np,
        'labels': labels_np,
    }
    return result


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    config,
    epoch=0,
    split='eval',
    amp_enabled=False,
    amp_dtype=torch.float16,
    force_cross_edge_weight: Optional[float] = None,
):
    """评估模型（atom-residue cross-edge 路径）"""
    if force_cross_edge_weight is not None:
        cross_edge_weight = force_cross_edge_weight
    else:
        cross_edge_weight = _compute_cross_edge_weight(config, epoch)
    train_cfg = config.get('train', {})
    eval_log_interval = max(1, int(train_cfg.get('eval_log_interval', train_cfg.get('log_interval', 20))))

    model.eval()
    sums = {
        'pred_loss': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_l1': torch.zeros((), device=device, dtype=torch.float32),
        'protein_other_drop_ratio': torch.zeros((), device=device, dtype=torch.float32),
        'cross_edge_weight': torch.zeros((), device=device, dtype=torch.float32),
        'token_valid_ratio': torch.zeros((), device=device, dtype=torch.float32),
        'token_fallback_ratio': torch.zeros((), device=device, dtype=torch.float32),
        'fusion_alpha_mean': torch.zeros((), device=device, dtype=torch.float32),
        'hier_flow_gate_drug_mean': torch.zeros((), device=device, dtype=torch.float32),
        'hier_flow_gate_protein_mean': torch.zeros((), device=device, dtype=torch.float32),
        'hier_flow_gate_drug_min': torch.zeros((), device=device, dtype=torch.float32),
        'hier_flow_gate_protein_min': torch.zeros((), device=device, dtype=torch.float32),
        'hier_flow_gate_drug_max': torch.zeros((), device=device, dtype=torch.float32),
        'hier_flow_gate_protein_max': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_mean': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_min': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_max': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_hierarchy_mean': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_hierarchy_min': torch.zeros((), device=device, dtype=torch.float32),
        'relation_msg_gate_hierarchy_max': torch.zeros((), device=device, dtype=torch.float32),
    }
    all_preds = []
    all_labels = []
    relation_msg_gate_value_sums = {}
    relation_msg_gate_value_counts = {}
    relation_msg_gate_hierarchy_mass_sum = 0.0
    relation_msg_gate_hierarchy_mass_count = 0
    relation_msg_gate_hierarchy_mass_batch_means = []
    n_batches = 0
    pin_memory_enabled = bool(getattr(dataloader, "pin_memory", False))

    with torch.inference_mode():
        pbar = tqdm(dataloader, desc=f'[{split}]')
        for batch_idx, (drug_graphs, protein_graphs, labels, _, _, typed_edge_batch) in enumerate(pbar):
            if drug_graphs is None or labels is None:
                continue

            n_batches += 1

            drug_graphs = drug_graphs.to(device, non_blocking=pin_memory_enabled)
            protein_graphs = protein_graphs.to(device, non_blocking=pin_memory_enabled)
            labels = labels.to(device, non_blocking=pin_memory_enabled)
            typed_edge_batch = _move_typed_edge_batch_to_device(
                typed_edge_batch,
                device=device,
            )

            amp_context = _build_amp_autocast(device, enabled=amp_enabled, dtype=amp_dtype)
            with amp_context:
                outputs = model(
                    drug_graphs,
                    protein_graphs,
                    typed_edge_batch=typed_edge_batch,
                    cross_edge_weight=cross_edge_weight,
                )
                logits = outputs['logits']
                pred_loss = criterion(logits, labels)

            ref_scalar = pred_loss.detach().float()
            sums['pred_loss'] += ref_scalar
            sums['relation_msg_gate_l1'] += _to_detached_scalar_tensor(outputs.get('relation_msg_gate_l1'), ref_scalar)
            sums['cross_edge_weight'] += _to_detached_scalar_tensor(outputs.get('cross_edge_weight'), ref_scalar)
            sums['token_valid_ratio'] += _to_detached_scalar_tensor(outputs.get('token_valid_ratio'), ref_scalar)
            sums['token_fallback_ratio'] += _to_detached_scalar_tensor(outputs.get('token_fallback_ratio'), ref_scalar)
            sums['fusion_alpha_mean'] += _to_detached_scalar_tensor(outputs.get('fusion_alpha_mean'), ref_scalar)
            sums['hier_flow_gate_drug_mean'] += _to_detached_scalar_tensor(outputs.get('hier_flow_gate_drug_mean'), ref_scalar)
            sums['hier_flow_gate_protein_mean'] += _to_detached_scalar_tensor(outputs.get('hier_flow_gate_protein_mean'), ref_scalar)
            sums['hier_flow_gate_drug_min'] += _to_detached_scalar_tensor(outputs.get('hier_flow_gate_drug_min'), ref_scalar)
            sums['hier_flow_gate_protein_min'] += _to_detached_scalar_tensor(outputs.get('hier_flow_gate_protein_min'), ref_scalar)
            sums['hier_flow_gate_drug_max'] += _to_detached_scalar_tensor(outputs.get('hier_flow_gate_drug_max'), ref_scalar)
            sums['hier_flow_gate_protein_max'] += _to_detached_scalar_tensor(outputs.get('hier_flow_gate_protein_max'), ref_scalar)
            sums['relation_msg_gate_mean'] += _to_detached_scalar_tensor(outputs.get('relation_msg_gate_mean'), ref_scalar)
            sums['relation_msg_gate_min'] += _to_detached_scalar_tensor(outputs.get('relation_msg_gate_min'), ref_scalar)
            sums['relation_msg_gate_max'] += _to_detached_scalar_tensor(outputs.get('relation_msg_gate_max'), ref_scalar)
            sums['relation_msg_gate_hierarchy_mean'] += _to_detached_scalar_tensor(outputs.get('relation_msg_gate_hierarchy_mean'), ref_scalar)
            sums['relation_msg_gate_hierarchy_min'] += _to_detached_scalar_tensor(outputs.get('relation_msg_gate_hierarchy_min'), ref_scalar)
            sums['relation_msg_gate_hierarchy_max'] += _to_detached_scalar_tensor(outputs.get('relation_msg_gate_hierarchy_max'), ref_scalar)
            sums['protein_other_drop_ratio'] += _to_detached_scalar_tensor(outputs.get('protein_other_drop_ratio'), ref_scalar)
            if hasattr(model, 'get_relation_msg_gate_value_stat_map'):
                stat_map = model.get_relation_msg_gate_value_stat_map()
                for edge_type, (budget_sum, count) in stat_map.items():
                    relation_msg_gate_value_sums[edge_type] = relation_msg_gate_value_sums.get(edge_type, 0.0) + float(budget_sum)
                    relation_msg_gate_value_counts[edge_type] = relation_msg_gate_value_counts.get(edge_type, 0) + int(count)
            if hasattr(model, 'get_relation_msg_gate_hierarchy_mass_stat'):
                hier_sum, hier_count = model.get_relation_msg_gate_hierarchy_mass_stat()
                relation_msg_gate_hierarchy_mass_sum += float(hier_sum)
                relation_msg_gate_hierarchy_mass_count += int(hier_count)
                if hier_count > 0:
                    relation_msg_gate_hierarchy_mass_batch_means.append(float(hier_sum) / float(hier_count))

            all_preds.append(torch.sigmoid(logits).detach())
            all_labels.append(labels.detach())

            if (batch_idx + 1) % eval_log_interval == 0:
                pbar.set_postfix({'pred': f'{float(pred_loss.detach().item()):.4f}'})

    denom = max(1, n_batches)
    inv_denom = 1.0 / float(denom)
    avg = lambda key: float((sums[key] * inv_denom).item())
    avg_pred_loss = avg('pred_loss')
    preds_np = torch.cat(all_preds, dim=0).float().cpu().numpy() if all_preds else np.array([])
    labels_np = torch.cat(all_labels, dim=0).float().cpu().numpy() if all_labels else np.array([])
    relation_msg_gate_value_map = {}
    for edge_type, total_budget in relation_msg_gate_value_sums.items():
        total_count = relation_msg_gate_value_counts.get(edge_type, 0)
        if total_count > 0:
            relation_msg_gate_value_map[edge_type] = total_budget / float(total_count)
    if relation_msg_gate_value_map:
        relation_value_list = list(relation_msg_gate_value_map.values())
        relation_msg_gate_mean = float(sum(relation_value_list) / len(relation_value_list))
        relation_msg_gate_min = float(min(relation_value_list))
        relation_msg_gate_max = float(max(relation_value_list))
    else:
        relation_msg_gate_mean = avg('relation_msg_gate_mean')
        relation_msg_gate_min = avg('relation_msg_gate_min')
        relation_msg_gate_max = avg('relation_msg_gate_max')
    if relation_msg_gate_hierarchy_mass_count > 0:
        relation_msg_gate_hierarchy_mean = (
            relation_msg_gate_hierarchy_mass_sum / float(relation_msg_gate_hierarchy_mass_count)
        )
        relation_msg_gate_hierarchy_min = float(min(relation_msg_gate_hierarchy_mass_batch_means))
        relation_msg_gate_hierarchy_max = float(max(relation_msg_gate_hierarchy_mass_batch_means))
    else:
        relation_msg_gate_hierarchy_mean = avg('relation_msg_gate_hierarchy_mean')
        relation_msg_gate_hierarchy_min = avg('relation_msg_gate_hierarchy_min')
        relation_msg_gate_hierarchy_max = avg('relation_msg_gate_hierarchy_max')
    result = {
        'total_loss': avg_pred_loss,
        'pred_loss': avg_pred_loss,
        'relation_msg_gate_l1': avg('relation_msg_gate_l1'),
        'protein_other_drop_ratio': avg('protein_other_drop_ratio'),
        'cross_edge_weight': avg('cross_edge_weight'),
        'token_valid_ratio': avg('token_valid_ratio'),
        'token_fallback_ratio': avg('token_fallback_ratio'),
        'fusion_alpha_mean': avg('fusion_alpha_mean'),
        'hier_flow_gate_drug_mean': avg('hier_flow_gate_drug_mean'),
        'hier_flow_gate_protein_mean': avg('hier_flow_gate_protein_mean'),
        'hier_flow_gate_drug_min': avg('hier_flow_gate_drug_min'),
        'hier_flow_gate_protein_min': avg('hier_flow_gate_protein_min'),
        'hier_flow_gate_drug_max': avg('hier_flow_gate_drug_max'),
        'hier_flow_gate_protein_max': avg('hier_flow_gate_protein_max'),
        'relation_msg_gate_mean': relation_msg_gate_mean,
        'relation_msg_gate_min': relation_msg_gate_min,
        'relation_msg_gate_max': relation_msg_gate_max,
        'relation_msg_gate_hierarchy_mean': relation_msg_gate_hierarchy_mean,
        'relation_msg_gate_hierarchy_min': relation_msg_gate_hierarchy_min,
        'relation_msg_gate_hierarchy_max': relation_msg_gate_hierarchy_max,
        'relation_msg_gate_value_map': relation_msg_gate_value_map,
        'preds': preds_np,
        'labels': labels_np,
    }
    return result


def train_single_dataset(dataset_name, dataset_config, base_config_path):
    """
    训练单个数据集

    参数:
        dataset_name: 数据集名称
        dataset_config: 数据集配置
        base_config_path: 基础配置文件路径
    """
    print(f"\n{'='*80}")
    print(f"开始训练数据集: {dataset_name}")
    print(f"{'='*80}")
    print(f"训练文件: {dataset_config['train_file']}")
    print(f"验证文件: {dataset_config['val_file']}")
    print(f"测试文件: {dataset_config['test_file']}")
    print(f"输出目录: {dataset_config['output_dir']}")

    # 加载基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 更新数据集路径
    config['data']['train_file'] = dataset_config['train_file']
    config['data']['val_file'] = dataset_config['val_file']
    config['data']['test_file'] = dataset_config['test_file']

    # 更新输出目录
    config['output']['output_dir'] = dataset_config['output_dir']
    config = _resolve_runtime_paths(config, base_config_path)

    # 创建输出目录
    os.makedirs(config['output']['output_dir'], exist_ok=True)

    # 保存当前训练配置到输出目录
    config_save_path = os.path.join(config['output']['output_dir'], 'config_snapshot.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info(f"训练配置已保存到: {config_save_path}")

    # 设置日志
    log_file = os.path.join(config['output']['output_dir'], config['output'].get('log_file', 'training.log'))
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # 设置设备
    device_config = config.get('device', {})
    use_cuda = device_config.get('use_cuda', True)
    cuda_device = device_config.get('cuda_device', 0)

    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_device}')
        logger.info(f"使用GPU: {cuda_device}")
    else:
        device = torch.device('cpu')
        logger.info("使用CPU")

    # 设置随机种子
    seed = config.get('seed', {}).get('seed', 42)
    deterministic = config.get('seed', {}).get('deterministic', True)
    benchmark = config.get('seed', {}).get('benchmark', False)
    set_seed(seed, deterministic, benchmark)
    logger.info(f"设置随机种子: {seed}")

    # 初始化模型
    model_config = config['model']
    model = HierHGTDTIModel.from_config(model_config, device=device)
    model = model.to(device)
    logger.info("预测头模式: %s", str(model_config.get('predictor', {}).get('mode', 'single_global')))

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")

    # 损失函数（仅 BCEWithLogits）
    loss_config = config.get('loss', {})
    loss_type = str(loss_config.get('type', 'bce')).lower()
    if loss_type not in {'bce', 'bce_with_logits'}:
        raise ValueError(
            f"Unsupported loss.type='{loss_type}'. Only 'bce' is supported."
        )
    criterion = nn.BCEWithLogitsLoss()

    train_config = config['train']
    num_epochs = int(train_config['epochs'])
    threshold = config['evaluation'].get('threshold', 0.5)

    # 优化器：只优化模型参数
    _suppress_known_dgl_amp_future_warnings()
    amp_enabled, amp_dtype, amp_use_scaler = _resolve_amp_settings(train_config, device)
    scaler = _build_grad_scaler(enabled=amp_use_scaler)
    logger.info(
        f"AMP: enabled={amp_enabled}, dtype={str(amp_dtype).replace('torch.', '')}, "
        f"grad_scaler={amp_use_scaler}"
    )
    wd = train_config.get('weight_decay', 1e-5)
    base_lr = float(train_config['lr'])
    rel_lr = float(train_config.get('rel_lr', base_lr))
    optimizer_groups = _build_optimizer_param_groups(
        model=model,
        base_lr=base_lr,
        rel_lr=rel_lr,
        weight_decay=wd,
    )
    if not optimizer_groups:
        raise RuntimeError("No trainable parameter groups found.")
    optimizer = optim.AdamW(optimizer_groups, lr=base_lr)

    # 学习率调度器
    lr_scheduler_type = str(train_config.get('lr_scheduler', 'reduce_on_plateau')).lower()
    scheduler_t_max = num_epochs
    scheduler = _create_scheduler(
        optimizer=optimizer,
        train_config=train_config,
        lr_scheduler_type=lr_scheduler_type,
        t_max_epochs=scheduler_t_max,
    )

    # 创建数据集和数据加载器
    data_config = config['data']
    cache_in_memory = data_config.get('cache_in_memory', True)
    
    train_dataset = HierHGTDTIDataset(
        csv_path=data_config['train_file'],
        drug_cache_dirs=data_config['drug_cache_dirs'],
        protein_cache_dirs=data_config['protein_cache_dirs'],
        smiles_col=data_config['smiles_col'],
        protein_col=data_config['protein_col'],
        label_col=data_config['label_col'],
        cache_in_memory=cache_in_memory,
    )
    precompute_etype_ids = bool(train_config.get('precompute_etype_ids', True))
    if not precompute_etype_ids:
        raise ValueError(
            "train.precompute_etype_ids must be true for typed-edge pipeline."
        )
    _maybe_precompute_dataset_etype_ids(model, train_dataset, split_name='train')
    _prepare_dataset_local_typed_edges(model, train_dataset, split_name='train')
    
    num_workers = int(data_config.get('num_workers', 0))
    pin_memory = bool(data_config.get('pin_memory', False))
    persistent_workers = bool(data_config.get('persistent_workers', num_workers > 0))
    prefetch_factor = max(1, int(data_config.get('prefetch_factor', 2)))

    loader_common_kwargs = {
        'batch_size': data_config['batch_size'],
        'collate_fn': hierhgt_dti_collate_fn,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        loader_common_kwargs['persistent_workers'] = persistent_workers
        loader_common_kwargs['prefetch_factor'] = prefetch_factor

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_common_kwargs,
    )

    val_dataloader = None
    if 'val_file' in data_config and data_config['val_file']:
        val_dataset = HierHGTDTIDataset(
            csv_path=data_config['val_file'],
            drug_cache_dirs=data_config['drug_cache_dirs'],
            protein_cache_dirs=data_config['protein_cache_dirs'],
            smiles_col=data_config['smiles_col'],
            protein_col=data_config['protein_col'],
            label_col=data_config['label_col'],
            cache_in_memory=cache_in_memory,
        )
        _maybe_precompute_dataset_etype_ids(model, val_dataset, split_name='val')
        _prepare_dataset_local_typed_edges(model, val_dataset, split_name='val')
        
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_common_kwargs,
        )

    test_dataloader = None
    if 'test_file' in data_config and data_config['test_file']:
        test_dataset = HierHGTDTIDataset(
            csv_path=data_config['test_file'],
            drug_cache_dirs=data_config['drug_cache_dirs'],
            protein_cache_dirs=data_config['protein_cache_dirs'],
            smiles_col=data_config['smiles_col'],
            protein_col=data_config['protein_col'],
            label_col=data_config['label_col'],
            cache_in_memory=cache_in_memory,
        )
        _maybe_precompute_dataset_etype_ids(model, test_dataset, split_name='test')
        _prepare_dataset_local_typed_edges(model, test_dataset, split_name='test')
        
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            **loader_common_kwargs,
        )

    # 训练历史
    history_keys = [
        'total_loss', 'pred_loss', 'cross_edge_weight',
        'protein_other_drop_ratio', 'token_valid_ratio', 'token_fallback_ratio', 'fusion_alpha_mean',
        'hier_flow_gate_drug_mean', 'hier_flow_gate_protein_mean',
        'relation_msg_gate_l1', 'relation_msg_gate_mean', 'relation_msg_gate_hierarchy_mean',
        'auc_roc', 'auc_pr', 'accuracy', 'precision', 'recall', 'f1',
    ]
    train_history = {k: [] for k in history_keys}
    val_history = {k: [] for k in history_keys}

    # 训练循环
    best_auc_pr = 0.0
    best_model_path = os.path.join(config['output']['output_dir'], 'best_model.pth')
    early_stopping_counter = 0
    early_stopping_patience = train_config.get('early_stopping_patience', 10)
    early_stopping_delta = train_config.get('early_stopping_delta', 0.001)

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{num_epochs} [atom_residue_cross_edge]")
        logger.info(f"{'='*50}")

        train_out = train_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion,
            device,
            config,
            epoch,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        train_metrics = compute_metrics(train_out['preds'], train_out['labels'], threshold)

        train_history['total_loss'].append(train_out['total_loss'])
        train_history['pred_loss'].append(train_out['pred_loss'])
        train_history['cross_edge_weight'].append(train_out['cross_edge_weight'])
        train_history['protein_other_drop_ratio'].append(train_out['protein_other_drop_ratio'])
        train_history['token_valid_ratio'].append(train_out.get('token_valid_ratio', 0.0))
        train_history['token_fallback_ratio'].append(train_out.get('token_fallback_ratio', 0.0))
        train_history['fusion_alpha_mean'].append(train_out.get('fusion_alpha_mean', 0.0))
        train_history['hier_flow_gate_drug_mean'].append(train_out.get('hier_flow_gate_drug_mean', 0.0))
        train_history['hier_flow_gate_protein_mean'].append(train_out.get('hier_flow_gate_protein_mean', 0.0))
        train_history['relation_msg_gate_l1'].append(train_out.get('relation_msg_gate_l1', 0.0))
        train_history['relation_msg_gate_mean'].append(train_out.get('relation_msg_gate_mean', 0.0))
        train_history['relation_msg_gate_hierarchy_mean'].append(train_out.get('relation_msg_gate_hierarchy_mean', 0.0))
        train_history['auc_roc'].append(train_metrics['auc_roc'])
        train_history['auc_pr'].append(train_metrics['auc_pr'])
        train_history['accuracy'].append(train_metrics['accuracy'])
        train_history['precision'].append(train_metrics['precision'])
        train_history['recall'].append(train_metrics['recall'])
        train_history['f1'].append(train_metrics['f1'])

        relation_msg_gate_l1_val = train_out.get('relation_msg_gate_l1', 0.0)
        logger.info(
            "Train Loss: total=%.4f pred=%.4f rel_gate_l1=%.4f cross_edge_w=%.4f",
            train_out['total_loss'],
            train_out['pred_loss'],
            relation_msg_gate_l1_val,
            train_out['cross_edge_weight'],
        )
        logger.info(
            "Train AUC-ROC/AUC-PR: %.4f / %.4f",
            train_metrics['auc_roc'],
            train_metrics['auc_pr'],
        )
        logger.info(
            "Train predictor stats: token_valid=%.4f fallback=%.4f fusion_alpha=%.4f hier_gate(drug/protein)=%.4f/%.4f rel_gate(avg/hier_mass)=%.4f/%.4f",
            train_out.get('token_valid_ratio', 0.0),
            train_out.get('token_fallback_ratio', 0.0),
            train_out.get('fusion_alpha_mean', 0.0),
            train_out.get('hier_flow_gate_drug_mean', 0.0),
            train_out.get('hier_flow_gate_protein_mean', 0.0),
            train_out.get('relation_msg_gate_mean', 0.0),
            train_out.get('relation_msg_gate_hierarchy_mean', 0.0),
        )

        if val_dataloader:
            val_out = evaluate(
                model,
                val_dataloader,
                criterion,
                device,
                config,
                epoch,
                split='val',
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            val_metrics = compute_metrics(val_out['preds'], val_out['labels'], threshold)

            val_history['total_loss'].append(val_out['total_loss'])
            val_history['pred_loss'].append(val_out['pred_loss'])
            val_history['cross_edge_weight'].append(val_out['cross_edge_weight'])
            val_history['protein_other_drop_ratio'].append(val_out['protein_other_drop_ratio'])
            val_history['token_valid_ratio'].append(val_out.get('token_valid_ratio', 0.0))
            val_history['token_fallback_ratio'].append(val_out.get('token_fallback_ratio', 0.0))
            val_history['fusion_alpha_mean'].append(val_out.get('fusion_alpha_mean', 0.0))
            val_history['hier_flow_gate_drug_mean'].append(val_out.get('hier_flow_gate_drug_mean', 0.0))
            val_history['hier_flow_gate_protein_mean'].append(val_out.get('hier_flow_gate_protein_mean', 0.0))
            val_history['relation_msg_gate_l1'].append(val_out.get('relation_msg_gate_l1', 0.0))
            val_history['relation_msg_gate_mean'].append(val_out.get('relation_msg_gate_mean', 0.0))
            val_history['relation_msg_gate_hierarchy_mean'].append(val_out.get('relation_msg_gate_hierarchy_mean', 0.0))
            val_history['auc_roc'].append(val_metrics['auc_roc'])
            val_history['auc_pr'].append(val_metrics['auc_pr'])
            val_history['accuracy'].append(val_metrics['accuracy'])
            val_history['precision'].append(val_metrics['precision'])
            val_history['recall'].append(val_metrics['recall'])
            val_history['f1'].append(val_metrics['f1'])

            logger.info(
                "Val Loss: pred=%.4f",
                val_out['pred_loss'],
            )
            logger.info(
                "Val AUC-ROC/AUC-PR: %.4f / %.4f",
                val_metrics['auc_roc'],
                val_metrics['auc_pr'],
            )
            logger.info(
                "Val predictor stats: token_valid=%.4f fallback=%.4f fusion_alpha=%.4f hier_gate(drug/protein)=%.4f/%.4f rel_gate(avg/hier_mass)=%.4f/%.4f",
                val_out.get('token_valid_ratio', 0.0),
                val_out.get('token_fallback_ratio', 0.0),
                val_out.get('fusion_alpha_mean', 0.0),
                val_out.get('hier_flow_gate_drug_mean', 0.0),
                val_out.get('hier_flow_gate_protein_mean', 0.0),
                val_out.get('relation_msg_gate_mean', 0.0),
                val_out.get('relation_msg_gate_hierarchy_mean', 0.0),
            )

            val_auc_pr = val_metrics['auc_pr']
            score_for_sched = -1e9 if np.isnan(val_auc_pr) else val_auc_pr

            if scheduler is not None and lr_scheduler_type == 'reduce_on_plateau':
                scheduler.step(score_for_sched)
            elif scheduler is not None and lr_scheduler_type == 'cosine':
                scheduler.step()

            if (not np.isnan(val_auc_pr)) and (val_auc_pr > best_auc_pr + early_stopping_delta):
                best_auc_pr = val_auc_pr
                early_stopping_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
                    'train_loss': train_out['total_loss'],
                    'val_loss': val_out['pred_loss'],
                    'val_auc_pr': val_auc_pr,
                    'config': config
                }, best_model_path)
                logger.info(f"保存最佳模型: AUC-PR = {best_auc_pr:.4f}")
            else:
                early_stopping_counter += 1
                logger.info(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")

            if early_stopping_counter >= early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

    # 保存训练历史
    history_path = os.path.join(config['output']['output_dir'], 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train': train_history,
            'val': val_history,
            'best_auc_pr': best_auc_pr
        }, f, indent=4)

    logger.info(f"\n训练完成！最佳AUC-PR: {best_auc_pr:.4f}")
    logger.info(f"训练历史保存到: {history_path}")
    logger.info(f"最佳模型保存到: {best_model_path}")

    # 在测试集上评估最佳模型
    if test_dataloader and os.path.exists(best_model_path):
        logger.info(f"\n{'='*50}")
        logger.info("Evaluating best model on test set")
        logger.info(f"{'='*50}")

        checkpoint = torch.load(best_model_path, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError as e:
            logger.warning("strict=True loading failed (%s); falling back to strict=False", e)
            load_state = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if getattr(load_state, "missing_keys", None):
                logger.warning("Checkpoint missing keys: %s", load_state.missing_keys)
            if getattr(load_state, "unexpected_keys", None):
                logger.warning("Checkpoint unexpected keys: %s", load_state.unexpected_keys)
        ckpt_epoch = checkpoint.get('epoch', num_epochs)
        try:
            test_eval_epoch = max(1, int(ckpt_epoch))
        except (TypeError, ValueError):
            test_eval_epoch = max(1, int(num_epochs))
        logger.info("Test evaluation warmup epoch: %d", test_eval_epoch)

        test_out = evaluate(
            model,
            test_dataloader,
            criterion,
            device,
            config,
            epoch=test_eval_epoch,
            split='test',
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            force_cross_edge_weight=1.0,
        )
        test_metrics = compute_metrics(test_out['preds'], test_out['labels'], threshold)

        logger.info(f"Test Loss: {test_out['pred_loss']:.4f}")
        logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}, AUC-PR: {test_metrics['auc_pr']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
        logger.info(
            "Test hierarchy gate: drug=%.4f [%.4f, %.4f], protein=%.4f [%.4f, %.4f]",
            test_out.get('hier_flow_gate_drug_mean', 0.0),
            test_out.get('hier_flow_gate_drug_min', 0.0),
            test_out.get('hier_flow_gate_drug_max', 0.0),
            test_out.get('hier_flow_gate_protein_mean', 0.0),
            test_out.get('hier_flow_gate_protein_min', 0.0),
            test_out.get('hier_flow_gate_protein_max', 0.0),
        )
        logger.info(
            "Test relation gate: avg=%.4f [%.4f, %.4f], hierarchy_mass=%.4f [%.4f, %.4f]",
            test_out.get('relation_msg_gate_mean', 0.0),
            test_out.get('relation_msg_gate_min', 0.0),
            test_out.get('relation_msg_gate_max', 0.0),
            test_out.get('relation_msg_gate_hierarchy_mean', 0.0),
            test_out.get('relation_msg_gate_hierarchy_min', 0.0),
            test_out.get('relation_msg_gate_hierarchy_max', 0.0),
        )

        # 保存测试结果
        test_results_path = os.path.join(config['output']['output_dir'], 'test_results.json')
        relation_msg_gate_values = test_out.get('relation_msg_gate_value_map', {})
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'test_total_loss': test_out['total_loss'],
                'test_pred_loss': test_out['pred_loss'],
                'test_metrics': {
                    'auc_roc': test_metrics['auc_roc'],
                    'auc_pr': test_metrics['auc_pr'],
                    'accuracy': test_metrics['accuracy'],
                    'precision': test_metrics['precision'],
                    'recall': test_metrics['recall'],
                    'f1': test_metrics['f1']
                },
                'debug': {
                    'cross_edge_weight': test_out['cross_edge_weight'],
                    'protein_other_drop_ratio': test_out['protein_other_drop_ratio'],
                    'predictor_mode': str(config.get('model', {}).get('predictor', {}).get('mode', 'single_global')),
                    'token_valid_ratio': test_out.get('token_valid_ratio', 0.0),
                    'token_fallback_ratio': test_out.get('token_fallback_ratio', 0.0),
                    'fusion_alpha_mean': test_out.get('fusion_alpha_mean', 0.0),
                    'hier_flow_gate_drug_mean': test_out.get('hier_flow_gate_drug_mean', 0.0),
                    'hier_flow_gate_protein_mean': test_out.get('hier_flow_gate_protein_mean', 0.0),
                    'hier_flow_gate_drug_min': test_out.get('hier_flow_gate_drug_min', 0.0),
                    'hier_flow_gate_protein_min': test_out.get('hier_flow_gate_protein_min', 0.0),
                    'hier_flow_gate_drug_max': test_out.get('hier_flow_gate_drug_max', 0.0),
                    'hier_flow_gate_protein_max': test_out.get('hier_flow_gate_protein_max', 0.0),
                    'relation_msg_gate_l1': test_out.get('relation_msg_gate_l1', 0.0),
                    'relation_msg_gate_mean': test_out.get('relation_msg_gate_mean', 0.0),
                    'relation_msg_gate_min': test_out.get('relation_msg_gate_min', 0.0),
                    'relation_msg_gate_max': test_out.get('relation_msg_gate_max', 0.0),
                    'relation_msg_gate_hierarchy_mean': test_out.get('relation_msg_gate_hierarchy_mean', 0.0),
                    'relation_msg_gate_hierarchy_min': test_out.get('relation_msg_gate_hierarchy_min', 0.0),
                    'relation_msg_gate_hierarchy_max': test_out.get('relation_msg_gate_hierarchy_max', 0.0),
                    'relation_msg_gate_values': relation_msg_gate_values,
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=4)
        logger.info(f"测试结果保存到: {test_results_path}")

    # 移除日志处理器
    logger.removeHandler(file_handler)
    file_handler.close()

    print(f"\nDataset {dataset_name} training finished.")


# 所有数据集配置

def _make_dataset_config(dataset_dir, split):
    dataset_key = f"{dataset_dir.lower()}_{split}"
    return {
        'name': dataset_key,
        'train_file': os.path.join(DATA_DIR, dataset_dir, split, 'train.csv'),
        'val_file': os.path.join(DATA_DIR, dataset_dir, split, 'val.csv'),
        'test_file': os.path.join(DATA_DIR, dataset_dir, split, 'test.csv'),
        'output_dir': os.path.join(DEFAULT_OUTPUT_ROOT, dataset_key),
    }


DATASET_CONFIGS = {
    'drugbank_random': _make_dataset_config('DrugBank', 'random'),
    'drugbank_cold_drug': _make_dataset_config('DrugBank', 'cold_drug'),
    'drugbank_cold_protein': _make_dataset_config('DrugBank', 'cold_protein'),
    'biosnap_random': _make_dataset_config('BioSnap', 'random'),
    'biosnap_cold_drug': _make_dataset_config('BioSnap', 'cold_drug'),
    'biosnap_cold_protein': _make_dataset_config('BioSnap', 'cold_protein'),
}



def _build_dataset_override_from_base_config(base_config_path):
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_runtime_paths(cfg, base_config_path)
    data_cfg = cfg.get("data", {})
    output_cfg = cfg.get("output", {})
    return {
        "name": "single",
        "train_file": data_cfg.get("train_file"),
        "val_file": data_cfg.get("val_file"),
        "test_file": data_cfg.get("test_file"),
        "output_dir": output_cfg.get("output_dir"),
    }


def train_all_datasets(base_config_path, dataset_names=None, resume_from=None, dataset_configs=None):
    dataset_configs = dataset_configs or DATASET_CONFIGS
    if dataset_names is None:
        selected = list(dataset_configs.keys())
    else:
        selected = list(dataset_names)

    invalid = [name for name in selected if name not in dataset_configs]
    if invalid:
        raise ValueError(f"Unknown datasets: {invalid}. Available: {sorted(dataset_configs.keys())}")

    if resume_from is not None:
        if resume_from not in selected:
            raise ValueError(f"resume_from '{resume_from}' not found in selected datasets: {selected}")
        selected = selected[selected.index(resume_from):]
        print(f"Resume training from dataset: {resume_from}")

    print(f"\n{'='*80}")
    print(f"Preparing to train {len(selected)} datasets")
    print(f"{'='*80}")

    for i, dataset_name in enumerate(selected):
        print(f"\n{'='*80}")
        print(f"[{i + 1}/{len(selected)}] Start dataset: {dataset_name}")
        print(f"{'='*80}")

        dataset_config = dataset_configs[dataset_name]
        try:
            train_single_dataset(dataset_name, dataset_config, base_config_path)
            print(f"\nDataset {dataset_name} finished successfully.")
        except Exception as e:
            print(f"\nDataset {dataset_name} failed: {str(e)}")
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("All dataset runs finished.")
    print(f"{'='*80}")


class HierHGTDTITrainer:
    def __init__(self, base_config_path, dataset_configs=None):
        self.base_config_path = base_config_path
        self.dataset_configs = dataset_configs or DATASET_CONFIGS

    def list_datasets(self):
        return sorted(self.dataset_configs.keys())

    def run_single(self, dataset_name=None):
        if dataset_name is None:
            dataset_override = _build_dataset_override_from_base_config(self.base_config_path)
            train_single_dataset("single", dataset_override, self.base_config_path)
            return

        if dataset_name not in self.dataset_configs:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available: {sorted(self.dataset_configs.keys())}"
            )
        train_single_dataset(dataset_name, self.dataset_configs[dataset_name], self.base_config_path)

    def run_all(self, dataset_names=None, resume_from=None):
        train_all_datasets(
            self.base_config_path,
            dataset_names=dataset_names,
            resume_from=resume_from,
            dataset_configs=self.dataset_configs,
        )

def main():
    parser = argparse.ArgumentParser(description="Unified HierHGT-DTI training entry")
    parser.add_argument("--config", type=str, default="config_hierhgt_dti.yaml", help="Base config path")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "all"],
        default="single",
        help="single: train one dataset/config; all: train a dataset list",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset key in built-in DATASET_CONFIGS for single mode",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Dataset keys for all mode; default is all built-in keys",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume key for all mode",
    )
    parser.add_argument(
        "--list_datasets",
        action="store_true",
        help="Print available built-in dataset keys",
    )
    args = parser.parse_args()

    trainer = HierHGTDTITrainer(args.config)
    if args.list_datasets:
        print("Available datasets:")
        for name in trainer.list_datasets():
            print(f"  - {name}")
        return

    mode = args.mode
    # Backward compatibility: old multi-dataset commands often only pass --data/--resume_from.
    if mode == "single" and (args.datasets is not None or args.resume_from is not None):
        mode = "all"

    if mode == "single":
        trainer.run_single(dataset_name=args.dataset)
    else:
        trainer.run_all(dataset_names=args.datasets, resume_from=args.resume_from)


if __name__ == "__main__":
    main()

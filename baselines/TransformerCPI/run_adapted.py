#!/usr/bin/env python3
"""Benchmark-compatible TransformerCPI runner for HierHGT-DTI."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

TRANSFORMERCPI_ROOT = Path(__file__).resolve().parent
os.chdir(TRANSFORMERCPI_ROOT)
COMPARE_DIR = TRANSFORMERCPI_ROOT.parent
if str(COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(COMPARE_DIR))
if str(TRANSFORMERCPI_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERCPI_ROOT))

from common_metrics import MaxMetricEarlyStopper, classification_metrics, select_threshold_by_f1
from model import (
    Decoder,
    DecoderLayer,
    Encoder,
    PositionwiseFeedforward,
    Predictor,
    SelfAttention,
    Trainer,
    pack,
)


SUPPORTED_SPLITS = ("random", "cold_drug", "cold_protein")


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def to_tensor_dataset(records, drug_cache, protein_cache):
    dataset = []
    for smiles, sequence, label in records:
        atom_feat, adj = drug_cache[smiles]
        atom_tensor = torch.as_tensor(atom_feat, dtype=torch.float32)
        atom_len = int(atom_tensor.shape[0])
        adj_tensor = torch.as_tensor(adj, dtype=torch.float32)
        adj_tensor = adj_tensor + torch.eye(atom_len, dtype=adj_tensor.dtype)
        protein_tensor = torch.as_tensor(protein_cache[sequence], dtype=torch.float32)
        protein_len = int(protein_tensor.shape[0])
        dataset.append(
            (
                atom_tensor,
                adj_tensor,
                protein_tensor,
                torch.tensor(int(label), dtype=torch.long),
                atom_len,
                protein_len,
            )
        )
    return dataset


def build_eval_batches(dataset, eval_batch_size):
    return [dataset[start : start + eval_batch_size] for start in range(0, len(dataset), eval_batch_size)]


def evaluate(eval_batches, model, device):
    model.eval()
    labels, scores = [], []
    with torch.inference_mode():
        for batch in tqdm(eval_batches, total=len(eval_batches), desc="Eval", leave=False):
            packed = pack(batch, device)
            batch_true, _, batch_scores = model.predict(packed)
            labels.append(batch_true.detach().cpu())
            scores.append(batch_scores.detach().cpu())
    return (
        torch.cat(labels).numpy().astype(np.int64, copy=False),
        torch.cat(scores).numpy().astype(np.float64, copy=False),
    )


def build_model(device):
    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    kernel_size = 5

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(
        atom_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        DecoderLayer,
        SelfAttention,
        PositionwiseFeedforward,
        dropout,
        device,
    )
    return Predictor(encoder, decoder, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="TransformerCPI adapted runner")
    parser.add_argument("--dataset", required=True, choices=["BioSnap", "DrugBank"])
    parser.add_argument("--split", required=True, choices=SUPPORTED_SPLITS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64, help="Optimizer step interval in samples.")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_train", type=int, default=None, help="Optional smoke-test cap.")
    parser.add_argument("--max_val", type=int, default=None, help="Optional smoke-test cap.")
    parser.add_argument("--max_test", type=int, default=None, help="Optional smoke-test cap.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    data_dir = (
        Path(args.data_dir)
        if args.data_dir is not None
        else TRANSFORMERCPI_ROOT / "data_adapted" / f"{args.dataset}_{args.split}"
    )
    if args.output_dir is None:
        output_dir = TRANSFORMERCPI_ROOT / "output_adapted" / f"{args.dataset}_{args.split}" / f"seed{args.seed}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cache = load_pickle(data_dir / "feature_cache.pkl")
    train_records = load_pickle(data_dir / "train_records.pkl")
    val_records = load_pickle(data_dir / "val_records.pkl")
    test_records = load_pickle(data_dir / "test_records.pkl")

    if args.max_train is not None:
        train_records = train_records[: args.max_train]
    if args.max_val is not None:
        val_records = val_records[: args.max_val]
    if args.max_test is not None:
        test_records = test_records[: args.max_test]

    train_dataset = to_tensor_dataset(train_records, feature_cache["drug_cache"], feature_cache["protein_cache"])
    val_dataset = to_tensor_dataset(val_records, feature_cache["drug_cache"], feature_cache["protein_cache"])
    test_dataset = to_tensor_dataset(test_records, feature_cache["drug_cache"], feature_cache["protein_cache"])
    val_batches = build_eval_batches(val_dataset, args.eval_batch_size)
    test_batches = build_eval_batches(test_dataset, args.eval_batch_size)

    print(f"Dataset: {args.dataset}/{args.split}")
    print(f"  train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    print(f"  device={device}")

    model = build_model(device).to(device)
    trainer = Trainer(model, args.lr, args.weight_decay, args.batch_size)
    stopper = MaxMetricEarlyStopper(patience=args.patience, min_delta=args.min_delta)

    best_threshold = 0.5
    best_val_metrics = None
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = float(trainer.train(train_dataset, device))
        val_labels, val_scores = evaluate(val_batches, model, device)
        val_threshold = select_threshold_by_f1(val_labels, val_scores)
        val_metrics = classification_metrics(val_labels, val_scores, val_threshold)

        stopper.update(
            val_metrics["aupr"],
            epoch,
            model,
            payload={
                "threshold": float(val_threshold),
                "metrics": val_metrics,
            },
        )
        if stopper.best_payload is not None:
            best_threshold = float(stopper.best_payload["threshold"])
            best_val_metrics = stopper.best_payload["metrics"]

        print(
            f"[{epoch:3d}/{args.epochs}] train_loss={train_loss:.4f} "
            f"val_AUC={val_metrics['auc']:.4f} val_AUPR={val_metrics['aupr']:.4f} "
            f"val_F1={val_metrics['f1']:.4f} val_thr={val_threshold:.4f} "
            f"(best_epoch={stopper.best_epoch}, best_AUPR={stopper.best_metric:.4f})"
        )

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    if stopper.best_state is None:
        raise RuntimeError("Training finished without a tracked best checkpoint.")

    model.load_state_dict(stopper.best_state)
    ckpt_path = output_dir / "model.pt"
    torch.save(stopper.best_state, ckpt_path)

    test_labels, test_scores = evaluate(test_batches, model, device)
    test_metrics = classification_metrics(test_labels, test_scores, best_threshold)
    elapsed = time.time() - train_start

    result = {
        "model": "TransformerCPI",
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "auc": round(float(test_metrics["auc"]), 4),
        "aupr": round(float(test_metrics["aupr"]), 4),
        "f1": round(float(test_metrics["f1"]), 4),
        "acc": round(float(test_metrics["acc"]), 4),
        "precision": round(float(test_metrics["precision"]), 4),
        "recall": round(float(test_metrics["recall"]), 4),
        "specificity": round(float(test_metrics["specificity"]), 4),
        "best_epoch": stopper.best_epoch,
        "early_stop_epoch": stopper.stop_epoch,
        "selection_metric": "val_aupr",
        "early_stop_min_delta": args.min_delta,
        "threshold": round(float(best_threshold), 6),
        "threshold_policy": "val_f1_optimal",
        "optimizer_step_batch_size": args.batch_size,
        "upstream_micro_batch_size": 8,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs_requested": args.epochs,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "runtime_sec": round(float(elapsed), 2),
        "val_metrics_at_best": {
            key: round(float(value), 4)
            for key, value in (best_val_metrics or {}).items()
            if key != "threshold"
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result_path = output_dir / "test_results.json"
    result_path.write_text(json.dumps(result, indent=4), encoding="utf-8")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()

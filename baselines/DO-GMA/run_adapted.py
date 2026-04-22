#!/usr/bin/env python
"""Benchmark-compatible DO-GMA runner for HierHGT-DTI fixed splits."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

DOGMA_ROOT = os.path.dirname(os.path.abspath(__file__))
COMPARE_ROOT = os.path.abspath(os.path.join(DOGMA_ROOT, ".."))
if COMPARE_ROOT not in sys.path:
    sys.path.insert(0, COMPARE_ROOT)

from common_metrics import MaxMetricEarlyStopper, classification_metrics, select_threshold_by_f1
from data_utils import CachedDTIDataset, build_cache, graph_collate_func, split_cache_paths
from model import DOGMAModel


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device, criterion):
    model.eval()
    losses = []
    scores = []
    labels = []
    with torch.inference_mode():
        for drug_graph, drug_tokens, protein_tokens, batch_labels in loader:
            drug_graph = drug_graph.to(device)
            drug_tokens = drug_tokens.to(device)
            protein_tokens = protein_tokens.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(drug_graph, drug_tokens, protein_tokens)
            loss = criterion(logits, batch_labels)
            losses.append(float(loss.item()))
            scores.append(torch.sigmoid(logits).detach().cpu())
            labels.append(batch_labels.detach().cpu())

    labels = torch.cat(labels, dim=0).numpy()
    scores = torch.cat(scores, dim=0).numpy()
    loss = float(np.mean(losses)) if losses else 0.0
    return labels, scores, loss


def main() -> None:
    parser = argparse.ArgumentParser(description="DO-GMA adapted runner")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=["random", "cold_drug", "cold_protein"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--data_root",
        default=os.path.abspath(os.path.join(DOGMA_ROOT, "..", "..", "datasets")),
        help="Root directory containing <dataset>/<split>/train,val,test.csv",
    )
    parser.add_argument(
        "--cache_root",
        default=os.path.join(DOGMA_ROOT, "data_adapted"),
        help="Cache root created by prepare_adapted.py",
    )
    parser.add_argument("--max_protein_len", type=int, default=1200)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    cache_paths = split_cache_paths(args.cache_root, args.dataset, args.split)
    if not os.path.exists(cache_paths.cache_pt):
        print(f"[INFO] Cache not found at {cache_paths.cache_pt}, building it now...")
        build_cache(
            dataset=args.dataset,
            split=args.split,
            data_root=args.data_root,
            output_root=args.cache_root,
            max_protein_len=args.max_protein_len,
        )

    payload = torch.load(cache_paths.cache_pt, weights_only=False)
    metadata = payload["metadata"]

    train_dataset = CachedDTIDataset(payload, "train")
    val_dataset = CachedDTIDataset(payload, "val")
    test_dataset = CachedDTIDataset(payload, "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=graph_collate_func,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=graph_collate_func,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=graph_collate_func,
    )

    if args.output_dir is None:
        args.output_dir = os.path.join(DOGMA_ROOT, "output_adapted", f"{args.dataset}_{args.split}", f"seed{args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)

    model = DOGMAModel(
        max_drug_nodes=int(metadata["max_drug_nodes"]),
        protein_grid_height=int(metadata["protein_grid_height"]),
        drug_grid_width=int(metadata["drug_sequence_width"]),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    stopper = MaxMetricEarlyStopper(patience=args.patience)

    training_history = []
    best_threshold = 0.5
    start_time = time.time()

    print(f"Dataset: {args.dataset}_{args.split}")
    print(
        f"  Shapes: max_drug_nodes={metadata['max_drug_nodes']} "
        f"max_drug_seq_len={metadata['max_drug_seq_len']} max_protein_len={metadata['max_protein_len']}"
    )
    print(
        f"  Entities: drugs={metadata['num_drugs']} proteins={metadata['num_proteins']} "
        f"truncated_proteins={metadata['num_truncated_proteins']}"
    )
    print(
        f"  Labels: train={payload['splits']['train']['size']} "
        f"val={payload['splits']['val']['size']} test={payload['splits']['test']['size']}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for drug_graph, drug_tokens, protein_tokens, labels in train_loader:
            drug_graph = drug_graph.to(device)
            drug_tokens = drug_tokens.to(device)
            protein_tokens = protein_tokens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(drug_graph, drug_tokens, protein_tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_labels, val_scores, val_loss = evaluate(model, val_loader, device, criterion)
        val_threshold = select_threshold_by_f1(val_labels, val_scores)
        val_metrics = classification_metrics(val_labels, val_scores, val_threshold)
        improved = stopper.update(
            val_metrics["aupr"],
            epoch,
            model,
            payload={
                "threshold": float(val_threshold),
                "val_metrics": val_metrics,
                "val_loss": val_loss,
            },
        )
        if improved:
            best_threshold = float(val_threshold)

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(float(val_loss), 6),
                "val_auc": round(float(val_metrics["auc"]), 4),
                "val_aupr": round(float(val_metrics["aupr"]), 4),
                "val_f1": round(float(val_metrics["f1"]), 4),
                "val_acc": round(float(val_metrics["acc"]), 4),
                "val_threshold": round(float(val_threshold), 6),
            }
        )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[{epoch:3d}/{args.epochs}] train_loss={train_loss:.5f} "
                f"val_AUC={val_metrics['auc']:.4f} val_AUPR={val_metrics['aupr']:.4f} "
                f"val_thr={val_threshold:.4f}"
            )

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    elapsed_minutes = (time.time() - start_time) / 60.0
    test_labels, test_scores, test_loss = evaluate(model, test_loader, device, criterion)
    test_metrics = classification_metrics(test_labels, test_scores, best_threshold)

    with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    result = {
        "model": "DO-GMA",
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "auc": round(float(test_metrics["auc"]), 4),
        "aupr": round(float(test_metrics["aupr"]), 4),
        "f1": round(float(test_metrics["f1"]), 4),
        "acc": round(float(test_metrics["acc"]), 4),
        "precision": round(float(test_metrics["precision"]), 4),
        "recall": round(float(test_metrics["recall"]), 4),
        "specificity": round(float(test_metrics["specificity"]), 4),
        "test_loss": round(float(test_loss), 6),
        "best_epoch": stopper.best_epoch,
        "early_stop_epoch": stopper.stop_epoch,
        "selection_metric": "val_aupr",
        "threshold": round(float(best_threshold), 6),
        "threshold_policy": "val_f1_optimal",
        "elapsed_minutes": round(float(elapsed_minutes), 4),
        "metadata": metadata,
        "val_metrics_at_best": {
            key: round(float(value), 4)
            for key, value in (stopper.best_payload or {}).get("val_metrics", {}).items()
            if key != "threshold"
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result_path = os.path.join(args.output_dir, "test_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Test Results ===")
    print(f"  AUC:  {test_metrics['auc']:.4f}")
    print(f"  AUPR: {test_metrics['aupr']:.4f}")
    print(f"  F1:   {test_metrics['f1']:.4f}")
    print(f"  ACC:  {test_metrics['acc']:.4f}")
    print(f"  Time: {elapsed_minutes:.3f} min")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()

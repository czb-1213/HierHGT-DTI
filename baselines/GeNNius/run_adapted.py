#!/usr/bin/env python
"""Benchmark-compatible GeNNius runner for HierHGT-DTI fixed splits."""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch_geometric.transforms as T

GENNIUS_ROOT = os.path.dirname(os.path.abspath(__file__))
COMPARE_ROOT = os.path.abspath(os.path.join(GENNIUS_ROOT, ".."))
if COMPARE_ROOT not in sys.path:
    sys.path.insert(0, COMPARE_ROOT)

from common_metrics import MaxMetricEarlyStopper, classification_metrics, select_threshold_by_f1
from data_utils import build_cache, split_cache_paths
from model import GeNNiusModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, data, edge_label_index, edge_label):
    model.eval()
    with torch.inference_mode():
        _, logits = model(data.x_dict, data.edge_index_dict, edge_label_index)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, edge_label)
        scores = logits.sigmoid().detach().cpu().numpy()
        labels = edge_label.detach().cpu().numpy()
    return labels, scores, float(loss.item())


def main():
    parser = argparse.ArgumentParser(description="GeNNius adapted runner")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=["random", "cold_drug", "cold_protein"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=17)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--data_root",
        default=os.path.abspath(os.path.join(GENNIUS_ROOT, "..", "..", "datasets")),
        help="Root directory containing <dataset>/<split>/train,val,test.csv",
    )
    parser.add_argument(
        "--cache_root",
        default=os.path.join(GENNIUS_ROOT, "data_adapted"),
        help="Cache root created by prepare_adapted.py",
    )
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
        )

    payload = torch.load(cache_paths.cache_pt, weights_only=False)
    data = payload["data"]
    split_payload = payload["splits"]
    metadata = payload["metadata"]

    data = T.ToUndirected()(data)
    data = data.to(device)
    train_edge_label_index = split_payload["train"]["edge_label_index"].to(device)
    train_edge_label = split_payload["train"]["edge_label"].to(device)
    val_edge_label_index = split_payload["val"]["edge_label_index"].to(device)
    val_edge_label = split_payload["val"]["edge_label"].to(device)
    test_edge_label_index = split_payload["test"]["edge_label_index"].to(device)
    test_edge_label = split_payload["test"]["edge_label"].to(device)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            GENNIUS_ROOT, "output_adapted", f"{args.dataset}_{args.split}", f"seed{args.seed}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Dataset: {args.dataset}_{args.split}")
    print(
        f"  Nodes: drugs={metadata['num_drugs']} proteins={metadata['num_proteins']} "
        f"train_pos_edges={metadata['num_train_pos_edges']}"
    )
    print(
        f"  Labels: train={split_payload['train']['size']} "
        f"val={split_payload['val']['size']} test={split_payload['test']['size']}"
    )

    model = GeNNiusModel(hidden_channels=args.hidden, metadata=data.metadata(), dropout=args.dropout).to(device)
    with torch.no_grad():
        model(data.x_dict, data.edge_index_dict, train_edge_label_index[:, : min(8, train_edge_label_index.size(1))])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = MaxMetricEarlyStopper(patience=args.patience)

    best_threshold = 0.5
    training_history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, train_logits = model(data.x_dict, data.edge_index_dict, train_edge_label_index)
        train_loss = torch.nn.functional.binary_cross_entropy_with_logits(train_logits, train_edge_label)
        train_loss.backward()
        optimizer.step()

        val_labels, val_scores, val_loss = evaluate(model, data, val_edge_label_index, val_edge_label)
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
                "train_loss": round(float(train_loss.item()), 6),
                "val_loss": round(float(val_loss), 6),
                "val_auc": round(float(val_metrics["auc"]), 4),
                "val_aupr": round(float(val_metrics["aupr"]), 4),
                "val_f1": round(float(val_metrics["f1"]), 4),
                "val_acc": round(float(val_metrics["acc"]), 4),
                "val_threshold": round(float(val_threshold), 6),
            }
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{epoch:3d}/{args.epochs}] train_loss={float(train_loss.item()):.5f} "
                f"val_AUC={val_metrics['auc']:.4f} val_AUPR={val_metrics['aupr']:.4f} "
                f"val_thr={val_threshold:.4f}"
            )

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    elapsed_minutes = (time.time() - start_time) / 60.0
    test_labels, test_scores, _ = evaluate(model, data, test_edge_label_index, test_edge_label)
    test_metrics = classification_metrics(test_labels, test_scores, best_threshold)

    with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    result = {
        "model": "GeNNius",
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "hidden": args.hidden,
        "lr": args.lr,
        "dropout": args.dropout,
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

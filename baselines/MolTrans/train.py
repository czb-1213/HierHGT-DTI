import json
import math
import os
import sys
from argparse import ArgumentParser
from time import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data

from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMPARE_DIR = os.path.join(ROOT, "baselines")
if COMPARE_DIR not in sys.path:
    sys.path.insert(0, COMPARE_DIR)

from common_metrics import (
    MaxMetricEarlyStopper,
    classification_metrics,
    select_threshold_by_f1,
)


SUPPORTED_SPLITS = ("random", "cold_drug", "cold_protein")

parser = ArgumentParser(description="MolTrans Training.")
parser.add_argument("-b", "--batch-size", default=16, type=int)
parser.add_argument("-j", "--workers", default=0, type=int)
parser.add_argument("--epochs", default=60, type=int)
parser.add_argument("--patience", default=8, type=int)
parser.add_argument("--task", default="", type=str)
parser.add_argument("--lr", "--learning-rate", default=1e-4, type=float, dest="lr")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--output_dir", default="./output", type=str)
parser.add_argument(
    "--data_root",
    default=os.path.join(ROOT, "datasets"),
    type=str,
    help="root directory containing <dataset>/<split>/train,val,test.csv",
)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_dataset_dir(data_root, dataset_name):
    candidate = os.path.join(data_root, dataset_name)
    if os.path.isdir(candidate):
        return candidate
    if os.path.isdir(data_root):
        for entry in os.listdir(data_root):
            entry_path = os.path.join(data_root, entry)
            if os.path.isdir(entry_path) and entry.lower() == dataset_name.lower():
                return entry_path
    raise FileNotFoundError(f"Dataset directory not found for '{dataset_name}' under {data_root}")


def infer_task_metadata(task_name, data_root):
    task_lower = task_name.lower()
    for split in SUPPORTED_SPLITS:
        suffix = f"_{split}"
        if task_lower.endswith(suffix):
            dataset_token = task_name[: -len(suffix)]
            dataset_dir = resolve_dataset_dir(data_root, dataset_token)
            return os.path.basename(dataset_dir), split
    return None, None


def get_task(task_name, data_root):
    task_map = {
        "biosnap": "./dataset/BIOSNAP/full_data",
        "bindingdb": "./dataset/BindingDB",
        "davis": "./dataset/DAVIS",
        "biosnap_random": os.path.join(data_root, "BioSnap", "random"),
        "biosnap_cold_drug": os.path.join(data_root, "BioSnap", "cold_drug"),
        "biosnap_cold_protein": os.path.join(data_root, "BioSnap", "cold_protein"),
        "drugbank_random": os.path.join(data_root, "DrugBank", "random"),
        "drugbank_cold_drug": os.path.join(data_root, "DrugBank", "cold_drug"),
        "drugbank_cold_protein": os.path.join(data_root, "DrugBank", "cold_protein"),
    }
    task_key = task_name.lower()
    if task_key in task_map:
        return task_map[task_key]
    if os.path.isdir(task_name):
        return os.path.abspath(task_name)

    dataset_name, split_name = infer_task_metadata(task_name, data_root)
    if dataset_name is not None:
        return os.path.join(data_root, dataset_name, split_name)
    raise KeyError(
        f"Unknown task '{task_name}'. Use a legacy task name, a direct directory path, "
        f"or '<dataset>_<split>' with splits in {SUPPORTED_SPLITS}."
    )


def evaluate_split(data_generator, model, device):
    scores = []
    labels_all = []
    loss_accumulate = 0.0
    count = 0
    loss_fct = nn.BCELoss()

    model.eval()
    with torch.no_grad():
        for d, p, d_mask, p_mask, label in data_generator:
            score = model(
                d.long().to(device),
                p.long().to(device),
                d_mask.long().to(device),
                p_mask.long().to(device),
            )
            probs = torch.sigmoid(score).view(-1)
            labels = torch.as_tensor(np.asarray(label), dtype=torch.float32, device=device).view(-1)
            loss = loss_fct(probs, labels)
            loss_accumulate += float(loss.item())
            count += 1
            scores.extend(probs.detach().cpu().numpy().tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = loss_accumulate / max(count, 1)
    return np.asarray(labels_all), np.asarray(scores), avg_loss


def main():
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = BIN_config_DBPE()
    config["batch_size"] = args.batch_size

    model = BIN_Interaction_Flat(**config).to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "drop_last": True,
    }
    eval_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "drop_last": False,
    }

    data_folder = get_task(args.task, args.data_root)
    dataset_name, split_name = infer_task_metadata(args.task, args.data_root)

    df_train = pd.read_csv(os.path.join(data_folder, "train.csv"))
    df_val = pd.read_csv(os.path.join(data_folder, "val.csv"))
    df_test = pd.read_csv(os.path.join(data_folder, "test.csv"))

    for frame in [df_train, df_val, df_test]:
        if "Protein" in frame.columns and "Target Sequence" not in frame.columns:
            frame.rename(columns={"Protein": "Target Sequence"}, inplace=True)
        if "Y" in frame.columns and "Label" not in frame.columns:
            frame.rename(columns={"Y": "Label"}, inplace=True)

    training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
    validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)

    training_generator = data.DataLoader(training_set, **train_params)
    validation_generator = data.DataLoader(validation_set, **eval_params)
    testing_generator = data.DataLoader(testing_set, **eval_params)

    stopper = MaxMetricEarlyStopper(patience=args.patience)
    training_history = []

    print("--- Go for Training ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for step, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
            scores = model(
                d.long().to(device),
                p.long().to(device),
                d_mask.long().to(device),
                p_mask.long().to(device),
            )
            labels = torch.as_tensor(np.asarray(label), dtype=torch.float32, device=device).view(-1)
            probs = torch.sigmoid(scores).view(-1)
            loss = nn.BCELoss()(probs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

            if step % 1000 == 0:
                print(
                    f"Training at Epoch {epoch} iteration {step} "
                    f"with loss {float(loss.item()):.6f}"
                )

        val_labels, val_scores, val_loss = evaluate_split(validation_generator, model, device)
        val_threshold = select_threshold_by_f1(val_labels, val_scores)
        val_metrics = classification_metrics(val_labels, val_scores, val_threshold)
        selection_score = val_metrics["aupr"]
        if math.isnan(selection_score):
            selection_score = float("-inf")

        stopper.update(
            selection_score,
            epoch,
            model,
            payload={
                "threshold": val_threshold,
                "val_metrics": val_metrics,
                "val_loss": val_loss,
            },
        )
        training_history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(np.mean(batch_losses)), 6),
                "val_loss": round(float(val_loss), 6),
                "val_auc": round(float(val_metrics["auc"]), 4),
                "val_aupr": round(float(val_metrics["aupr"]), 4),
                "val_f1": round(float(val_metrics["f1"]), 4),
                "val_acc": round(float(val_metrics["acc"]), 4),
                "val_threshold": round(float(val_threshold), 6),
            }
        )
        print(
            f"Validation at Epoch {epoch} , AUROC: {val_metrics['auc']:.4f} , "
            f"AUPRC: {val_metrics['aupr']:.4f} , F1: {val_metrics['f1']:.4f} , "
            f"ACC: {val_metrics['acc']:.4f} , threshold: {val_threshold:.6f}"
        )
        if stopper.should_stop:
            print(f"Early stopping triggered at epoch {stopper.stop_epoch}")
            break

    if stopper.best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    model.load_state_dict(stopper.best_state)
    best_payload = stopper.best_payload or {}
    best_threshold = float(best_payload.get("threshold", 0.5))
    best_val_metrics = best_payload.get("val_metrics", {})
    best_val_loss = float(best_payload.get("val_loss", float("nan")))

    print("--- Go for Testing ---")
    test_labels, test_scores, test_loss = evaluate_split(testing_generator, model, device)
    test_metrics = classification_metrics(test_labels, test_scores, best_threshold)
    print(
        f"Testing AUROC: {test_metrics['auc']:.4f} , "
        f"AUPRC: {test_metrics['aupr']:.4f} , "
        f"F1: {test_metrics['f1']:.4f} , "
        f"ACC: {test_metrics['acc']:.4f} , "
        f"threshold: {best_threshold:.6f} , "
        f"Test loss: {test_loss:.6f}"
    )

    results = {
        "model": "MolTrans",
        "dataset": dataset_name,
        "split": split_name,
        "seed": args.seed,
        "task": args.task,
        "auc": round(float(test_metrics["auc"]), 4),
        "aupr": round(float(test_metrics["aupr"]), 4),
        "f1": round(float(test_metrics["f1"]), 4),
        "acc": round(float(test_metrics["acc"]), 4),
        "precision": round(float(test_metrics["precision"]), 4),
        "recall": round(float(test_metrics["recall"]), 4),
        "specificity": round(float(test_metrics["specificity"]), 4),
        "test_loss": round(float(test_loss), 4),
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "best_epoch": stopper.best_epoch,
        "early_stop_epoch": stopper.stop_epoch,
        "selection_metric": "val_aupr",
        "threshold": round(best_threshold, 6),
        "threshold_policy": "val_f1_optimal",
        "val_loss_at_best": round(best_val_loss, 4) if not math.isnan(best_val_loss) else None,
        "val_metrics_at_best": {
            key: round(float(value), 4)
            for key, value in best_val_metrics.items()
            if key != "threshold"
        },
        "training_history": training_history,
    }
    out_dir = os.path.join(args.output_dir, args.task, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Results saved to {out_dir}/test_results.json")


if __name__ == "__main__":
    start_time = time()
    main()
    print(time() - start_time)

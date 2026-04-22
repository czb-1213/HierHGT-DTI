import copy
import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as_numpy_1d(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array


def _finite_binary_inputs(labels, scores):
    labels = _as_numpy_1d(labels)
    scores = _as_numpy_1d(scores)
    finite_mask = np.isfinite(labels) & np.isfinite(scores)
    if finite_mask.all():
        return labels, scores
    return labels[finite_mask], scores[finite_mask]


def safe_auc(labels, scores):
    labels, scores = _finite_binary_inputs(labels, scores)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_aupr(labels, scores):
    labels, scores = _finite_binary_inputs(labels, scores)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def select_threshold_by_f1(labels, scores, default_threshold=0.5):
    labels, scores = _finite_binary_inputs(labels, scores)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float(default_threshold)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return float(default_threshold)

    f1_scores = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-8, None)
    best_idx = int(np.nanargmax(f1_scores))
    best_threshold = thresholds[best_idx]
    if math.isnan(float(best_threshold)):
        return float(default_threshold)
    return float(best_threshold)


def classification_metrics(labels, scores, threshold):
    labels, scores = _finite_binary_inputs(labels, scores)
    labels = labels.astype(np.int64)
    preds = (scores >= float(threshold)).astype(np.int64)

    metrics = {
        "auc": safe_auc(labels, scores),
        "aupr": safe_aupr(labels, scores),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "acc": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "threshold": float(threshold),
    }

    if labels.size == 0:
        metrics["specificity"] = float("nan")
        return metrics

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    metrics["specificity"] = float(tn / denom) if denom else float("nan")
    return metrics


@dataclass
class MaxMetricEarlyStopper:
    patience: int
    min_delta: float = 0.0

    def __post_init__(self):
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.best_state = None
        self.best_payload = None
        self.counter = 0
        self.stop_epoch = None

    def update(self, metric, epoch, model, payload=None):
        improved = metric > self.best_metric + self.min_delta
        if improved:
            self.best_metric = float(metric)
            self.best_epoch = int(epoch)
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_payload = copy.deepcopy(payload) if payload is not None else None
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience and self.stop_epoch is None:
                self.stop_epoch = int(epoch)
        return improved

    @property
    def should_stop(self):
        return self.stop_epoch is not None

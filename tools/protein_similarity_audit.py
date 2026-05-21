#!/usr/bin/env python
"""Nearest-neighbor k-mer similarity audit for cold-protein splits.

The audit checks whether cold-protein test sequences are exact held-out
entities and how similar each test sequence is to the nearest train/validation
sequence. It does not replace a full sequence-cluster split; it documents the
similarity profile of the frozen benchmark splits.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DATASETS = ("BioSnap", "DrugBank")
SPLIT = "cold_protein"


def read_proteins(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "Protein" not in reader.fieldnames:
            raise ValueError(f"Missing Protein column in {path}")
        return [row["Protein"].strip() for row in reader if row.get("Protein", "").strip()]


def kmers(sequence: str, k: int) -> frozenset[str]:
    if len(sequence) < k:
        return frozenset({sequence})
    return frozenset(sequence[i : i + k] for i in range(len(sequence) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "pct_ge_0.3": sum(v >= 0.3 for v in values) / len(values) * 100.0,
        "pct_ge_0.5": sum(v >= 0.5 for v in values) / len(values) * 100.0,
        "pct_ge_0.7": sum(v >= 0.7 for v in values) / len(values) * 100.0,
        "pct_ge_0.9": sum(v >= 0.9 for v in values) / len(values) * 100.0,
    }


def audit_dataset(root: Path, dataset: str, k: int, top_n: int) -> tuple[list[str], list[dict[str, str | float | int]]]:
    split_dir = root / dataset / SPLIT
    train = read_proteins(split_dir / "train.csv")
    val = read_proteins(split_dir / "val.csv")
    test = read_proteins(split_dir / "test.csv")
    trainval_unique = sorted(set(train) | set(val))
    test_unique = sorted(set(test))
    exact_overlap = set(trainval_unique) & set(test_unique)

    trainval_kmers = [kmers(seq, k) for seq in trainval_unique]
    trainval_kmer_sizes = [len(item) for item in trainval_kmers]
    inverted_index: dict[str, list[int]] = defaultdict(list)
    for ref_idx, ref_kmers in enumerate(trainval_kmers):
        for token in ref_kmers:
            inverted_index[token].append(ref_idx)

    rows: list[dict[str, str | float | int]] = []
    for seq in test_unique:
        seq_kmers = kmers(seq, k)
        overlap_counts: dict[int, int] = defaultdict(int)
        for token in seq_kmers:
            for ref_idx in inverted_index.get(token, ()):
                overlap_counts[ref_idx] += 1

        best_seq = trainval_unique[0] if trainval_unique else ""
        best_score = -1.0
        for ref_idx, intersection_size in overlap_counts.items():
            union_size = len(seq_kmers) + trainval_kmer_sizes[ref_idx] - intersection_size
            score = intersection_size / union_size if union_size else 0.0
            if score > best_score:
                best_score = score
                best_seq = trainval_unique[ref_idx]
        if best_score < 0.0:
            best_score = 0.0
        rows.append(
            {
                "test_length": len(seq),
                "nearest_trainval_length": len(best_seq),
                "nearest_jaccard": best_score,
                "test_prefix": seq[:24],
                "nearest_prefix": best_seq[:24],
            }
        )

    stats = summarize([float(row["nearest_jaccard"]) for row in rows])
    lines = [
        f"## {dataset} {SPLIT}",
        "",
        f"- unique train+val proteins: {len(trainval_unique)}",
        f"- unique test proteins: {len(test_unique)}",
        f"- exact test protein overlap with train+val: {len(exact_overlap)}",
        f"- k-mer size: {k}",
        f"- nearest-neighbor Jaccard mean / median: {stats['mean']:.3f} / {stats['median']:.3f}",
        f"- p90 / p95 / max: {stats['p90']:.3f} / {stats['p95']:.3f} / {stats['max']:.3f}",
        f"- percent >= 0.3 / 0.5 / 0.7 / 0.9: "
        f"{stats['pct_ge_0.3']:.1f}% / {stats['pct_ge_0.5']:.1f}% / "
        f"{stats['pct_ge_0.7']:.1f}% / {stats['pct_ge_0.9']:.1f}%",
        "",
        "| Rank | Test length | Nearest train/val length | Nearest Jaccard | Test prefix | Nearest prefix |",
        "|---:|---:|---:|---:|---|---|",
    ]
    top_rows = sorted(rows, key=lambda item: float(item["nearest_jaccard"]), reverse=True)[:top_n]
    for idx, row in enumerate(top_rows, start=1):
        lines.append(
            f"| {idx} | {row['test_length']} | {row['nearest_trainval_length']} | "
            f"{float(row['nearest_jaccard']):.3f} | `{row['test_prefix']}` | `{row['nearest_prefix']}` |"
        )
    lines.append("")
    return lines, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", help="Dataset root")
    parser.add_argument("--k", type=int, default=5, help="Protein k-mer size")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output", default="data/protein_similarity_report.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    report = [
        "# Cold-Protein Sequence Similarity Audit",
        "",
        "This report uses nearest-neighbor k-mer Jaccard similarity between each unique cold-protein test sequence and the unique train/validation protein sequences in the same dataset. Exact overlap tests whether the split is entity-cold. Jaccard values describe similarity of the frozen split and should not be interpreted as a sequence-cluster holdout.",
        "",
    ]
    for dataset in DATASETS:
        lines, _ = audit_dataset(root, dataset, args.k, args.top_n)
        report.extend(lines)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Precompute TransformerCPI features for HierHGT-DTI benchmark splits."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

from mol_featurizer import mol_features
from word2vec import seq_to_kmers


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets"

SUPPORTED_SPLITS = ("random", "cold_drug", "cold_protein")


def protein_embedding(model: Word2Vec, sequence: str) -> np.ndarray:
    kmers = seq_to_kmers(sequence)
    dim = int(model.vector_size)
    if not kmers:
        return np.zeros((1, dim), dtype=np.float32)

    rows = []
    for kmer in kmers:
        if kmer in model.wv:
            rows.append(np.asarray(model.wv[kmer], dtype=np.float32))
        else:
            rows.append(np.zeros(dim, dtype=np.float32))
    return np.stack(rows, axis=0)


def load_split_records(path: Path, limit: int | None) -> list[tuple[str, str, int]]:
    frame = pd.read_csv(path)
    if limit is not None:
        frame = frame.head(limit)
    return [
        (str(row["SMILES"]), str(row["Protein"]), int(row["Y"]))
        for _, row in frame.iterrows()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HierHGT-DTI CSV splits to TransformerCPI caches")
    parser.add_argument("--dataset", required=True, choices=["BioSnap", "DrugBank"])
    parser.add_argument("--split", required=True, choices=SUPPORTED_SPLITS)
    parser.add_argument(
        "--data_root",
        default=str(DATASET_ROOT),
        help="Root containing <dataset>/<split>/{train,val,test}.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: baselines/TransformerCPI/data_adapted/<dataset>_<split>/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit applied independently to each split file.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_root = data_root / args.dataset / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    if args.output_dir is None:
        output_dir = ROOT / "data_adapted" / f"{args.dataset}_{args.split}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_records = {}
    for part in ("train", "val", "test"):
        split_path = split_root / f"{part}.csv"
        split_records[part] = load_split_records(split_path, args.limit)

    all_records = split_records["train"] + split_records["val"] + split_records["test"]
    unique_smiles = sorted({smiles for smiles, _, _ in all_records})
    unique_proteins = sorted({protein for _, protein, _ in all_records})

    print(f"Dataset: {args.dataset}/{args.split}")
    print(
        f"  train={len(split_records['train'])} val={len(split_records['val'])} "
        f"test={len(split_records['test'])}"
    )
    print(f"  unique_smiles={len(unique_smiles)} unique_proteins={len(unique_proteins)}")

    w2v_path = ROOT / "word2vec_30.model"
    if not w2v_path.exists():
        raise FileNotFoundError(f"Word2Vec model not found: {w2v_path}")
    word2vec_model = Word2Vec.load(str(w2v_path))

    drug_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    protein_cache: dict[str, np.ndarray] = {}

    for smiles in tqdm(unique_smiles, desc="Drug features"):
        atom_feat, adj = mol_features(smiles)
        atom_feat = np.asarray(atom_feat, dtype=np.float32)
        adj = np.asarray(adj, dtype=np.float32)
        # The batched official packer adds identity again during training.
        np.fill_diagonal(adj, 0.0)
        drug_cache[smiles] = (atom_feat, adj)

    for sequence in tqdm(unique_proteins, desc="Protein embeddings"):
        protein_cache[sequence] = protein_embedding(word2vec_model, sequence)

    with (output_dir / "feature_cache.pkl").open("wb") as f:
        pickle.dump(
            {
                "drug_cache": drug_cache,
                "protein_cache": protein_cache,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    for part, records in split_records.items():
        with (output_dir / f"{part}_records.pkl").open("wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "model": "TransformerCPI",
        "dataset": args.dataset,
        "split": args.split,
        "train_size": len(split_records["train"]),
        "val_size": len(split_records["val"]),
        "test_size": len(split_records["test"]),
        "unique_smiles": len(unique_smiles),
        "unique_proteins": len(unique_proteins),
        "limit_per_split": args.limit,
        "word2vec_model": str(w2v_path.relative_to(ROOT)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved adapted cache to {output_dir}")


if __name__ == "__main__":
    main()

"""Data preparation utilities for the adapted DO-GMA baseline."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, List

import dgl
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset

from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph


SUPPORTED_SPLITS = ("random", "cold_drug", "cold_protein")
PROTEIN_GRID_HEIGHT = 30
DRUG_SEQUENCE_WIDTH = 4

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARISOSMISET = {
    "(": 1,
    ".": 2,
    "0": 3,
    "2": 4,
    "4": 5,
    "6": 6,
    "8": 7,
    "@": 8,
    "B": 9,
    "D": 10,
    "F": 11,
    "H": 12,
    "L": 13,
    "N": 14,
    "P": 15,
    "R": 16,
    "T": 17,
    "V": 18,
    "Z": 19,
    "\\": 20,
    "b": 21,
    "d": 22,
    "f": 23,
    "h": 24,
    "l": 25,
    "n": 26,
    "r": 27,
    "t": 28,
    "#": 29,
    "%": 30,
    ")": 31,
    "+": 32,
    "-": 33,
    "/": 34,
    "1": 35,
    "3": 36,
    "5": 37,
    "7": 38,
    "9": 39,
    "=": 40,
    "A": 41,
    "C": 42,
    "E": 43,
    "G": 44,
    "I": 45,
    "K": 46,
    "M": 47,
    "O": 48,
    "S": 49,
    "U": 50,
    "W": 51,
    "Y": 52,
    "[": 53,
    "]": 54,
    "a": 55,
    "c": 56,
    "e": 57,
    "g": 58,
    "i": 59,
    "m": 60,
    "o": 61,
    "s": 62,
    "u": 63,
    "y": 64,
}


@dataclass(frozen=True)
class SplitCachePaths:
    root: str
    cache_pt: str
    metadata_json: str


def resolve_dataset_dir(data_root: str, dataset_name: str) -> str:
    candidate = os.path.join(data_root, dataset_name)
    if os.path.isdir(candidate):
        return candidate
    if os.path.isdir(data_root):
        for entry in os.listdir(data_root):
            entry_path = os.path.join(data_root, entry)
            if os.path.isdir(entry_path) and entry.lower() == dataset_name.lower():
                return entry_path
    raise FileNotFoundError(f"Dataset directory not found for '{dataset_name}' under {data_root}")


def split_cache_paths(base_dir: str, dataset: str, split: str) -> SplitCachePaths:
    root = os.path.join(base_dir, f"{dataset}_{split}")
    return SplitCachePaths(
        root=root,
        cache_pt=os.path.join(root, "cache.pt"),
        metadata_json=os.path.join(root, "metadata.json"),
    )


def normalize_split_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Target Sequence" in df.columns and "Protein" not in df.columns:
        df.rename(columns={"Target Sequence": "Protein"}, inplace=True)
    if "Label" in df.columns and "Y" not in df.columns:
        df.rename(columns={"Label": "Y"}, inplace=True)
    required = {"SMILES", "Protein", "Y"}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_str}")
    out = df[["SMILES", "Protein", "Y"]].copy()
    out["Y"] = out["Y"].astype(int)
    return out


def load_split_frames(data_root: str, dataset: str, split: str) -> Dict[str, pd.DataFrame]:
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {SUPPORTED_SPLITS}.")
    dataset_dir = resolve_dataset_dir(data_root, dataset)
    split_dir = os.path.join(dataset_dir, split)
    frames = {}
    for part in ("train", "val", "test"):
        frames[part] = normalize_split_frame(pd.read_csv(os.path.join(split_dir, f"{part}.csv")))
    return frames


def _ordered_unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(v) for v in values))


def round_up_multiple(value: int, divisor: int) -> int:
    return int(math.ceil(max(value, 1) / divisor) * divisor)


def integer_label_protein(sequence: str, max_length: int) -> np.ndarray:
    encoding = np.zeros(max_length, dtype=np.int64)
    for idx, letter in enumerate(str(sequence)[:max_length]):
        letter = letter.upper()
        if letter not in CHARPROTSET:
            logging.warning("Unknown protein character %s encountered; treating as padding.", letter)
            continue
        encoding[idx] = CHARPROTSET[letter]
    return encoding


def integer_label_drug(sequence: str, max_length: int) -> np.ndarray:
    encoding = np.zeros(max_length, dtype=np.int64)
    for idx, letter in enumerate(str(sequence)[:max_length]):
        letter = letter.upper()
        if letter not in CHARISOSMISET:
            logging.warning("Unknown SMILES character %s encountered; treating as padding.", letter)
            continue
        encoding[idx] = CHARISOSMISET[letter]
    return encoding


def inspect_drug_space(smiles_values: Iterable[str]) -> Dict[str, int]:
    max_atoms = 0
    max_smiles_len = 0
    for smiles in smiles_values:
        smiles = str(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
        max_atoms = max(max_atoms, mol.GetNumAtoms())
        max_smiles_len = max(max_smiles_len, len(smiles))

    max_drug_nodes = max(max_atoms, math.ceil(max_smiles_len / DRUG_SEQUENCE_WIDTH))
    return {
        "max_atoms": int(max_atoms),
        "max_smiles_len": int(max_smiles_len),
        "max_drug_nodes": int(max_drug_nodes),
        "max_drug_seq_len": int(max_drug_nodes * DRUG_SEQUENCE_WIDTH),
    }


def build_padded_drug_graph(
    smiles: str,
    max_drug_nodes: int,
    atom_featurizer: CanonicalAtomFeaturizer,
    bond_featurizer: CanonicalBondFeaturizer,
):
    smiles_to_graph = partial(smiles_to_bigraph, add_self_loop=True)
    graph = smiles_to_graph(smiles=smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
    actual_node_feats = graph.ndata.pop("h")
    num_actual_nodes = int(actual_node_feats.shape[0])
    if num_actual_nodes > max_drug_nodes:
        raise ValueError(f"Drug graph for {smiles} has {num_actual_nodes} nodes, exceeding max {max_drug_nodes}")

    num_virtual_nodes = max_drug_nodes - num_actual_nodes
    virtual_node_bit = torch.zeros([num_actual_nodes, 1], dtype=actual_node_feats.dtype)
    graph.ndata["h"] = torch.cat((actual_node_feats, virtual_node_bit), dim=1)
    if num_virtual_nodes > 0:
        virtual_node_feat = torch.cat(
            (
                torch.zeros(num_virtual_nodes, 74, dtype=actual_node_feats.dtype),
                torch.ones(num_virtual_nodes, 1, dtype=actual_node_feats.dtype),
            ),
            dim=1,
        )
        graph.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
    graph = dgl.add_self_loop(graph)
    return graph, num_actual_nodes


def build_cache(dataset: str, split: str, data_root: str, output_root: str, max_protein_len: int = 1200) -> SplitCachePaths:
    frames = load_split_frames(data_root, dataset, split)
    cache_paths = split_cache_paths(output_root, dataset, split)
    os.makedirs(cache_paths.root, exist_ok=True)

    all_smiles = _ordered_unique(pd.concat([frames["train"]["SMILES"], frames["val"]["SMILES"], frames["test"]["SMILES"]]))
    all_proteins = _ordered_unique(
        pd.concat([frames["train"]["Protein"], frames["val"]["Protein"], frames["test"]["Protein"]])
    )

    drug_stats = inspect_drug_space(all_smiles)
    raw_max_protein_len = max(len(seq) for seq in all_proteins)
    effective_protein_len = min(round_up_multiple(raw_max_protein_len, PROTEIN_GRID_HEIGHT), max_protein_len)
    if effective_protein_len % PROTEIN_GRID_HEIGHT != 0:
        raise ValueError(f"max_protein_len={effective_protein_len} must be divisible by {PROTEIN_GRID_HEIGHT}")

    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer(self_loop=True)

    drug_graphs = []
    drug_sequences = []
    drug_actual_nodes = []
    for smiles in all_smiles:
        graph, num_actual_nodes = build_padded_drug_graph(
            smiles=smiles,
            max_drug_nodes=drug_stats["max_drug_nodes"],
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
        )
        drug_graphs.append(graph)
        drug_sequences.append(torch.from_numpy(integer_label_drug(smiles, drug_stats["max_drug_seq_len"])).long())
        drug_actual_nodes.append(num_actual_nodes)

    protein_sequences = [
        torch.from_numpy(integer_label_protein(sequence, effective_protein_len)).long() for sequence in all_proteins
    ]

    drug_mapping = {smiles: idx for idx, smiles in enumerate(all_smiles)}
    protein_mapping = {sequence: idx for idx, sequence in enumerate(all_proteins)}

    split_payload = {}
    for part, frame in frames.items():
        split_payload[part] = {
            "drug_ids": torch.tensor([drug_mapping[smiles] for smiles in frame["SMILES"]], dtype=torch.long),
            "protein_ids": torch.tensor([protein_mapping[seq] for seq in frame["Protein"]], dtype=torch.long),
            "labels": torch.tensor(frame["Y"].tolist(), dtype=torch.float32),
            "size": int(len(frame)),
            "positive": int(frame["Y"].sum()),
        }

    metadata = {
        "dataset": dataset,
        "split": split,
        "num_drugs": len(all_smiles),
        "num_proteins": len(all_proteins),
        "max_drug_nodes": drug_stats["max_drug_nodes"],
        "max_drug_atoms": drug_stats["max_atoms"],
        "max_drug_seq_len": drug_stats["max_drug_seq_len"],
        "max_smiles_len": drug_stats["max_smiles_len"],
        "max_protein_len": int(effective_protein_len),
        "raw_max_protein_len": int(raw_max_protein_len),
        "protein_grid_height": PROTEIN_GRID_HEIGHT,
        "drug_sequence_width": DRUG_SEQUENCE_WIDTH,
        "drug_graph_feature_dim": 75,
        "protein_vocab_size": 26,
        "drug_vocab_size": 65,
        "num_truncated_proteins": int(sum(len(seq) > effective_protein_len for seq in all_proteins)),
        "mean_drug_nodes": round(float(np.mean(drug_actual_nodes)), 2),
    }

    payload = {
        "drug_graphs": drug_graphs,
        "drug_sequences": torch.stack(drug_sequences, dim=0),
        "protein_sequences": torch.stack(protein_sequences, dim=0),
        "splits": split_payload,
        "metadata": metadata,
    }

    torch.save(payload, cache_paths.cache_pt)
    with open(cache_paths.metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return cache_paths


class CachedDTIDataset(Dataset):
    def __init__(self, payload: Dict, split_name: str):
        split = payload["splits"][split_name]
        self.drug_graphs = payload["drug_graphs"]
        self.drug_sequences = payload["drug_sequences"]
        self.protein_sequences = payload["protein_sequences"]
        self.drug_ids = split["drug_ids"]
        self.protein_ids = split["protein_ids"]
        self.labels = split["labels"]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        drug_id = int(self.drug_ids[index])
        protein_id = int(self.protein_ids[index])
        return (
            self.drug_graphs[drug_id],
            self.drug_sequences[drug_id],
            self.protein_sequences[protein_id],
            self.labels[index],
        )


def graph_collate_func(batch):
    graphs, drug_tokens, protein_tokens, labels = zip(*batch)
    return (
        dgl.batch(graphs),
        torch.stack(drug_tokens, dim=0),
        torch.stack(protein_tokens, dim=0),
        torch.stack(labels, dim=0),
    )

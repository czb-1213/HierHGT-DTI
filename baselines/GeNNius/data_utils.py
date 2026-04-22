import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import HeteroData


SUPPORTED_SPLITS = ("random", "cold_drug", "cold_protein")

DRUG_DESCRIPTOR_NAMES = (
    "MolLogP",
    "MolWt",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "TPSA",
    "RingCount",
    "NHOHCount",
    "NOCount",
    "HeavyAtomCount",
    "NumValenceElectrons",
)

AA_ALPHABET = tuple(sorted("ARNDCQEGHILKMFPSTWYV"))


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
    if "Protein" in df.columns and "Target Sequence" not in df.columns:
        df.rename(columns={"Protein": "Target Sequence"}, inplace=True)
    if "Y" in df.columns and "Label" not in df.columns:
        df.rename(columns={"Y": "Label"}, inplace=True)
    required = {"SMILES", "Target Sequence", "Label"}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_str}")
    return df[["SMILES", "Target Sequence", "Label"]].copy()


def load_split_frames(data_root: str, dataset: str, split: str) -> Dict[str, pd.DataFrame]:
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {SUPPORTED_SPLITS}.")
    dataset_dir = resolve_dataset_dir(data_root, dataset)
    split_dir = os.path.join(dataset_dir, split)
    frames = {}
    for part in ("train", "val", "test"):
        frame = pd.read_csv(os.path.join(split_dir, f"{part}.csv"))
        frames[part] = normalize_split_frame(frame)
    return frames


def seq2rat(sequence: str):
    sequence = str(sequence).upper()
    seq_len = max(len(sequence), 1)
    counts = {aa: sequence.count(aa) for aa in AA_ALPHABET}
    return [counts[aa] / seq_len for aa in AA_ALPHABET]


def compute_drug_descriptor_vector(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    return [
        float(Descriptors.MolLogP(mol)),
        float(Descriptors.MolWt(mol)),
        float(Descriptors.NumHAcceptors(mol)),
        float(Descriptors.NumHDonors(mol)),
        float(Descriptors.NumHeteroatoms(mol)),
        float(Descriptors.NumRotatableBonds(mol)),
        float(Descriptors.TPSA(mol)),
        float(Descriptors.RingCount(mol)),
        float(Descriptors.NHOHCount(mol)),
        float(Descriptors.NOCount(mol)),
        float(Descriptors.HeavyAtomCount(mol)),
        float(Descriptors.NumValenceElectrons(mol)),
    ]


def _ordered_unique(values: Iterable[str]):
    return list(dict.fromkeys(str(v) for v in values))


def _minmax_scale_by_train(all_values: Dict[str, list], train_keys: Iterable[str]):
    train_keys = list(dict.fromkeys(train_keys))
    train_tensor = torch.tensor([all_values[key] for key in train_keys], dtype=torch.float32)
    min_vals = train_tensor.min(dim=0).values
    max_vals = train_tensor.max(dim=0).values
    denom = torch.where((max_vals - min_vals) > 0, max_vals - min_vals, torch.ones_like(max_vals))

    scaled = {}
    for key, values in all_values.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        tensor = (tensor - min_vals) / denom
        scaled[key] = tensor.clamp(0.0, 1.0)
    return scaled


def build_cache(dataset: str, split: str, data_root: str, output_root: str):
    frames = load_split_frames(data_root, dataset, split)
    cache_paths = split_cache_paths(output_root, dataset, split)
    os.makedirs(cache_paths.root, exist_ok=True)

    all_drugs = _ordered_unique(
        pd.concat([frames["train"]["SMILES"], frames["val"]["SMILES"], frames["test"]["SMILES"]])
    )
    all_proteins = _ordered_unique(
        pd.concat(
            [
                frames["train"]["Target Sequence"],
                frames["val"]["Target Sequence"],
                frames["test"]["Target Sequence"],
            ]
        )
    )
    train_drugs = _ordered_unique(frames["train"]["SMILES"])

    raw_drug_features = {smiles: compute_drug_descriptor_vector(smiles) for smiles in all_drugs}
    drug_features = _minmax_scale_by_train(raw_drug_features, train_drugs)
    protein_features = {
        seq: torch.tensor(seq2rat(seq), dtype=torch.float32) for seq in all_proteins
    }

    drug_mapping = {smiles: idx for idx, smiles in enumerate(all_drugs)}
    protein_mapping = {seq: idx for idx, seq in enumerate(all_proteins)}

    train_pos = frames["train"][frames["train"]["Label"] == 1].drop_duplicates(
        subset=["SMILES", "Target Sequence"]
    )
    src = [drug_mapping[smiles] for smiles in train_pos["SMILES"]]
    dst = [protein_mapping[seq] for seq in train_pos["Target Sequence"]]

    data = HeteroData()
    data["drug"].x = torch.stack([drug_features[smiles] for smiles in all_drugs], dim=0)
    data["protein"].x = torch.stack([protein_features[seq] for seq in all_proteins], dim=0)
    data["drug", "interaction", "protein"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    split_payload = {}
    for part, frame in frames.items():
        edge_src = [drug_mapping[smiles] for smiles in frame["SMILES"]]
        edge_dst = [protein_mapping[seq] for seq in frame["Target Sequence"]]
        split_payload[part] = {
            "edge_label_index": torch.tensor([edge_src, edge_dst], dtype=torch.long),
            "edge_label": torch.tensor(frame["Label"].tolist(), dtype=torch.float32),
            "size": int(len(frame)),
            "positive": int(frame["Label"].sum()),
        }

    payload = {
        "data": data,
        "splits": split_payload,
        "metadata": {
            "dataset": dataset,
            "split": split,
            "num_drugs": len(all_drugs),
            "num_proteins": len(all_proteins),
            "num_train_pos_edges": int(len(train_pos)),
            "drug_feature_dim": int(data["drug"].x.size(-1)),
            "protein_feature_dim": int(data["protein"].x.size(-1)),
            "descriptor_names": list(DRUG_DESCRIPTOR_NAMES),
        },
    }

    torch.save(payload, cache_paths.cache_pt)
    with open(cache_paths.metadata_json, "w", encoding="utf-8") as f:
        json.dump(payload["metadata"], f, indent=2)
    return cache_paths

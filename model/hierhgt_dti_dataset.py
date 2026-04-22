import hashlib
import logging
import os
from typing import Dict, List, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HierHGTDTIDataset(Dataset):
    """
    HierHGT-DTI DTI dataset with strict cache validation.
    """

    def __init__(
        self,
        csv_path: str,
        drug_cache_dirs: List[str],
        protein_cache_dirs: List[str],
        smiles_col: str = "SMILES",
        protein_col: str = "Protein",
        label_col: str = "Y",
        cache_in_memory: bool = True,
    ):
        self.csv_path = csv_path
        self.drug_cache_dirs = drug_cache_dirs
        self.protein_cache_dirs = protein_cache_dirs
        self.smiles_col = smiles_col
        self.protein_col = protein_col
        self.label_col = label_col
        self.cache_in_memory = bool(cache_in_memory)

        self.dataset_name = self._infer_dataset_name_from_csv()
        self.drug_cache_index = self._build_drug_cache_index()
        self.protein_cache_index = self._build_protein_cache_index()
        logger.info(
            "Cache index built: drug=%d, protein=%d",
            len(self.drug_cache_index),
            len(self.protein_cache_index),
        )

        raw_data = self._load_data()
        self.data = raw_data

        unique_smiles = sorted(set(item["smiles"] for item in self.data))
        unique_proteins = sorted(set(item["protein_seq"] for item in self.data))
        self.smiles_to_id = {s: i for i, s in enumerate(unique_smiles)}
        self.protein_to_id = {p: i for i, p in enumerate(unique_proteins)}
        self.drug_cache_path_by_smiles: Dict[str, Optional[str]] = {
            s: self.drug_cache_index.get(hashlib.md5(s.encode()).hexdigest())
            for s in unique_smiles
        }
        self.protein_cache_path_by_seq: Dict[str, Optional[str]] = {
            p: self.protein_cache_index.get(hashlib.md5(p.encode()).hexdigest())
            for p in unique_proteins
        }
        self._assert_all_cached(self.data)
        logger.info("Loaded DTI dataset with %d samples (cache validated).", len(self.data))

        self.drug_graph_cache: Dict[str, dgl.DGLGraph] = {}
        self.protein_graph_cache: Dict[str, dgl.DGLGraph] = {}

        # Typed-edge cache state.
        self.local_typed_edge_ready = False
        self.drug_num_etypes: Optional[int] = None
        self.protein_num_etypes: Optional[int] = None
        self.protein_etype_key: Optional[str] = None
        self.drug_local_typed_edge_cache: Dict[str, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = {}
        self.protein_local_typed_edge_cache: Dict[str, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = {}
        self.protein_other_drop_ratio_cache: Dict[str, float] = {}
        self.protein_local_typed_ew_cache: Dict[str, Optional[Tuple[torch.Tensor, ...]]] = {}

        if self.cache_in_memory:
            self._preload_graphs()

    @staticmethod
    def _normalize_path(path: str) -> str:
        return os.path.normcase(os.path.normpath(path))

    def _register_cache_entry(self, index: dict, hash_key: str, item_path: str, cache_type: str):
        existing = index.get(hash_key)
        if existing is None:
            index[hash_key] = item_path
            return
        if self._normalize_path(existing) != self._normalize_path(item_path):
            raise RuntimeError(
                f"{cache_type} cache hash conflict: hash={hash_key}, "
                f"path1={existing}, path2={item_path}"
            )

    def _scan_drug_dir(self, directory: str, index: dict):
        try:
            for item in sorted(os.listdir(directory)):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    self._scan_drug_dir(item_path, index)
                elif item.startswith("drug_") and item.endswith(".bin"):
                    hash_key = item[5:-4]
                    self._register_cache_entry(index, hash_key, item_path, cache_type="drug")
        except PermissionError:
            return

    def _scan_protein_dir(self, directory: str, index: dict):
        try:
            for item in sorted(os.listdir(directory)):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    self._scan_protein_dir(item_path, index)
                elif item.endswith(".bin"):
                    if item.startswith("protein_"):
                        hash_key = item[8:-4]
                    else:
                        hash_key = item[:-4]
                    self._register_cache_entry(index, hash_key, item_path, cache_type="protein")
        except PermissionError:
            return

    def _infer_dataset_name_from_csv(self) -> Optional[str]:
        # Expected shape: .../data/<DatasetName>/<split>/train.csv
        norm = os.path.normpath(self.csv_path)
        parts = norm.split(os.sep)
        if len(parts) >= 3:
            guessed = parts[-3]
            if guessed:
                logger.info("Inferred dataset name from CSV path: %s", guessed)
                return guessed
        logger.warning(
            "Unable to infer dataset name from CSV path; scanning provided cache directories."
        )
        return None

    def _resolve_drug_scan_dirs(self, cache_dir: str) -> List[str]:
        if self.dataset_name:
            ds_dir = os.path.join(cache_dir, self.dataset_name)
            if os.path.isdir(ds_dir):
                return [ds_dir]
        return [cache_dir]

    def _resolve_protein_scan_dirs(self, cache_dir: str) -> List[str]:
        if self.dataset_name:
            graph_dir = os.path.join(cache_dir, f"{self.dataset_name}_graphs")
            if os.path.isdir(graph_dir):
                return [graph_dir]
            ds_dir = os.path.join(cache_dir, self.dataset_name)
            if os.path.isdir(ds_dir):
                return [ds_dir]
        return [cache_dir]

    def _build_drug_cache_index(self) -> dict:
        index = {}
        for cache_dir in self.drug_cache_dirs:
            for scan_dir in self._resolve_drug_scan_dirs(cache_dir):
                if os.path.exists(scan_dir):
                    self._scan_drug_dir(scan_dir, index)
        return index

    def _build_protein_cache_index(self) -> dict:
        index = {}
        for cache_dir in self.protein_cache_dirs:
            for scan_dir in self._resolve_protein_scan_dirs(cache_dir):
                if os.path.exists(scan_dir):
                    self._scan_protein_dir(scan_dir, index)
        return index

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        required_cols = [self.smiles_col, self.protein_col, self.label_col]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}. Existing: {list(df.columns)}"
            )

        data_df = df[[self.smiles_col, self.protein_col, self.label_col]].rename(
            columns={
                self.smiles_col: "smiles",
                self.protein_col: "protein_seq",
                self.label_col: "label",
            }
        ).copy()

        null_mask = data_df[["smiles", "protein_seq", "label"]].isnull().any(axis=1)
        if bool(null_mask.any()):
            bad_rows = data_df.index[null_mask].tolist()[:10]
            raise ValueError(f"Detected missing smiles/protein_seq/label values, sample rows: {bad_rows}")

        data_df["smiles"] = data_df["smiles"].astype(str).str.strip()
        data_df["protein_seq"] = data_df["protein_seq"].astype(str).str.strip()
        empty_mask = (data_df["smiles"] == "") | (data_df["protein_seq"] == "")
        if bool(empty_mask.any()):
            bad_rows = data_df.index[empty_mask].tolist()[:10]
            raise ValueError(f"Detected empty smiles/protein_seq strings, sample rows: {bad_rows}")

        labels = pd.to_numeric(data_df["label"], errors="coerce")
        finite_mask = np.isfinite(labels.to_numpy())
        if not bool(finite_mask.all()):
            bad_rows = data_df.index[~finite_mask].tolist()[:10]
            raise ValueError(f"Detected invalid labels (NaN/Inf/non-numeric), sample rows: {bad_rows}")

        range_mask = (labels < 0.0) | (labels > 1.0)
        if bool(range_mask.any()):
            bad_rows = data_df.index[range_mask].tolist()[:10]
            bad_vals = labels[range_mask].astype(float).tolist()[:10]
            raise ValueError(
                f"Detected labels outside [0,1], sample rows: {bad_rows}, sample values: {bad_vals}"
            )

        data_df["label"] = labels.astype(float)
        return data_df.to_dict("records")

    def _assert_all_cached(self, data):
        n_total = len(data)
        drug_miss_set = set()
        protein_miss_set = set()
        for item in data:
            if self._get_drug_cache_path(item["smiles"]) is None:
                drug_miss_set.add(item["smiles"])
            if self._get_protein_cache_path(item["protein_seq"]) is None:
                protein_miss_set.add(item["protein_seq"])

        if drug_miss_set or protein_miss_set:
            example_drugs = list(drug_miss_set)[:3]
            example_proteins = list(protein_miss_set)[:3]
            raise RuntimeError(
                "Missing graph cache detected; stop training in strict mode. "
                f"n_samples={n_total}, missing_drug={len(drug_miss_set)}, "
                f"missing_protein={len(protein_miss_set)}, "
                f"drug_examples={example_drugs}, protein_examples={example_proteins}."
            )

    def _preload_graphs(self):
        logger.info("Preloading graphs into memory...")
        for item in self.data:
            smiles = item["smiles"]
            protein_seq = item["protein_seq"]

            drug_cache_path = self._get_drug_cache_path(smiles)
            if drug_cache_path and smiles not in self.drug_graph_cache:
                drug_graph = self._load_graph_from_cache(drug_cache_path)
                if drug_graph is not None:
                    self.drug_graph_cache[smiles] = drug_graph

            protein_cache_path = self._get_protein_cache_path(protein_seq)
            if protein_cache_path and protein_seq not in self.protein_graph_cache:
                protein_graph = self._load_graph_from_cache(protein_cache_path)
                if protein_graph is not None:
                    self.protein_graph_cache[protein_seq] = protein_graph

        logger.info(
            "Preload completed: %d drug graphs, %d protein graphs",
            len(self.drug_graph_cache),
            len(self.protein_graph_cache),
        )

    def _typed_edge_disk_cache_path(self, drug_num_etypes: int, protein_num_etypes: int, protein_etype_key: str) -> Optional[str]:
        """Compute a deterministic disk cache path for typed-edge data."""
        csv_dir = os.path.dirname(os.path.abspath(self.csv_path))
        cache_dir = os.path.join(csv_dir, ".typed_edge_cache")
        # Key: csv filename + entity counts + etype config
        n_drug = len(self.drug_graph_cache) if self.cache_in_memory else 0
        n_prot = len(self.protein_graph_cache) if self.cache_in_memory else 0
        if n_drug == 0 and n_prot == 0:
            return None
        csv_stem = os.path.splitext(os.path.basename(self.csv_path))[0]
        tag = f"{csv_stem}_d{n_drug}_p{n_prot}_de{drug_num_etypes}_pe{protein_num_etypes}_{protein_etype_key}"
        return os.path.join(cache_dir, f"typed_edges_{tag}.pt")

    def prepare_local_typed_edges(
        self,
        drug_num_etypes: int,
        protein_num_etypes: int,
        protein_etype_key: str,
    ):
        """
        Prepare local typed-edge cache:
        - cache_in_memory=True: full pre-build over cached graphs, with disk caching.
        - cache_in_memory=False: only record config; __getitem__ lazy-builds and reuses.
        """
        self.drug_num_etypes = int(drug_num_etypes)
        self.protein_num_etypes = int(protein_num_etypes)
        self.protein_etype_key = str(protein_etype_key)
        if self.drug_num_etypes <= 0 or self.protein_num_etypes <= 0:
            raise ValueError(
                f"Invalid etype bucket sizes: drug={self.drug_num_etypes}, protein={self.protein_num_etypes}"
            )
        if not self.protein_etype_key:
            raise ValueError("protein_etype_key must be non-empty.")

        self.drug_local_typed_edge_cache = {}
        self.protein_local_typed_edge_cache = {}
        self.protein_other_drop_ratio_cache = {}
        self.protein_local_typed_ew_cache: Dict[str, Optional[Tuple[torch.Tensor, ...]]] = {}

        if self.cache_in_memory:
            disk_path = self._typed_edge_disk_cache_path(
                self.drug_num_etypes, self.protein_num_etypes, self.protein_etype_key
            )

            # Try loading from disk
            if disk_path and os.path.isfile(disk_path):
                try:
                    cached = torch.load(disk_path, map_location="cpu", weights_only=False)
                    self.drug_local_typed_edge_cache = cached["drug"]
                    self.protein_local_typed_edge_cache = cached["protein"]
                    self.protein_other_drop_ratio_cache = cached["protein_drop"]
                    self.protein_local_typed_ew_cache = cached.get("protein_ew", {})
                    self.local_typed_edge_ready = True
                    logger.info(
                        "Typed-edge cache loaded from disk: %s (drug=%d, protein=%d)",
                        disk_path,
                        len(self.drug_local_typed_edge_cache),
                        len(self.protein_local_typed_edge_cache),
                    )
                    return
                except Exception as e:
                    logger.warning("Failed to load typed-edge disk cache (%s), rebuilding: %s", disk_path, e)

            # Build from scratch
            for smiles, graph in self.drug_graph_cache.items():
                drug_local, _, _ = self._build_local_typed_edges(
                    graph=graph,
                    etype_key="drug_etype_id",
                    num_etypes=self.drug_num_etypes,
                    filter_negative=False,
                    entity_name=f"drug[{smiles}]",
                )
                self.drug_local_typed_edge_cache[smiles] = drug_local

            for protein_seq, graph in self.protein_graph_cache.items():
                protein_local, drop_ratio, typed_ew = self._build_local_typed_edges(
                    graph=graph,
                    etype_key=self.protein_etype_key,
                    num_etypes=self.protein_num_etypes,
                    filter_negative=True,
                    entity_name=f"protein[{protein_seq[:24]}]",
                    extract_edge_weight=True,
                )
                self.protein_local_typed_edge_cache[protein_seq] = protein_local
                self.protein_other_drop_ratio_cache[protein_seq] = drop_ratio
                self.protein_local_typed_ew_cache[protein_seq] = typed_ew

            # Save to disk
            if disk_path:
                try:
                    os.makedirs(os.path.dirname(disk_path), exist_ok=True)
                    torch.save({
                        "drug": self.drug_local_typed_edge_cache,
                        "protein": self.protein_local_typed_edge_cache,
                        "protein_drop": self.protein_other_drop_ratio_cache,
                        "protein_ew": self.protein_local_typed_ew_cache,
                    }, disk_path)
                    logger.info("Typed-edge cache saved to disk: %s", disk_path)
                except Exception as e:
                    logger.warning("Failed to save typed-edge disk cache: %s", e)

        self.local_typed_edge_ready = True
        logger.info(
            "Local typed-edge cache prepared: cache_in_memory=%s, drug_cached=%d, protein_cached=%d",
            bool(self.cache_in_memory),
            len(self.drug_local_typed_edge_cache),
            len(self.protein_local_typed_edge_cache),
        )

    @staticmethod
    def _validate_edge_pair(src: torch.Tensor, dst: torch.Tensor, edge_name: str):
        if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
            raise TypeError(f"{edge_name} must be tensor pair.")
        if src.dtype != torch.int64 or dst.dtype != torch.int64:
            raise TypeError(
                f"{edge_name} expects int64 tensors, got {src.dtype} and {dst.dtype}."
            )
        if src.ndim != 1 or dst.ndim != 1:
            raise ValueError(f"{edge_name} expects 1D tensors, got {src.ndim} and {dst.ndim}.")
        if src.numel() != dst.numel():
            raise ValueError(
                f"{edge_name} src/dst edge count mismatch: {src.numel()} vs {dst.numel()}."
            )

    def _build_local_typed_edges(
        self,
        graph: dgl.DGLGraph,
        etype_key: str,
        num_etypes: int,
        filter_negative: bool,
        entity_name: str,
        extract_edge_weight: bool = False,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], ...], float, Optional[Tuple[torch.Tensor, ...]]]:
        if etype_key not in graph.edata:
            raise KeyError(
                f"{entity_name} missing edge type ids '{etype_key}'. "
                "Ensure etype-id precomputation runs before local typed-edge preparation."
            )

        src, dst = graph.edges(order="eid")
        src = src.to(dtype=torch.int64, device="cpu")
        dst = dst.to(dtype=torch.int64, device="cpu")
        self._validate_edge_pair(src, dst, f"{entity_name}.edges(order='eid')")

        etype_idx = graph.edata[etype_key]
        if not isinstance(etype_idx, torch.Tensor):
            raise TypeError(f"{entity_name}.{etype_key} must be a Tensor.")
        etype_idx = etype_idx.to(dtype=torch.long, device="cpu").reshape(-1)
        if etype_idx.numel() != src.numel():
            raise ValueError(
                f"{entity_name}.{etype_key} edge count mismatch: {etype_idx.numel()} vs {src.numel()}."
            )

        has_ew = extract_edge_weight and "edge_weight" in graph.edata
        if has_ew:
            ew = graph.edata["edge_weight"].to(dtype=torch.float32, device="cpu").reshape(-1)
        else:
            ew = None

        drop_ratio = 0.0
        if filter_negative:
            total_edges = int(etype_idx.numel())
            invalid_mask = etype_idx < 0
            invalid_edges = int(invalid_mask.sum().item()) if total_edges > 0 else 0
            drop_ratio = (invalid_edges / float(total_edges)) if total_edges > 0 else 0.0
            valid_mask = ~invalid_mask
            src = src[valid_mask]
            dst = dst[valid_mask]
            etype_idx = etype_idx[valid_mask]
            if ew is not None:
                ew = ew[valid_mask]

        if etype_idx.numel() > 0:
            etype_min = int(etype_idx.min().item())
            etype_max = int(etype_idx.max().item())
            if etype_min < 0:
                raise ValueError(
                    f"{entity_name}.{etype_key} has negative etype id after filtering: min={etype_min}."
                )
            if etype_max >= int(num_etypes):
                raise ValueError(
                    f"{entity_name}.{etype_key} out-of-range etype id: max={etype_max}, "
                    f"num_etypes={int(num_etypes)}."
                )

        empty = torch.zeros(0, dtype=torch.int64)
        empty_ew = torch.zeros(0, dtype=torch.float32)
        edge_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        ew_buckets: List[torch.Tensor] = []
        for etype_id in range(int(num_etypes)):
            if etype_idx.numel() == 0:
                edge_pairs.append((empty, empty))
                ew_buckets.append(empty_ew)
                continue
            mask = etype_idx == etype_id
            if bool(mask.any()):
                edge_pairs.append((src[mask], dst[mask]))
                ew_buckets.append(ew[mask] if ew is not None else empty_ew)
            else:
                edge_pairs.append((empty, empty))
                ew_buckets.append(empty_ew)

        typed_ew: Optional[Tuple[torch.Tensor, ...]] = tuple(ew_buckets) if has_ew else None
        return tuple(edge_pairs), float(drop_ratio), typed_ew

    def _get_or_build_drug_typed_local(
        self,
        smiles: str,
        drug_graph: dgl.DGLGraph,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        if smiles not in self.drug_local_typed_edge_cache:
            drug_local, _, _ = self._build_local_typed_edges(
                graph=drug_graph,
                etype_key="drug_etype_id",
                num_etypes=int(self.drug_num_etypes),
                filter_negative=False,
                entity_name=f"drug[{smiles}]",
            )
            self.drug_local_typed_edge_cache[smiles] = drug_local
        return self.drug_local_typed_edge_cache[smiles]

    def _get_or_build_protein_typed_local(
        self,
        protein_seq: str,
        protein_graph: dgl.DGLGraph,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], ...], float, Optional[Tuple[torch.Tensor, ...]]]:
        if protein_seq not in self.protein_local_typed_edge_cache:
            protein_local, drop_ratio, typed_ew = self._build_local_typed_edges(
                graph=protein_graph,
                etype_key=str(self.protein_etype_key),
                num_etypes=int(self.protein_num_etypes),
                filter_negative=True,
                entity_name=f"protein[{protein_seq[:24]}]",
                extract_edge_weight=True,
            )
            self.protein_local_typed_edge_cache[protein_seq] = protein_local
            self.protein_other_drop_ratio_cache[protein_seq] = float(drop_ratio)
            self.protein_local_typed_ew_cache[protein_seq] = typed_ew
        return (
            self.protein_local_typed_edge_cache[protein_seq],
            float(self.protein_other_drop_ratio_cache[protein_seq]),
            self.protein_local_typed_ew_cache.get(protein_seq),
        )

    def _assert_local_typed_edge_ready(self):
        if not bool(self.local_typed_edge_ready):
            raise RuntimeError(
                "Local typed-edge cache is not prepared. "
                "Call prepare_local_typed_edges(...) after etype-id precomputation."
            )

    def _get_drug_cache_path(self, smiles: str) -> Optional[str]:
        if smiles in self.drug_cache_path_by_smiles:
            return self.drug_cache_path_by_smiles[smiles]
        smiles_hash = hashlib.md5(smiles.encode()).hexdigest()
        return self.drug_cache_index.get(smiles_hash)

    def _get_protein_cache_path(self, protein_seq: str) -> Optional[str]:
        if protein_seq in self.protein_cache_path_by_seq:
            return self.protein_cache_path_by_seq[protein_seq]
        protein_hash = hashlib.md5(protein_seq.encode()).hexdigest()
        return self.protein_cache_index.get(protein_hash)

    def _load_graph_from_cache(self, cache_path: str) -> Optional[dgl.DGLGraph]:
        if cache_path is None:
            return None
        try:
            graphs, _ = dgl.load_graphs(cache_path)
            return graphs[0]
        except Exception as e:
            logger.warning("Failed to load graph cache %s: %s", cache_path, e)
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self._assert_local_typed_edge_ready()
        item = self.data[idx]

        if self.cache_in_memory and item["smiles"] in self.drug_graph_cache:
            drug_graph = self.drug_graph_cache[item["smiles"]]
        else:
            drug_cache_path = self._get_drug_cache_path(item["smiles"])
            drug_graph = self._load_graph_from_cache(drug_cache_path)

        if self.cache_in_memory and item["protein_seq"] in self.protein_graph_cache:
            protein_graph = self.protein_graph_cache[item["protein_seq"]]
        else:
            protein_cache_path = self._get_protein_cache_path(item["protein_seq"])
            protein_graph = self._load_graph_from_cache(protein_cache_path)

        if drug_graph is None or protein_graph is None:
            missing = []
            if drug_graph is None:
                missing.append(f"drug({item['smiles'][:50]})")
            if protein_graph is None:
                missing.append(f"protein({item['protein_seq'][:30]})")
            raise RuntimeError(f"Sample {idx} graph load failed: {', '.join(missing)}")

        drug_typed_local = self._get_or_build_drug_typed_local(item["smiles"], drug_graph)
        protein_typed_local, protein_other_drop_ratio, protein_typed_ew = self._get_or_build_protein_typed_local(
            item["protein_seq"],
            protein_graph,
        )
        label = torch.tensor([item["label"]], dtype=torch.float32)

        return {
            "drug_graph": drug_graph,
            "protein_graph": protein_graph,
            "drug_typed_local": drug_typed_local,
            "protein_typed_local": protein_typed_local,
            "protein_typed_ew": protein_typed_ew,
            "protein_other_drop_ratio": protein_other_drop_ratio,
            "label": label,
            "drug_id": self.smiles_to_id[item["smiles"]],
            "protein_id": self.protein_to_id[item["protein_seq"]],
        }


def _build_node_offsets(graphs: List[dgl.DGLGraph]) -> List[int]:
    offsets: List[int] = []
    cur = 0
    for g in graphs:
        offsets.append(cur)
        cur += int(g.num_nodes())
    return offsets


def _pack_typed_edges(
    local_typed_list: List[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
    offsets: List[int],
    expected_num_etypes: int,
    edge_name: str,
    local_typed_ew_list: Optional[List[Optional[Tuple[torch.Tensor, ...]]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if len(local_typed_list) != len(offsets):
        raise ValueError(
            f"{edge_name}: sample count mismatch between local_typed_list and offsets."
        )

    has_ew = (
        local_typed_ew_list is not None
        and any(ew is not None for ew in local_typed_ew_list)
    )

    src_chunks: List[List[torch.Tensor]] = [[] for _ in range(expected_num_etypes)]
    dst_chunks: List[List[torch.Tensor]] = [[] for _ in range(expected_num_etypes)]
    ew_chunks: List[List[torch.Tensor]] = [[] for _ in range(expected_num_etypes)] if has_ew else []

    for i, typed_edges in enumerate(local_typed_list):
        if not isinstance(typed_edges, (tuple, list)):
            raise TypeError(f"{edge_name}[{i}] must be a tuple/list of edge pairs.")
        if len(typed_edges) != int(expected_num_etypes):
            raise ValueError(
                f"{edge_name}[{i}] etype bucket mismatch: {len(typed_edges)} vs {expected_num_etypes}."
            )

        sample_ew = local_typed_ew_list[i] if has_ew and local_typed_ew_list is not None else None

        node_offset = int(offsets[i])
        for etype_id in range(int(expected_num_etypes)):
            edge_pair = typed_edges[etype_id]
            if not isinstance(edge_pair, (tuple, list)) or len(edge_pair) != 2:
                raise TypeError(f"{edge_name}[{i}][{etype_id}] must be a (src, dst) pair.")
            src, dst = edge_pair
            HierHGTDTIDataset._validate_edge_pair(src, dst, f"{edge_name}[{i}][{etype_id}]")
            if src.numel() == 0:
                continue
            src_chunks[etype_id].append(src + node_offset)
            dst_chunks[etype_id].append(dst + node_offset)
            if has_ew and sample_ew is not None:
                ew_chunks[etype_id].append(sample_ew[etype_id])

    empty = torch.zeros(0, dtype=torch.int64)
    empty_ew = torch.zeros(0, dtype=torch.float32)
    packed_src_chunks: List[torch.Tensor] = []
    packed_dst_chunks: List[torch.Tensor] = []
    packed_ew_chunks: List[torch.Tensor] = [] if has_ew else []
    ptr: List[int] = [0]
    for etype_id in range(int(expected_num_etypes)):
        if len(src_chunks[etype_id]) == 0:
            ptr.append(ptr[-1])
            continue
        etype_src = torch.cat(src_chunks[etype_id], dim=0)
        etype_dst = torch.cat(dst_chunks[etype_id], dim=0)
        packed_src_chunks.append(etype_src)
        packed_dst_chunks.append(etype_dst)
        if has_ew:
            if len(ew_chunks[etype_id]) > 0:
                packed_ew_chunks.append(torch.cat(ew_chunks[etype_id], dim=0))
            else:
                packed_ew_chunks.append(torch.ones(etype_src.numel(), dtype=torch.float32))
        ptr.append(ptr[-1] + int(etype_src.numel()))

    packed_src = (
        torch.cat(packed_src_chunks, dim=0)
        if len(packed_src_chunks) > 0
        else empty
    ).contiguous()
    packed_dst = (
        torch.cat(packed_dst_chunks, dim=0)
        if len(packed_dst_chunks) > 0
        else empty
    ).contiguous()
    packed_ptr = torch.tensor(ptr, dtype=torch.int64, device="cpu").contiguous()
    packed_ew: Optional[torch.Tensor] = None
    if has_ew:
        packed_ew = (
            torch.cat(packed_ew_chunks, dim=0)
            if len(packed_ew_chunks) > 0
            else empty_ew
        ).contiguous()
    return packed_src, packed_dst, packed_ptr, packed_ew


def hierhgt_dti_collate_fn(batch):
    """
    Collate function:
    - Graphs are merged by dgl.batch.
    - Local typed edges are offset+packed into batch-global indices.
    - Protein edge weights are packed alongside edges.
    """
    if any(item is None for item in batch):
        raise RuntimeError("Detected None sample in batch under strict mode.")

    drug_graph_list = [item["drug_graph"] for item in batch]
    protein_graph_list = [item["protein_graph"] for item in batch]
    drug_graphs = dgl.batch(drug_graph_list)
    protein_graphs = dgl.batch(protein_graph_list)

    if "drug_typed_local" not in batch[0] or "protein_typed_local" not in batch[0]:
        raise KeyError(
            "Batch is missing local typed edges. "
            "Ensure dataset.__getitem__ returns drug_typed_local/protein_typed_local."
        )

    num_drug_etypes = len(batch[0]["drug_typed_local"])
    num_protein_etypes = len(batch[0]["protein_typed_local"])
    if num_drug_etypes <= 0 or num_protein_etypes <= 0:
        raise ValueError(
            f"Invalid etype bucket sizes in batch: drug={num_drug_etypes}, protein={num_protein_etypes}."
        )

    drug_offsets = _build_node_offsets(drug_graph_list)
    protein_offsets = _build_node_offsets(protein_graph_list)
    drug_edges_src, drug_edges_dst, drug_edges_ptr, _ = _pack_typed_edges(
        [item["drug_typed_local"] for item in batch],
        drug_offsets,
        expected_num_etypes=num_drug_etypes,
        edge_name="drug_typed_local",
    )
    protein_edges_src, protein_edges_dst, protein_edges_ptr, protein_edges_ew = _pack_typed_edges(
        [item["protein_typed_local"] for item in batch],
        protein_offsets,
        expected_num_etypes=num_protein_etypes,
        edge_name="protein_typed_local",
        local_typed_ew_list=[item.get("protein_typed_ew") for item in batch],
    )
    protein_other_drop_ratio = torch.tensor(
        [float(item["protein_other_drop_ratio"]) for item in batch],
        dtype=torch.float32,
    )

    labels = torch.stack([item["label"] for item in batch]).squeeze(-1)
    drug_ids = torch.tensor([item["drug_id"] for item in batch], dtype=torch.long)
    protein_ids = torch.tensor([item["protein_id"] for item in batch], dtype=torch.long)

    typed_edge_batch = {
        "drug_edges_src": drug_edges_src,
        "drug_edges_dst": drug_edges_dst,
        "drug_edges_ptr": drug_edges_ptr,
        "protein_edges_src": protein_edges_src,
        "protein_edges_dst": protein_edges_dst,
        "protein_edges_ptr": protein_edges_ptr,
        "protein_edges_ew": protein_edges_ew,
        "protein_other_drop_ratio": protein_other_drop_ratio,
    }

    # --- SubPocket: extract atom_to_sub with per-sample sub-ID offsets ---
    atom_to_sub_list = []
    sub_counts = []
    sub_offset = 0
    for i, g in enumerate(drug_graph_list):
        if 'atom_to_sub' in g.ndata:
            local_map = g.ndata['atom_to_sub']
            atom_to_sub_list.append(local_map + sub_offset)
            n_subs = getattr(g, 'num_subs', int(local_map.max().item()) + 1)
        else:
            # Fallback: all atoms in one substructure
            n_atoms = g.num_nodes()
            atom_to_sub_list.append(torch.zeros(n_atoms, dtype=torch.long) + sub_offset)
            n_subs = 1
        sub_counts.append(n_subs)
        sub_offset += n_subs

    if atom_to_sub_list:
        batch_atom_to_sub = torch.cat(atom_to_sub_list, dim=0)
    else:
        batch_atom_to_sub = torch.zeros(0, dtype=torch.long)
    batch_sub_counts = torch.tensor(sub_counts, dtype=torch.long)
    total_subs = sub_offset

    # --- SubPocket: extract res_to_pocket with per-sample pocket-ID offsets ---
    res_to_pocket_list = []
    pocket_counts = []
    pocket_offset = 0
    for i, g in enumerate(protein_graph_list):
        if 'res_to_pocket' in g.ndata:
            local_map = g.ndata['res_to_pocket']
            res_to_pocket_list.append(local_map + pocket_offset)
            n_pockets = getattr(g, 'num_pockets', int(local_map.max().item()) + 1)
        else:
            # Fallback: all residues in one pocket
            n_res = g.num_nodes()
            res_to_pocket_list.append(torch.zeros(n_res, dtype=torch.long) + pocket_offset)
            n_pockets = 1
        pocket_counts.append(n_pockets)
        pocket_offset += n_pockets

    if res_to_pocket_list:
        batch_res_to_pocket = torch.cat(res_to_pocket_list, dim=0)
    else:
        batch_res_to_pocket = torch.zeros(0, dtype=torch.long)
    batch_pocket_counts = torch.tensor(pocket_counts, dtype=torch.long)
    total_pockets = pocket_offset

    typed_edge_batch['atom_to_sub'] = batch_atom_to_sub
    typed_edge_batch['sub_counts'] = batch_sub_counts
    typed_edge_batch['total_subs'] = total_subs
    typed_edge_batch['res_to_pocket'] = batch_res_to_pocket
    typed_edge_batch['pocket_counts'] = batch_pocket_counts
    typed_edge_batch['total_pockets'] = total_pockets

    return (
        drug_graphs,
        protein_graphs,
        labels,
        drug_ids,
        protein_ids,
        typed_edge_batch,
    )

import logging
from typing import Any, Dict, List, Optional, Tuple

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    # Package import path.
    from .packed_hgt_layers import (
        PackedJointHGT,
        is_torch_scatter_available,
        scatter_add,
        scatter_max,
    )
    from .encoders import SharedNodeStem
except ImportError:
    # Direct script execution path.
    from packed_hgt_layers import (
        PackedJointHGT,
        is_torch_scatter_available,
        scatter_add,
        scatter_max,
    )
    from encoders import SharedNodeStem

try:
    from torch_scatter import scatter_mean as _scatter_mean
    _HAS_SCATTER_MEAN = True
except ImportError:
    _scatter_mean = None
    _HAS_SCATTER_MEAN = False

logger = logging.getLogger("HierHGT-DTI-Model")


class HierarchicalAggregator(nn.Module):
    """Aggregate child-level embeddings into parent super-nodes.

    Used for: atoms → substructures, residues → pockets.

    Supports three aggregation modes:
      - ``mean``:  scatter_mean (simple average per group).
      - ``attention``: group-mean query × per-child key → segmented softmax
        → weighted sum.  Learns *which* children matter most for each group.
      - ``gated``: per-child sigmoid gate (element-wise) → scatter_add / size.
        Learns *which feature dimensions* to keep, independently per child.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        aggregation: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.aggregation = aggregation.lower()
        if self.aggregation not in {"mean", "attention", "gated"}:
            raise ValueError(
                f"HierarchicalAggregator: aggregation must be one of "
                f"['mean', 'attention', 'gated'], got '{self.aggregation}'."
            )
        self.eps = float(eps)

        # --- mode-specific layers ---
        if self.aggregation == "attention":
            self.attn_query = nn.Linear(hidden_dim, hidden_dim)
            self.attn_key = nn.Linear(hidden_dim, hidden_dim)
            self.attn_scale = hidden_dim ** -0.5

        elif self.aggregation == "gated":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )

        # shared post-processing
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _segmented_softmax(
        scores: torch.Tensor,       # [N_child]
        segment_ids: torch.Tensor,  # [N_child]  int64
        num_segments: int,
        eps: float,
    ) -> torch.Tensor:
        """Numerically-stable softmax within each segment (group)."""
        if scores.numel() == 0:
            return scores
        if scatter_add is None or scatter_max is None:
            raise RuntimeError(
                "HierarchicalAggregator(attention) requires torch_scatter."
            )
        seg = segment_ids.long()
        s_fp32 = scores.float()
        max_per_seg, _ = scatter_max(s_fp32, seg, dim=0, dim_size=num_segments)
        shifted = s_fp32 - max_per_seg.gather(0, seg)
        exp_s = torch.exp(shifted)
        sum_per_seg = scatter_add(exp_s, seg, dim=0, dim_size=num_segments)
        denom = sum_per_seg.gather(0, seg).clamp_min(eps)
        return (exp_s / denom).to(dtype=scores.dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        child_emb: torch.Tensor,       # [N_child, d]
        child_to_parent: torch.Tensor,  # [N_child] int64, parent id per child
        num_parents: int,
    ) -> torch.Tensor:
        """Returns parent embeddings ``[num_parents, d]``."""
        if not _HAS_SCATTER_MEAN:
            raise ImportError(
                "HierarchicalAggregator requires torch_scatter (scatter_mean)."
            )

        if self.aggregation == "mean":
            parent_emb = _scatter_mean(
                child_emb, child_to_parent, dim=0, dim_size=num_parents,
            )

        elif self.aggregation == "attention":
            # 1) group mean as query
            group_mean = _scatter_mean(
                child_emb, child_to_parent, dim=0, dim_size=num_parents,
            )
            q = self.attn_query(group_mean)            # [N_groups, d]
            k = self.attn_key(child_emb)               # [N_child, d]

            # 2) per-child attention score
            q_per_child = q.index_select(0, child_to_parent)  # [N_child, d]
            attn_score = (q_per_child * k).sum(dim=-1) * self.attn_scale  # [N_child]

            # 3) segmented softmax within each group
            attn_weight = self._segmented_softmax(
                attn_score, child_to_parent, num_parents, self.eps,
            )  # [N_child]
            self._last_attn_weight = attn_weight.detach()

            # 4) weighted sum
            weighted = child_emb * attn_weight.unsqueeze(-1)  # [N_child, d]
            if scatter_add is None:
                raise RuntimeError(
                    "HierarchicalAggregator(attention) requires torch_scatter."
                )
            parent_emb = scatter_add(
                weighted, child_to_parent, dim=0, dim_size=num_parents,
            )

        elif self.aggregation == "gated":
            # per-child element-wise gate (independent, no cross-child competition)
            gate_val = self.gate(child_emb)                     # [N_child, d]
            gated = child_emb * gate_val                        # [N_child, d]
            if scatter_add is None:
                raise RuntimeError(
                    "HierarchicalAggregator(gated) requires torch_scatter."
                )
            parent_emb = scatter_add(
                gated, child_to_parent, dim=0, dim_size=num_parents,
            )
            # normalize by group size
            ones = child_emb.new_ones((child_emb.shape[0], 1))
            group_sizes = scatter_add(
                ones, child_to_parent, dim=0, dim_size=num_parents,
            ).clamp_min(1.0)
            parent_emb = parent_emb / group_sizes

        else:
            # Should be unreachable (caught in __init__).
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        parent_emb = self.norm(parent_emb)
        parent_emb = self.dropout(parent_emb)
        return parent_emb


class AffinityMapGenerator(nn.Module):
    """Build sparse substructure-pocket affinity outputs from relation attention."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        mid_dim = max(1, hidden_dim // 4)
        self.refine = nn.Sequential(
            nn.Linear(1, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        raw_attn: torch.Tensor,      # [E] or [E, H]
        edge_src: torch.Tensor,      # [E]
        edge_dst: torch.Tensor,      # [E]
        edge_batch: torch.Tensor,    # [E]
    ) -> Dict[str, torch.Tensor]:
        if raw_attn.ndim == 2:
            raw_attn = raw_attn.mean(dim=-1)
        if raw_attn.numel() == 0:
            refined = raw_attn
        else:
            refined = self.refine(raw_attn.unsqueeze(-1)).squeeze(-1)
        return {
            "edge_attn": refined,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_batch": edge_batch,
        }



class HierHGTDTIModel(nn.Module):
    """
    Atom-residue cross-edge-centric DTI model:
    - SharedNodeStem for node-space alignment
    - Two-stage atom-residue cross-edge router (Stage-1a/1b) builds sparse cross-modal edges
    - Joint packed typed-edge HGT propagates relation-aware intra/inter-modal messages
    - pair_mlp_r -> predictor_rel -> BCE loss + router_entropy_reg
    """

    def __init__(
        self,
        hidden_dim=64,
        n_heads=4,
        dropout=0.1,
        predictor_hidden_dims=None,
        predictor_dropout=0.3,
        predictor_mode: str = "single_global",
        predictor_token_proj_dim: Optional[int] = None,
        predictor_token_dropout: Optional[float] = None,
        predictor_fusion_gate_hidden_dim: int = 64,
        predictor_fusion_branch_dropout: float = 0.0,
        device=None,
        drug_encoder_config=None,
        protein_encoder_config=None,
        shared_stem_config=None,
        relational_view_config=None,
        strict_feature_dims=True,
        # Legacy kwargs accepted but ignored for config compatibility
        **_ignored,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.strict_feature_dims = bool(strict_feature_dims)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        drug_enc_cfg = drug_encoder_config or {}
        protein_enc_cfg = protein_encoder_config or {}
        stem_cfg = shared_stem_config or {}
        rel_view_cfg = relational_view_config or {}

        self.drug_input_dim = int(drug_enc_cfg.get("in_dim", 74))
        self.protein_input_dim = int(protein_enc_cfg.get("in_dim", 320))

        self.shared_encoder_type = str(stem_cfg.get("type", "mlp")).lower()
        if self.shared_encoder_type not in ["mlp", "gnn"]:
            self.shared_encoder_type = "mlp"
        stem_dropout = float(stem_cfg.get("dropout", dropout))

        self.atom_encoder = SharedNodeStem(
            in_dim=self.drug_input_dim,
            out_dim=hidden_dim,
            dropout=stem_dropout,
        )
        self.residue_encoder = SharedNodeStem(
            in_dim=self.protein_input_dim,
            out_dim=hidden_dim,
            dropout=stem_dropout,
        )

        self.protein_edge_type_dim = int(rel_view_cfg.get("protein_edge_type_dim", 3))
        self.protein_seq_gap_bin_dim = int(rel_view_cfg.get("protein_seq_gap_bin_dim", 4))
        self.protein_contact_bin_dim = int(rel_view_cfg.get("protein_contact_bin_dim", 5))
        self.protein_edge_feat_dim = (
            self.protein_edge_type_dim + self.protein_seq_gap_bin_dim + self.protein_contact_bin_dim
        )
        self.drug_edge_feat_dim = int(rel_view_cfg.get("drug_edge_feat_dim", 14))

        hgt_cfg = rel_view_cfg.get("hgt", {})
        self.n_layers = int(hgt_cfg.get("n_layers", hgt_cfg.get("n_layers_level2", 2)))
        self.protein_edge_granularity = str(rel_view_cfg.get("protein_edge_granularity", "coarse")).lower()
        if self.protein_edge_granularity not in {"coarse", "fine"}:
            raise ValueError(
                f"protein_edge_granularity must be one of ['coarse', 'fine'], got {self.protein_edge_granularity}"
            )
        self.coarse_other_policy = str(rel_view_cfg.get("coarse_other_policy", "drop")).lower()
        if self.coarse_other_policy not in {"drop", "map_to_near_weak"}:
            raise ValueError(
                f"coarse_other_policy must be one of ['drop', 'map_to_near_weak'], got {self.coarse_other_policy}"
            )
        # SubPocket config (replaces router config)
        subpocket_cfg = rel_view_cfg.get("subpocket", {})
        self.cross_edge_warmup_epochs = int(subpocket_cfg.get("cross_edge_warmup_epochs", 5))
        self.cross_edge_warmup_start = float(subpocket_cfg.get("cross_edge_warmup_start", 0.1))
        self.hier_aggregation = str(subpocket_cfg.get("aggregation", "mean")).lower()

        # Ablation switches
        self.disable_cross_edges = bool(subpocket_cfg.get("disable_cross_edges", False))
        self.disable_sub_pocket_cross = bool(subpocket_cfg.get("disable_sub_pocket_cross", False))
        self.disable_hierarchy = bool(subpocket_cfg.get("disable_hierarchy", False))
        self.no_shortcut = bool(subpocket_cfg.get("no_shortcut", False))
        self.mid_aggregation = bool(subpocket_cfg.get("mid_aggregation", False))
        flow_gate_cfg = subpocket_cfg.get("flow_gate", {})
        self.flow_gate_enabled = bool(flow_gate_cfg.get("enabled", False))
        self.flow_gate_hidden_dim = max(1, int(flow_gate_cfg.get("hidden_dim", hidden_dim)))
        self.flow_gate_init_bias = float(flow_gate_cfg.get("init_bias", -2.0))
        relation_msg_gate_cfg = subpocket_cfg.get("relation_msg_gate", {})
        self.relation_msg_gate_init_bias = float(relation_msg_gate_cfg.get("init_bias", 0.0))
        self.relation_msg_gate_temperature = float(relation_msg_gate_cfg.get("temperature", 1.0))
        self.relation_msg_gate_degree_scale = float(relation_msg_gate_cfg.get("degree_scale", 1.0))
        self.relation_msg_gate_freeze = bool(relation_msg_gate_cfg.get("freeze", False))

        # Custom edge type masking (ablation: permanently zero-out specific relations)
        self.custom_mask_edge_types = set(subpocket_cfg.get("mask_edge_types", []))

        if self.disable_hierarchy:
            self.flow_gate_enabled = False

        packed_backend_cfg = rel_view_cfg.get("packed_backend", {})
        self.packed_require_torch_scatter = bool(packed_backend_cfg.get("require_torch_scatter", True))
        self.packed_attn_eps = float(packed_backend_cfg.get("attn_eps", 1e-8))
        if self.packed_attn_eps <= 0.0:
            raise ValueError(f"packed_backend.attn_eps must be > 0, got {self.packed_attn_eps}.")
        self.rel_debug_checks = bool(rel_view_cfg.get("debug_checks", False))
        self._typed_edge_batch_schema_checked = False

        self.drug_use_bond_types = bool(rel_view_cfg.get("drug_use_bond_types", True))
        self.protein_use_seq_gap_types = bool(rel_view_cfg.get("protein_use_seq_gap_types", True))
        self.protein_use_contact_bin_types = bool(rel_view_cfg.get("protein_use_contact_bin_types", True))
        self.protein_coarse_cache_key = (
            "protein_etype_id_coarse_drop"
            if self.coarse_other_policy == "drop"
            else "protein_etype_id_coarse_map"
        )

        self.drug_ntype_dict = {"atom": 0, "drug": 1}
        (
            self.drug_homo_etypes,
            self.drug_etype_dict,
            self.drug_edge_feat_dims_by_etype,
        ) = self._build_drug_type_spec()
        if self.protein_edge_granularity == "coarse":
            (
                self.protein_homo_etypes,
                self.protein_etype_dict,
                self.protein_edge_feat_dims_by_etype,
            ) = self._build_protein_type_spec_coarse()
        else:
            (
                self.protein_homo_etypes,
                self.protein_etype_dict,
                self.protein_edge_feat_dims_by_etype,
            ) = self._build_protein_type_spec()
        self.drug_homo_etype_to_idx = {name: idx for idx, name in enumerate(self.drug_homo_etypes)}
        self.protein_homo_etype_to_idx = {name: idx for idx, name in enumerate(self.protein_homo_etypes)}
        self.protein_homo_etype_set = set(self.protein_homo_etypes)

        # ---- 6-node-type Joint HGT ----
        self.joint_drug_homo_etypes = list(self.drug_homo_etypes)
        self.joint_protein_homo_etypes = list(self.protein_homo_etypes)
        # Cross etypes: full bipartite sub↔pocket + drug↔protein
        self.joint_cross_etypes = [
            "sub_binds_pocket", "pocket_bound_by_sub",
            "interacts_with", "interacted_by",
        ]
        # 6 node types: atom(0), substructure(1), drug(2), residue(3), pocket(4), protein(5)
        self.joint_ntype_dict = {
            "atom": 0, "substructure": 1, "drug": 2,
            "residue": 3, "pocket": 4, "protein": 5,
        }
        # Hierarchical edges
        joint_etypes = (
            # Drug hierarchy
            ["belongs_to_drug"]            # atom -> drug
            + self.joint_drug_homo_etypes  # atom -> atom
            + ["contains_atom"]            # drug -> atom
            + ["atom_to_sub"]              # atom -> substructure
            + ["sub_to_atom"]              # substructure -> atom
            + ["sub_to_drug"]              # substructure -> drug
            + ["drug_to_sub"]              # drug -> substructure
            # Protein hierarchy
            + ["belongs_to_protein"]       # residue -> protein
            + self.joint_protein_homo_etypes  # residue -> residue
            + ["contains_residue"]         # protein -> residue
            + ["res_to_pocket"]            # residue -> pocket
            + ["pocket_to_res"]            # pocket -> residue
            + ["pocket_to_protein"]        # pocket -> protein
            + ["protein_to_pocket"]        # protein -> pocket
            # Cross-modal
            + self.joint_cross_etypes
        )
        self.joint_etype_dict = {name: idx for idx, name in enumerate(joint_etypes)}

        # --- Relational view modules ---
        self.protein_ntype_dict = {"residue": 0, "protein": 1}
        self.drug_super_node_emb = nn.Parameter(torch.empty(1, hidden_dim))
        self.protein_super_node_emb = nn.Parameter(torch.empty(1, hidden_dim))
        # Learnable embeddings for substructure and pocket super-nodes
        self.sub_super_node_emb = nn.Parameter(torch.empty(1, hidden_dim))
        self.pocket_super_node_emb = nn.Parameter(torch.empty(1, hidden_dim))

        # Hierarchical aggregators
        self.atom_to_sub_agg = HierarchicalAggregator(
            hidden_dim, dropout=dropout, aggregation=self.hier_aggregation,
        )
        self.res_to_pocket_agg = HierarchicalAggregator(
            hidden_dim, dropout=dropout, aggregation=self.hier_aggregation,
        )
        self.affinity_map_generator = AffinityMapGenerator(hidden_dim)
        if self.flow_gate_enabled:
            self.hier_flow_gate = nn.Sequential(
                nn.Linear(hidden_dim * 4, self.flow_gate_hidden_dim),
                nn.GELU(),
                nn.Linear(self.flow_gate_hidden_dim, 1),
            )
        else:
            self.hier_flow_gate = None

        if self.packed_require_torch_scatter and (not is_torch_scatter_available()):
            raise ImportError(
                "Packed backend requires torch-scatter, but torch_scatter is not available."
            )
        if self.mid_aggregation and self.flow_gate_enabled:
            logger.warning(
                "subpocket.flow_gate is not supported together with mid_aggregation; disabling flow gate."
            )
            self.flow_gate_enabled = False
            self.hier_flow_gate = None
        if self.mid_aggregation and self.n_layers >= 2 and not self.disable_hierarchy:
            # Stage-1 HGT: 1 layer, only atom/residue/drug/protein (no sub/pocket)
            self.hgt_stage1 = PackedJointHGT(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=n_heads,
                num_layers=1,
                dropout=dropout,
                use_norm=True,
                joint_ntype_dict=self.joint_ntype_dict,
                joint_etype_dict=self.joint_etype_dict,
                eps=self.packed_attn_eps,
                require_torch_scatter=self.packed_require_torch_scatter,
                return_attn_relations=[],
                relation_gate_init_bias=self.relation_msg_gate_init_bias,
                relation_gate_temperature=self.relation_msg_gate_temperature,
                relation_gate_degree_scale=self.relation_msg_gate_degree_scale,
                relation_gate_freeze=self.relation_msg_gate_freeze,
            )
            # Stage-2 HGT: remaining layers, full 6-type graph
            self.hgt_stage2 = PackedJointHGT(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=n_heads,
                num_layers=self.n_layers - 1,
                dropout=dropout,
                use_norm=True,
                joint_ntype_dict=self.joint_ntype_dict,
                joint_etype_dict=self.joint_etype_dict,
                eps=self.packed_attn_eps,
                require_torch_scatter=self.packed_require_torch_scatter,
                return_attn_relations=[],
                relation_gate_init_bias=self.relation_msg_gate_init_bias,
                relation_gate_temperature=self.relation_msg_gate_temperature,
                relation_gate_degree_scale=self.relation_msg_gate_degree_scale,
                relation_gate_freeze=self.relation_msg_gate_freeze,
            )
            self.hgt = None  # not used in mid_aggregation mode
            # Mid-aggregation aggregators (reuse same architecture)
            self.mid_atom_to_sub_agg = HierarchicalAggregator(
                hidden_dim, dropout=dropout, aggregation=self.hier_aggregation,
            )
            self.mid_res_to_pocket_agg = HierarchicalAggregator(
                hidden_dim, dropout=dropout, aggregation=self.hier_aggregation,
            )
        else:
            self.hgt = PackedJointHGT(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=n_heads,
                num_layers=self.n_layers,
                dropout=dropout,
                use_norm=True,
                joint_ntype_dict=self.joint_ntype_dict,
                joint_etype_dict=self.joint_etype_dict,
                eps=self.packed_attn_eps,
                require_torch_scatter=self.packed_require_torch_scatter,
                return_attn_relations=[],
                relation_gate_init_bias=self.relation_msg_gate_init_bias,
                relation_gate_temperature=self.relation_msg_gate_temperature,
                relation_gate_degree_scale=self.relation_msg_gate_degree_scale,
                relation_gate_freeze=self.relation_msg_gate_freeze,
            )
            self.hgt_stage1 = None
            self.hgt_stage2 = None
        # Pair MLP for relational super-node global representations.
        self.pair_mlp_r = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Predictor modes ---
        self.predictor_mode = str(predictor_mode).lower()
        if self.predictor_mode not in {"single_global", "single_token", "fusion"}:
            raise ValueError(
                f"predictor_mode must be one of ['single_global', 'single_token', 'fusion'], got {self.predictor_mode}."
            )
        self.predictor_token_proj_dim = int(hidden_dim if predictor_token_proj_dim is None else predictor_token_proj_dim)
        if self.predictor_token_proj_dim <= 0:
            raise ValueError(f"predictor_token_proj_dim must be > 0, got {self.predictor_token_proj_dim}.")
        self.predictor_fusion_gate_hidden_dim = max(1, int(predictor_fusion_gate_hidden_dim))
        self.predictor_fusion_branch_dropout = float(predictor_fusion_branch_dropout)
        if not (0.0 <= self.predictor_fusion_branch_dropout < 1.0):
            raise ValueError(
                f"predictor_fusion_branch_dropout must be in [0, 1), got {self.predictor_fusion_branch_dropout}."
            )

        # --- Predictor ---
        if predictor_hidden_dims is None:
            predictor_hidden_dims = [128, 64, 32]
        predictor_token_dropout = predictor_dropout if predictor_token_dropout is None else float(predictor_token_dropout)

        def _build_predictor_head(input_dim: int) -> nn.Sequential:
            layers: List[nn.Module] = []
            cur_dim = input_dim
            for hidden_dim_layer in predictor_hidden_dims:
                layers.append(nn.Linear(cur_dim, hidden_dim_layer))
                layers.append(nn.LayerNorm(hidden_dim_layer))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(predictor_dropout))
                cur_dim = hidden_dim_layer
            layers.append(nn.Linear(cur_dim, 1))
            return nn.Sequential(*layers)

        # Global pair predictor (existing default branch)
        self.predictor_rel = _build_predictor_head(hidden_dim)
        # Token-level bilinear predictor branch
        self.token_atom_proj = nn.Linear(hidden_dim, self.predictor_token_proj_dim)
        self.token_res_proj = nn.Linear(hidden_dim, self.predictor_token_proj_dim)
        self.token_pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(predictor_token_dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predictor_token = _build_predictor_head(hidden_dim)
        # Residual fusion gate: token branch predicts a gated correction on top of the global branch.
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, self.predictor_fusion_gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(self.predictor_fusion_gate_hidden_dim, 1),
        )

        self._init_parameters()

        logger.info("HierHGTDTIModel initialized (SubPocket-HGT)")
        logger.info(f"  - Shared encoder: {self.shared_encoder_type}")
        logger.info(f"  - Drug typed etypes: {len(self.drug_etype_dict)}")
        logger.info(f"  - Protein typed etypes: {len(self.protein_etype_dict)}")
        logger.info(f"  - Joint etypes: {len(self.joint_etype_dict)}")
        logger.info(f"  - Joint ntypes: {len(self.joint_ntype_dict)} (6-node-type SubPocket)")
        logger.info(
            "  - Cross edge warmup: start=%.2f, epochs=%d",
            self.cross_edge_warmup_start,
            self.cross_edge_warmup_epochs,
        )
        logger.info(f"  - Hierarchical aggregation: {self.hier_aggregation}")
        logger.info(
            "  - Ablation: disable_cross_edges=%s, disable_sub_pocket_cross=%s, disable_hierarchy=%s, no_shortcut=%s, mid_aggregation=%s, flow_gate=%s",
            self.disable_cross_edges,
            self.disable_sub_pocket_cross,
            self.disable_hierarchy,
            self.no_shortcut,
            self.mid_aggregation,
            self.flow_gate_enabled,
        )
        logger.info(
            "  - Relation message aggregation: freeze=%s, init_bias=%.2f, temperature=%.2f, degree_scale=%.2f",
            self.relation_msg_gate_freeze,
            self.relation_msg_gate_init_bias,
            self.relation_msg_gate_temperature,
            self.relation_msg_gate_degree_scale,
        )
        logger.info(
            "  - Predictor mode: %s (fusion_branch_dropout=%.2f)",
            self.predictor_mode,
            self.predictor_fusion_branch_dropout,
        )
        logger.info(f"  - Protein granularity: {self.protein_edge_granularity}, other_policy={self.coarse_other_policy}")
        logger.info(
            "  - Rel execution backend: packed-only (torch_scatter_available=%s)",
            is_torch_scatter_available(),
        )
        logger.info(f"  - Strict feature dims: {self.strict_feature_dims}")

    @staticmethod
    def _kwargs_from_config(model_config: Dict[str, Any], device=None) -> Dict[str, Any]:
        model_cfg = model_config or {}
        hgt_cfg = model_cfg.get("relational_view", {}).get("hgt", {})
        rel_cfg = model_cfg.get("relational_view", {})
        predictor_cfg = model_cfg.get("predictor", {})
        token_bilinear_cfg = predictor_cfg.get("token_bilinear", {})
        fusion_cfg = predictor_cfg.get("fusion", {})
        shared_stem_cfg = model_cfg.get("shared_stem", {})

        return {
            "hidden_dim": hgt_cfg.get("hidden_dim", 128),
            "n_heads": hgt_cfg.get("n_heads", 8),
            "dropout": hgt_cfg.get("dropout", 0.1),
            "predictor_hidden_dims": predictor_cfg.get("hidden_dims", [128, 64, 32]),
            "predictor_dropout": predictor_cfg.get("dropout", 0.3),
            "predictor_mode": predictor_cfg.get("mode", "single_global"),
            "predictor_token_proj_dim": token_bilinear_cfg.get("proj_dim", None),
            "predictor_token_dropout": token_bilinear_cfg.get("dropout", None),
            "predictor_fusion_gate_hidden_dim": fusion_cfg.get("gate_hidden_dim", 64),
            "predictor_fusion_branch_dropout": fusion_cfg.get("branch_dropout", 0.0),
            "device": device,
            "drug_encoder_config": model_cfg.get("drug_encoder", None),
            "protein_encoder_config": model_cfg.get("protein_encoder", None),
            "shared_stem_config": shared_stem_cfg,
            "relational_view_config": rel_cfg,
            "strict_feature_dims": model_cfg.get("strict_feature_dims", True),
        }

    @classmethod
    def from_config(cls, model_config: Dict[str, Any], device=None):
        return cls(**cls._kwargs_from_config(model_config=model_config, device=device))

    def _model_device(self) -> torch.device:
        """Return the actual runtime device of model parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            # Fallback for edge cases (e.g., empty module during tests).
            return self.device

    def _build_drug_type_spec(self) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
        if self.drug_use_bond_types:
            homo_etypes = [
                "bond_single",
                "bond_double",
                "bond_triple",
                "bond_aromatic",
                "bond_other",
                "atom_self_loop",
            ]
        else:
            homo_etypes = ["bond", "atom_self_loop"]

        all_etypes = ["belongs_to_drug"] + homo_etypes + ["contains_atom"]
        etype_dict = {name: idx for idx, name in enumerate(all_etypes)}
        edge_feat_dims_by_etype = {name: self.drug_edge_feat_dim for name in homo_etypes}
        return homo_etypes, etype_dict, edge_feat_dims_by_etype

    def _build_protein_type_spec(self) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
        homo_etypes = []
        if self.protein_use_seq_gap_types:
            max_gap_bin = max(1, self.protein_seq_gap_bin_dim - 1)
            homo_etypes.extend([f"seq_gap_{k}" for k in range(1, max_gap_bin + 1)])
            homo_etypes.append("seq_gap_other")
        if self.protein_use_contact_bin_types:
            max_contact_bin = max(1, self.protein_contact_bin_dim - 1)
            homo_etypes.extend([f"contact_bin_{k}" for k in range(1, max_contact_bin + 1)])
            homo_etypes.append("contact_bin_other")
        if not self.protein_use_seq_gap_types and not self.protein_use_contact_bin_types:
            homo_etypes.append("sequence")

        homo_etypes.append("residue_self_loop")
        all_etypes = ["belongs_to_protein"] + homo_etypes + ["contains_residue"]
        etype_dict = {name: idx for idx, name in enumerate(all_etypes)}
        edge_feat_dims_by_etype = {name: self.protein_edge_feat_dim for name in homo_etypes}
        return homo_etypes, etype_dict, edge_feat_dims_by_etype

    def _build_protein_type_spec_coarse(self) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
        homo_etypes = [
            "seq_adjacent",
            "seq_near",
            "contact_weak",
            "contact_strong",
            "residue_self_loop",
        ]
        all_etypes = ["belongs_to_protein"] + homo_etypes + ["contains_residue"]
        etype_dict = {name: idx for idx, name in enumerate(all_etypes)}
        edge_feat_dims_by_etype = {name: self.protein_edge_feat_dim for name in homo_etypes}
        return homo_etypes, etype_dict, edge_feat_dims_by_etype

    def _align_feature_dim(
        self,
        feats: torch.Tensor,
        expected_dim: int,
        feature_name: str = "feature",
    ) -> torch.Tensor:
        if feats.shape[-1] == expected_dim:
            return feats
        if self.strict_feature_dims:
            raise ValueError(
                f"{feature_name} dim mismatch: expected {expected_dim}, got {feats.shape[-1]}. "
                "Set strict_feature_dims=False to allow legacy truncate/pad fallback."
            )
        if feats.shape[-1] > expected_dim:
            return feats[:, :expected_dim]
        pad = feats.new_zeros((feats.shape[0], expected_dim - feats.shape[-1]))
        return torch.cat([feats, pad], dim=-1)

    @staticmethod
    def _create_batch_ids(graph: dgl.DGLGraph) -> torch.Tensor:
        if graph.batch_size == 1:
            return torch.zeros(graph.num_nodes(), dtype=torch.long, device=graph.device)
        batch_num_nodes = graph.batch_num_nodes()
        if not isinstance(batch_num_nodes, torch.Tensor):
            batch_num_nodes = torch.tensor(
                [int(v) for v in batch_num_nodes],
                device=graph.device,
                dtype=torch.int64,
            )
        else:
            batch_num_nodes = batch_num_nodes.to(device=graph.device, dtype=torch.int64)
        return torch.repeat_interleave(
            torch.arange(int(batch_num_nodes.shape[0]), device=graph.device),
            batch_num_nodes,
        )

    @staticmethod
    def _batch_num_nodes_tensor(graph: dgl.DGLGraph, device: torch.device) -> torch.Tensor:
        counts = graph.batch_num_nodes()
        if isinstance(counts, torch.Tensor):
            return counts.to(device=device, dtype=torch.int64)
        return torch.tensor([int(v) for v in counts], device=device, dtype=torch.int64)

    def _get_graph_node_feat(self, graph: dgl.DGLGraph, expected_dim: int, graph_name: str) -> torch.Tensor:
        if "h" not in graph.ndata:
            if self.strict_feature_dims:
                raise KeyError(
                    f"{graph_name} is missing node feature 'h'. "
                    "Set strict_feature_dims=False to allow zero-filled fallback."
                )
            return torch.zeros(graph.num_nodes(), expected_dim, device=graph.device, dtype=torch.float32)
        return self._align_feature_dim(
            graph.ndata["h"].float(),
            expected_dim,
            feature_name=f"{graph_name}.ndata['h']",
        )

    def _get_graph_edge_feat(self, graph: dgl.DGLGraph, expected_dim: int, graph_name: str) -> torch.Tensor:
        if "edge_feat" not in graph.edata:
            if graph.num_edges() == 0:
                return torch.zeros(0, expected_dim, device=graph.device, dtype=torch.float32)
            if self.strict_feature_dims:
                raise KeyError(
                    f"{graph_name} is missing edge feature 'edge_feat'. "
                    "Set strict_feature_dims=False to allow zero-filled fallback."
                )
            return torch.zeros(graph.num_edges(), expected_dim, device=graph.device, dtype=torch.float32)
        edge_feat = graph.edata["edge_feat"].float()
        return self._align_feature_dim(
            edge_feat,
            expected_dim,
            feature_name=f"{graph_name}.edata['edge_feat']",
        )

    def _shared_encode(
        self,
        graph: dgl.DGLGraph,
        feats: torch.Tensor,
        encoder: nn.Module,
    ) -> torch.Tensor:
        if self.shared_encoder_type == "mlp":
            return encoder(feats)
        return encoder(graph, feats)

    def _decode_drug_homo_etype_indices(self, src: torch.Tensor, dst: torch.Tensor, edge_feat: torch.Tensor):
        num_edges = src.shape[0]
        if num_edges == 0:
            return torch.zeros(0, dtype=torch.long, device=src.device)

        fallback_idx = self.drug_homo_etype_to_idx.get("bond_other", 0)
        etype_indices = torch.full((num_edges,), fallback_idx, dtype=torch.long, device=src.device)

        # Prioritize self-loop assignment.
        self_loop_mask = (src == dst)
        if edge_feat is not None and edge_feat.numel() > 0 and edge_feat.size(-1) > 13:
            self_loop_mask = self_loop_mask | (edge_feat[:, 13] > 0.5)

        if "atom_self_loop" in self.drug_homo_etype_to_idx:
            etype_indices[self_loop_mask] = self.drug_homo_etype_to_idx["atom_self_loop"]

        non_self = ~self_loop_mask
        if non_self.any() and edge_feat is not None and edge_feat.numel() > 0:
            bond_type_dim = min(4, edge_feat.size(-1))
            bond_scores = edge_feat[non_self, :bond_type_dim]                       # [E', K]
            has_signal = (bond_scores.abs().sum(dim=-1) > 0)                        # [E']
            bond_idx = bond_scores.argmax(dim=-1)                                   # [E']

            bond_names = ["bond_single", "bond_double", "bond_triple", "bond_aromatic"]
            mapped = torch.full_like(bond_idx, fallback_idx)
            for i, name in enumerate(bond_names[:bond_type_dim]):
                if name in self.drug_homo_etype_to_idx:
                    mapped[bond_idx == i] = self.drug_homo_etype_to_idx[name]

            etype_indices[non_self] = torch.where(has_signal, mapped, etype_indices[non_self])

        return etype_indices
    def _resolve_protein_feat_splits(self, feat_dim: int) -> Tuple[int, int, int]:
        edge_type_dim = min(self.protein_edge_type_dim, feat_dim)
        remaining = max(0, feat_dim - edge_type_dim)

        seq_dim = self.protein_seq_gap_bin_dim
        contact_dim = self.protein_contact_bin_dim
        if seq_dim + contact_dim != remaining:
            if remaining <= 1:
                seq_dim = remaining
                contact_dim = 0
            else:
                seq_dim = min(max(1, seq_dim), remaining - 1)
                contact_dim = remaining - seq_dim
        return edge_type_dim, seq_dim, contact_dim

    def _decode_protein_homo_etype_indices(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        num_edges = src.shape[0]
        if num_edges == 0:
            return torch.zeros(0, dtype=torch.long, device=src.device)

        self_loop_idx = self.protein_homo_etype_to_idx.get("residue_self_loop", 0)
        fallback_idx = self.protein_homo_etype_to_idx.get(
            "seq_gap_other",
            self.protein_homo_etype_to_idx.get(
                "contact_bin_other",
                self.protein_homo_etype_to_idx.get("sequence", self_loop_idx),
            ),
        )
        etype_indices = torch.full((num_edges,), fallback_idx, dtype=torch.long, device=src.device)

        edge_type_dim, seq_dim, contact_dim = self._resolve_protein_feat_splits(edge_feat.shape[-1])

        if edge_type_dim > 0:
            edge_type_idx = torch.argmax(edge_feat[:, :edge_type_dim], dim=1)
        else:
            edge_type_idx = torch.zeros(num_edges, dtype=torch.long, device=src.device)

        seq_start = edge_type_dim
        seq_end = seq_start + seq_dim
        contact_end = seq_end + contact_dim

        if seq_dim > 0:
            seq_gap_idx = torch.argmax(edge_feat[:, seq_start:seq_end], dim=1)
        else:
            seq_gap_idx = torch.zeros(num_edges, dtype=torch.long, device=src.device)

        if contact_dim > 0:
            contact_idx = torch.argmax(edge_feat[:, seq_end:contact_end], dim=1)
        else:
            contact_idx = torch.zeros(num_edges, dtype=torch.long, device=src.device)

        is_self_loop = (src == dst) | (edge_type_idx == 2)
        etype_indices[is_self_loop] = self_loop_idx

        seq_mask = (~is_self_loop) & (edge_type_idx == 0)
        if seq_mask.any():
            if self.protein_use_seq_gap_types:
                seq_other_idx = self.protein_homo_etype_to_idx.get("seq_gap_other", fallback_idx)
                seq_lookup = torch.full((max(1, seq_dim),), seq_other_idx, dtype=torch.long, device=src.device)
                for k in range(1, seq_dim):
                    etype_name = f"seq_gap_{k}"
                    if etype_name in self.protein_homo_etype_to_idx:
                        seq_lookup[k] = self.protein_homo_etype_to_idx[etype_name]
                seq_values = seq_lookup[seq_gap_idx.clamp(max=seq_lookup.shape[0] - 1)]
                etype_indices[seq_mask] = seq_values[seq_mask]
            else:
                etype_indices[seq_mask] = self.protein_homo_etype_to_idx.get("sequence", fallback_idx)

        contact_mask = (~is_self_loop) & (edge_type_idx == 1)
        if contact_mask.any():
            if self.protein_use_contact_bin_types:
                contact_other_idx = self.protein_homo_etype_to_idx.get("contact_bin_other", fallback_idx)
                contact_lookup = torch.full(
                    (max(1, contact_dim),),
                    contact_other_idx,
                    dtype=torch.long,
                    device=src.device,
                )
                for k in range(1, contact_dim):
                    etype_name = f"contact_bin_{k}"
                    if etype_name in self.protein_homo_etype_to_idx:
                        contact_lookup[k] = self.protein_homo_etype_to_idx[etype_name]
                contact_values = contact_lookup[contact_idx.clamp(max=contact_lookup.shape[0] - 1)]
                etype_indices[contact_mask] = contact_values[contact_mask]
            else:
                etype_indices[contact_mask] = self.protein_homo_etype_to_idx.get("sequence", fallback_idx)

        return etype_indices

    def _decode_protein_homo_etype_indices_coarse(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        num_edges = src.shape[0]
        if num_edges == 0:
            return torch.zeros(0, dtype=torch.long, device=src.device)

        invalid = -1
        out = torch.full((num_edges,), invalid, dtype=torch.long, device=src.device)
        edge_type_dim, seq_dim, contact_dim = self._resolve_protein_feat_splits(edge_feat.shape[-1])
        edge_type_idx = (
            torch.argmax(edge_feat[:, :edge_type_dim], dim=1)
            if edge_type_dim > 0
            else torch.zeros(num_edges, dtype=torch.long, device=src.device)
        )
        seq_start = edge_type_dim
        seq_end = seq_start + seq_dim
        contact_end = seq_end + contact_dim
        seq_gap_idx = (
            torch.argmax(edge_feat[:, seq_start:seq_end], dim=1)
            if seq_dim > 0
            else torch.zeros(num_edges, dtype=torch.long, device=src.device)
        )
        contact_idx = (
            torch.argmax(edge_feat[:, seq_end:contact_end], dim=1)
            if contact_dim > 0
            else torch.zeros(num_edges, dtype=torch.long, device=src.device)
        )

        self_loop_mask = (src == dst) | (edge_type_idx == 2)
        out[self_loop_mask] = self.protein_homo_etype_to_idx["residue_self_loop"]

        seq_mask = (~self_loop_mask) & (edge_type_idx == 0)
        out[seq_mask & (seq_gap_idx == 1)] = self.protein_homo_etype_to_idx["seq_adjacent"]
        out[seq_mask & ((seq_gap_idx == 2) | (seq_gap_idx == 3))] = self.protein_homo_etype_to_idx["seq_near"]
        if self.coarse_other_policy == "map_to_near_weak":
            out[seq_mask & (seq_gap_idx == 0)] = self.protein_homo_etype_to_idx["seq_near"]

        contact_mask = (~self_loop_mask) & (edge_type_idx == 1)
        out[contact_mask & ((contact_idx == 1) | (contact_idx == 2))] = self.protein_homo_etype_to_idx["contact_weak"]
        out[contact_mask & ((contact_idx == 3) | (contact_idx == 4))] = self.protein_homo_etype_to_idx["contact_strong"]
        if self.coarse_other_policy == "map_to_near_weak":
            out[contact_mask & (contact_idx == 0)] = self.protein_homo_etype_to_idx["contact_weak"]

        return out

    def _maybe_validate_typed_edge_batch(
        self,
        typed_edge_batch: Dict[str, Any],
        device: torch.device,
        bsz: int,
    ) -> None:
        if (not self.rel_debug_checks) and self._typed_edge_batch_schema_checked:
            return

        if not isinstance(typed_edge_batch, dict):
            raise ValueError("typed_edge_batch must be a dict.")
        required_keys = (
            "drug_edges_src",
            "drug_edges_dst",
            "drug_edges_ptr",
            "protein_edges_src",
            "protein_edges_dst",
            "protein_edges_ptr",
            "protein_other_drop_ratio",
        )
        missing_keys = [k for k in required_keys if k not in typed_edge_batch]
        if missing_keys:
            raise ValueError(f"typed_edge_batch missing required packed keys: {missing_keys}")

        drug_edges_src = typed_edge_batch["drug_edges_src"]
        drug_edges_dst = typed_edge_batch["drug_edges_dst"]
        drug_edges_ptr = typed_edge_batch["drug_edges_ptr"]
        protein_edges_src = typed_edge_batch["protein_edges_src"]
        protein_edges_dst = typed_edge_batch["protein_edges_dst"]
        protein_edges_ptr = typed_edge_batch["protein_edges_ptr"]
        protein_other_drop_ratio = typed_edge_batch["protein_other_drop_ratio"]

        for key, tensor in (
            ("drug_edges_src", drug_edges_src),
            ("drug_edges_dst", drug_edges_dst),
            ("drug_edges_ptr", drug_edges_ptr),
            ("protein_edges_src", protein_edges_src),
            ("protein_edges_dst", protein_edges_dst),
            ("protein_edges_ptr", protein_edges_ptr),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"typed_edge_batch['{key}'] must be a Tensor.")
            if tensor.dtype != torch.int64:
                raise ValueError(f"typed_edge_batch['{key}'] must be int64, got {tensor.dtype}.")
            if tensor.ndim != 1:
                raise ValueError(f"typed_edge_batch['{key}'] must be 1D, got shape={tuple(tensor.shape)}.")
            if tensor.device != device:
                raise ValueError(
                    f"typed_edge_batch['{key}'] must already be on {device}, got {tensor.device}."
                )

        if not isinstance(protein_other_drop_ratio, torch.Tensor):
            raise ValueError("typed_edge_batch['protein_other_drop_ratio'] must be a Tensor.")
        if protein_other_drop_ratio.dtype != torch.float32:
            raise ValueError(
                f"typed_edge_batch['protein_other_drop_ratio'] must be float32, got {protein_other_drop_ratio.dtype}."
            )
        if protein_other_drop_ratio.ndim != 1:
            raise ValueError(
                "typed_edge_batch['protein_other_drop_ratio'] must be 1D, "
                f"got shape={tuple(protein_other_drop_ratio.shape)}."
            )
        if protein_other_drop_ratio.device != device:
            raise ValueError(
                f"typed_edge_batch['protein_other_drop_ratio'] must already be on {device}, got {protein_other_drop_ratio.device}."
            )
        if int(protein_other_drop_ratio.shape[0]) != int(bsz):
            raise ValueError(
                f"protein_other_drop_ratio size mismatch: {protein_other_drop_ratio.shape[0]} vs {bsz}."
            )

        if drug_edges_src.numel() != drug_edges_dst.numel():
            raise ValueError(
                f"drug packed src/dst mismatch: {drug_edges_src.numel()} vs {drug_edges_dst.numel()}."
            )
        if protein_edges_src.numel() != protein_edges_dst.numel():
            raise ValueError(
                f"protein packed src/dst mismatch: {protein_edges_src.numel()} vs {protein_edges_dst.numel()}."
            )
        if int(drug_edges_ptr.numel()) != int(len(self.drug_homo_etypes) + 1):
            raise ValueError(
                f"drug_edges_ptr length mismatch: {drug_edges_ptr.numel()} vs {len(self.drug_homo_etypes) + 1}."
            )
        if int(protein_edges_ptr.numel()) != int(len(self.protein_homo_etypes) + 1):
            raise ValueError(
                f"protein_edges_ptr length mismatch: {protein_edges_ptr.numel()} vs {len(self.protein_homo_etypes) + 1}."
            )
        if int(drug_edges_ptr[0].item()) != 0 or int(protein_edges_ptr[0].item()) != 0:
            raise ValueError("Packed ptr tensors must start from 0.")
        if bool((drug_edges_ptr[1:] < drug_edges_ptr[:-1]).any().item()):
            raise ValueError("drug_edges_ptr must be monotonic non-decreasing.")
        if bool((protein_edges_ptr[1:] < protein_edges_ptr[:-1]).any().item()):
            raise ValueError("protein_edges_ptr must be monotonic non-decreasing.")
        if int(drug_edges_ptr[-1].item()) != int(drug_edges_src.numel()):
            raise ValueError(
                f"drug_edges_ptr[-1] mismatch: {drug_edges_ptr[-1].item()} vs {drug_edges_src.numel()}."
            )
        if int(protein_edges_ptr[-1].item()) != int(protein_edges_src.numel()):
            raise ValueError(
                f"protein_edges_ptr[-1] mismatch: {protein_edges_ptr[-1].item()} vs {protein_edges_src.numel()}."
            )

        if not self.rel_debug_checks:
            self._typed_edge_batch_schema_checked = True

    @staticmethod
    def _hierarchy_edge_types() -> set:
        return {
            "atom_to_sub",
            "sub_to_atom",
            "sub_to_drug",
            "drug_to_sub",
            "res_to_pocket",
            "pocket_to_res",
            "pocket_to_protein",
            "protein_to_pocket",
            "sub_binds_pocket",
            "pocket_bound_by_sub",
        }

    def _relation_gate_modules(self) -> List[PackedJointHGT]:
        return [module for module in (self.hgt_stage1, self.hgt_stage2, self.hgt) if module is not None]

    def _collect_relation_msg_gate_tensors(self) -> Dict[str, Any]:
        zero = self.drug_super_node_emb.new_zeros((), dtype=torch.float32)
        budget_stats_by_relation: Dict[str, List[Tuple[torch.Tensor, int]]] = {}
        # Collect per-dst_type negative entropy from forward pass for diagnostics.
        entropy_values: List[Tuple[torch.Tensor, int]] = []
        hierarchy_mass_values: List[Tuple[torch.Tensor, int]] = []
        all_prior_values: List[torch.Tensor] = []
        has_any_gate_module = False
        for module in self._relation_gate_modules():
            for layer in module.layers:
                logits = getattr(layer, "relation_gate_logits", None)
                if logits is None:
                    continue
                has_any_gate_module = True
                all_prior_values.append(torch.sigmoid(logits.float()))
                # Collect per-dst_type negative entropy from the latest forward pass.
                ent_map = getattr(layer, "_last_relation_budget_entropy", {})
                if isinstance(ent_map, dict):
                    for dst_type, val in ent_map.items():
                        if isinstance(val, tuple) and len(val) == 2:
                            neg_ent_sum, cnt = val
                            if isinstance(neg_ent_sum, torch.Tensor) and cnt > 0:
                                entropy_values.append((neg_ent_sum.float(), cnt))
                hierarchy_mass_map = getattr(layer, "_last_relation_hierarchy_mass", {})
                if isinstance(hierarchy_mass_map, dict):
                    for _, val in hierarchy_mass_map.items():
                        if isinstance(val, tuple) and len(val) == 2:
                            hier_sum, cnt = val
                            if isinstance(hier_sum, torch.Tensor) and cnt > 0:
                                hierarchy_mass_values.append((hier_sum.float(), cnt))
                # Collect per-relation realized budget sums/counts (detached, for diagnostics).
                budget_map = getattr(layer, "_last_relation_gate_stats", {})
                if isinstance(budget_map, dict):
                    for edge_type, val in budget_map.items():
                        if isinstance(val, tuple) and len(val) == 2:
                            rel_sum, cnt = val
                            if isinstance(rel_sum, torch.Tensor) and cnt > 0:
                                budget_stats_by_relation.setdefault(edge_type, []).append((rel_sum.float(), cnt))

        if not has_any_gate_module:
            return {
                "relation_budget_entropy": zero,
                "relation_msg_gate_l1": zero,
                "relation_msg_gate_mean": zero,
                "relation_msg_gate_min": zero,
                "relation_msg_gate_max": zero,
                "relation_msg_gate_hierarchy_mean": zero,
                "relation_msg_gate_hierarchy_min": zero,
                "relation_msg_gate_hierarchy_max": zero,
                "relation_msg_gate_value_map": {},
                "relation_msg_gate_value_stat_map": {},
                "relation_msg_gate_hierarchy_mass_stat": (zero, 0),
            }

        relation_msg_gate_l1 = (
            torch.cat([gate_val.reshape(-1) for gate_val in all_prior_values]).mean()
            if all_prior_values
            else zero
        )

        # Diagnostic only: node-weighted mean entropy of realized relation budgets.
        if entropy_values:
            total_neg_ent = sum(s for s, _ in entropy_values)
            total_count = sum(c for _, c in entropy_values)
            weighted_neg_entropy = total_neg_ent / max(total_count, 1)
            entropy_value = -weighted_neg_entropy
        else:
            entropy_value = zero

        budget_mean_map: Dict[str, torch.Tensor] = {}
        budget_stat_map: Dict[str, Any] = {}
        for edge_type, stat_list in budget_stats_by_relation.items():
            total_budget = sum(rel_sum for rel_sum, _ in stat_list)
            total_count = sum(cnt for _, cnt in stat_list)
            budget_stat_map[edge_type] = (total_budget.detach(), total_count)
            budget_mean_map[edge_type] = total_budget / max(total_count, 1)

        value_map = budget_mean_map
        value_list = list(value_map.values())
        if value_list:
            value_stack = torch.stack(value_list).detach()
            gate_mean = value_stack.mean()
            gate_min = value_stack.min()
            gate_max = value_stack.max()
        else:
            gate_mean = zero
            gate_min = zero
            gate_max = zero

        if hierarchy_mass_values:
            total_hier_budget = sum(hier_sum for hier_sum, _ in hierarchy_mass_values)
            total_hier_count = sum(cnt for _, cnt in hierarchy_mass_values)
            hierarchy_mean = (total_hier_budget / max(total_hier_count, 1)).detach()
            hierarchy_unit_values = torch.stack(
                [(hier_sum / max(cnt, 1)).detach() for hier_sum, cnt in hierarchy_mass_values]
            )
            hierarchy_min = hierarchy_unit_values.min()
            hierarchy_max = hierarchy_unit_values.max()
        else:
            hierarchy_mean = zero
            hierarchy_min = zero
            hierarchy_max = zero
            total_hier_budget = zero
            total_hier_count = 0

        return {
            "relation_budget_entropy": entropy_value,
            "relation_msg_gate_l1": relation_msg_gate_l1,
            "relation_msg_gate_mean": gate_mean,
            "relation_msg_gate_min": gate_min,
            "relation_msg_gate_max": gate_max,
            "relation_msg_gate_hierarchy_mean": hierarchy_mean,
            "relation_msg_gate_hierarchy_min": hierarchy_min,
            "relation_msg_gate_hierarchy_max": hierarchy_max,
            "relation_msg_gate_value_map": value_map,
            "relation_msg_gate_value_stat_map": budget_stat_map,
            "relation_msg_gate_hierarchy_mass_stat": (total_hier_budget, total_hier_count),
        }

    def get_relation_msg_gate_value_map(self) -> Dict[str, float]:
        gate_stats = self._collect_relation_msg_gate_tensors()
        gate_map = gate_stats.get("relation_msg_gate_value_map", {})
        return {edge_type: float(gate_val.detach().item()) for edge_type, gate_val in gate_map.items()}

    def get_relation_msg_gate_value_stat_map(self) -> Dict[str, Tuple[float, int]]:
        gate_stats = self._collect_relation_msg_gate_tensors()
        stat_map = gate_stats.get("relation_msg_gate_value_stat_map", {})
        result: Dict[str, Tuple[float, int]] = {}
        for edge_type, val in stat_map.items():
            if isinstance(val, tuple) and len(val) == 2:
                rel_sum, cnt = val
                if isinstance(rel_sum, torch.Tensor):
                    result[edge_type] = (float(rel_sum.detach().item()), int(cnt))
        return result

    def get_relation_msg_gate_hierarchy_mass_stat(self) -> Tuple[float, int]:
        gate_stats = self._collect_relation_msg_gate_tensors()
        val = gate_stats.get("relation_msg_gate_hierarchy_mass_stat", None)
        if isinstance(val, tuple) and len(val) == 2:
            hier_sum, cnt = val
            if isinstance(hier_sum, torch.Tensor):
                return float(hier_sum.detach().item()), int(cnt)
        return 0.0, 0

    @staticmethod
    def _capture_rng_state(device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cpu_state = torch.get_rng_state()
        cuda_state = None
        if device.type == "cuda":
            cuda_state = torch.cuda.get_rng_state(device)
        return cpu_state, cuda_state

    @staticmethod
    def _restore_rng_state(
        cpu_state: torch.Tensor,
        cuda_state: Optional[torch.Tensor],
        device: torch.device,
    ) -> None:
        torch.set_rng_state(cpu_state)
        if device.type == "cuda" and cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)

    def _mask_relations(
        self,
        packed_relations: List[Dict[str, Any]],
        masked_edge_types: set,
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        empty = torch.zeros(0, dtype=torch.int64, device=device)
        masked_relations: List[Dict[str, Any]] = []
        for rel in packed_relations:
            if rel["edge_type"] in masked_edge_types:
                masked_relations.append(
                    {
                        **rel,
                        "src_idx": empty,
                        "dst_idx": empty,
                        "edge_weight": None,
                        "message_scale": None,
                    }
                )
            else:
                masked_relations.append(rel)
        return masked_relations

    def _run_single_stage_hgt(
        self,
        packed_relations: List[Dict[str, Any]],
        num_nodes_dict: Dict[str, int],
        drug_h_init: Dict[str, torch.Tensor],
        protein_h_init: Dict[str, torch.Tensor],
        target_relations: set,
    ) -> Tuple[Dict[str, torch.Tensor], Any]:
        if self.hgt is None:
            raise RuntimeError("Single-stage HGT backbone is required for this execution path.")
        for layer in self.hgt.layers:
            layer.return_attn_relations = target_relations
        out = self.hgt(
            packed_relations=packed_relations,
            num_nodes_by_type=num_nodes_dict,
            drug_h_init=drug_h_init,
            protein_h_init=protein_h_init,
        )
        return out, self.hgt.layers[-1]

    def _compute_hierarchical_gate(
        self,
        direct_h: torch.Tensor,
        full_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hier_flow_gate is None:
            raise RuntimeError("Hierarchical flow gate requested but not initialized.")
        delta = full_h - direct_h
        gate_input = torch.cat([direct_h, full_h, delta, torch.abs(delta)], dim=-1)
        gate = torch.sigmoid(self.hier_flow_gate(gate_input))
        mixed = direct_h + gate * delta
        return mixed, gate.squeeze(-1)

    def _build_joint_relations_batch(
        self,
        drug_graph: dgl.DGLGraph,
        protein_graph: dgl.DGLGraph,
        atom_emb: torch.Tensor,
        residue_emb: torch.Tensor,
        drug_super_init: torch.Tensor,
        protein_super_init: torch.Tensor,
        typed_edge_batch: Dict[str, Any],
        cross_edge_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """Build 6-node-type joint relations with hierarchical sub/pocket nodes.

        Node types: atom(0), substructure(1), drug(2), residue(3), pocket(4), protein(5)
        Cross edges: full bipartite between substructures and pockets.
        """
        device = atom_emb.device
        atom_counts_t = self._batch_num_nodes_tensor(drug_graph, device=device)
        res_counts_t = self._batch_num_nodes_tensor(protein_graph, device=device)
        bsz = int(atom_counts_t.shape[0])
        if bsz != int(res_counts_t.shape[0]):
            raise ValueError(f"Drug/protein batch mismatch: {bsz} vs {int(res_counts_t.shape[0])}")

        self._maybe_validate_typed_edge_batch(
            typed_edge_batch=typed_edge_batch,
            device=device,
            bsz=bsz,
        )
        drug_edges_src = typed_edge_batch["drug_edges_src"]
        drug_edges_dst = typed_edge_batch["drug_edges_dst"]
        drug_edges_ptr = typed_edge_batch["drug_edges_ptr"]
        protein_edges_src = typed_edge_batch["protein_edges_src"]
        protein_edges_dst = typed_edge_batch["protein_edges_dst"]
        protein_edges_ptr = typed_edge_batch["protein_edges_ptr"]
        protein_edges_ew = typed_edge_batch.get("protein_edges_ew", None)
        protein_other_drop_ratio = typed_edge_batch["protein_other_drop_ratio"]

        # --- SubPocket mappings from collate ---
        atom_to_sub = typed_edge_batch["atom_to_sub"].to(device=device, dtype=torch.long)
        sub_counts = typed_edge_batch["sub_counts"].to(device=device, dtype=torch.long)
        total_subs = int(typed_edge_batch["total_subs"])
        res_to_pocket = typed_edge_batch["res_to_pocket"].to(device=device, dtype=torch.long)
        pocket_counts = typed_edge_batch["pocket_counts"].to(device=device, dtype=torch.long)
        total_pockets = int(typed_edge_batch["total_pockets"])

        # --- Hierarchical aggregation (or bypass for ablation) ---
        if self.disable_hierarchy:
            # Ablation: no substructure/pocket intermediate nodes.
            # Force 0 sub/pocket nodes; only atom→drug & residue→protein hierarchy.
            total_subs = 0
            total_pockets = 0
            sub_emb = atom_emb.new_zeros((0, self.hidden_dim))
            pocket_emb = atom_emb.new_zeros((0, self.hidden_dim))
            sub_counts = torch.zeros(bsz, device=device, dtype=torch.long)
            pocket_counts = torch.zeros(bsz, device=device, dtype=torch.long)
        else:
            sub_emb = self.atom_to_sub_agg(atom_emb, atom_to_sub, total_subs)
            sub_emb = sub_emb + self.sub_super_node_emb
            pocket_emb = self.res_to_pocket_agg(residue_emb, res_to_pocket, total_pockets)
            pocket_emb = pocket_emb + self.pocket_super_node_emb

        # --- Build global ID tensors ---
        atom_ids = torch.arange(atom_emb.shape[0], device=device, dtype=torch.int64)
        res_ids = torch.arange(residue_emb.shape[0], device=device, dtype=torch.int64)
        sub_ids = torch.arange(total_subs, device=device, dtype=torch.int64)
        pocket_ids = torch.arange(total_pockets, device=device, dtype=torch.int64)
        drug_ids = torch.arange(bsz, device=device, dtype=torch.int64)
        protein_ids_t = torch.arange(bsz, device=device, dtype=torch.int64)
        empty = torch.zeros(0, dtype=torch.int64, device=device)

        atom_batch_ids = self._create_batch_ids(drug_graph).to(device)
        res_batch_ids = self._create_batch_ids(protein_graph).to(device)

        # Sub batch IDs: which sample each substructure belongs to
        sub_batch_ids = torch.repeat_interleave(
            torch.arange(bsz, device=device, dtype=torch.long),
            sub_counts,
        )
        # Pocket batch IDs
        pocket_batch_ids = torch.repeat_interleave(
            torch.arange(bsz, device=device, dtype=torch.long),
            pocket_counts,
        )

        edges: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

        # --- Drug hierarchy edges ---
        if self.no_shortcut and not self.disable_hierarchy:
            edges[("atom", "belongs_to_drug", "drug")] = (empty, empty)
            edges[("drug", "contains_atom", "atom")] = (empty, empty)
        else:
            edges[("atom", "belongs_to_drug", "drug")] = (atom_ids, atom_batch_ids)
            edges[("drug", "contains_atom", "atom")] = (atom_batch_ids, atom_ids)
        if self.disable_hierarchy:
            edges[("atom", "atom_to_sub", "substructure")] = (empty, empty)
            edges[("substructure", "sub_to_atom", "atom")] = (empty, empty)
            edges[("substructure", "sub_to_drug", "drug")] = (empty, empty)
            edges[("drug", "drug_to_sub", "substructure")] = (empty, empty)
        else:
            edges[("atom", "atom_to_sub", "substructure")] = (atom_ids, atom_to_sub)
            edges[("substructure", "sub_to_atom", "atom")] = (atom_to_sub, atom_ids)
            edges[("substructure", "sub_to_drug", "drug")] = (sub_ids, sub_batch_ids)
            edges[("drug", "drug_to_sub", "substructure")] = (sub_batch_ids, sub_ids)

        # --- Protein hierarchy edges ---
        if self.no_shortcut and not self.disable_hierarchy:
            edges[("residue", "belongs_to_protein", "protein")] = (empty, empty)
            edges[("protein", "contains_residue", "residue")] = (empty, empty)
        else:
            edges[("residue", "belongs_to_protein", "protein")] = (res_ids, res_batch_ids)
            edges[("protein", "contains_residue", "residue")] = (res_batch_ids, res_ids)
        if self.disable_hierarchy:
            edges[("residue", "res_to_pocket", "pocket")] = (empty, empty)
            edges[("pocket", "pocket_to_res", "residue")] = (empty, empty)
            edges[("pocket", "pocket_to_protein", "protein")] = (empty, empty)
            edges[("protein", "protein_to_pocket", "pocket")] = (empty, empty)
        else:
            edges[("residue", "res_to_pocket", "pocket")] = (res_ids, res_to_pocket)
            edges[("pocket", "pocket_to_res", "residue")] = (res_to_pocket, res_ids)
            edges[("pocket", "pocket_to_protein", "protein")] = (pocket_ids, pocket_batch_ids)
            edges[("protein", "protein_to_pocket", "pocket")] = (pocket_batch_ids, pocket_ids)

        # --- Drug/Protein super-node cross edges ---
        if not self.disable_cross_edges:
            edges[("drug", "interacts_with", "protein")] = (drug_ids, protein_ids_t)
            edges[("protein", "interacted_by", "drug")] = (protein_ids_t, drug_ids)
        else:
            edges[("drug", "interacts_with", "protein")] = (empty, empty)
            edges[("protein", "interacted_by", "drug")] = (empty, empty)

        # --- Drug homogeneous edges (atom-atom bonds) ---
        for idx, name in enumerate(self.drug_homo_etypes):
            st = int(drug_edges_ptr[idx].item())
            ed = int(drug_edges_ptr[idx + 1].item())
            edges[("atom", name, "atom")] = (drug_edges_src[st:ed], drug_edges_dst[st:ed])

        # --- Protein homogeneous edges (residue-residue) ---
        protein_homo_ew: Dict[str, Optional[torch.Tensor]] = {}
        for idx, name in enumerate(self.protein_homo_etypes):
            st = int(protein_edges_ptr[idx].item())
            ed = int(protein_edges_ptr[idx + 1].item())
            edges[("residue", name, "residue")] = (protein_edges_src[st:ed], protein_edges_dst[st:ed])
            if protein_edges_ew is not None and ed > st:
                protein_homo_ew[name] = protein_edges_ew[st:ed].unsqueeze(-1)
            else:
                protein_homo_ew[name] = None

        # --- Cross-modal edges: full bipartite sub ↔ pocket per sample ---
        if self.disable_cross_edges or self.disable_sub_pocket_cross:
            cross_src = empty
            cross_dst = empty
            cross_batch = empty
        else:
            cross_src_list = []
            cross_dst_list = []
            sub_offset = 0
            pocket_offset = 0
            for b in range(bsz):
                n_sub_b = int(sub_counts[b].item())
                n_pocket_b = int(pocket_counts[b].item())
                if n_sub_b > 0 and n_pocket_b > 0:
                    # Full bipartite: each sub connects to all pockets in this sample
                    s_local = torch.arange(n_sub_b, device=device, dtype=torch.int64) + sub_offset
                    p_local = torch.arange(n_pocket_b, device=device, dtype=torch.int64) + pocket_offset
                    # Cartesian product
                    s_rep = s_local.repeat_interleave(n_pocket_b)
                    p_rep = p_local.repeat(n_sub_b)
                    cross_src_list.append(s_rep)
                    cross_dst_list.append(p_rep)
                sub_offset += n_sub_b
                pocket_offset += n_pocket_b

            if cross_src_list:
                cross_src = torch.cat(cross_src_list, dim=0)
                cross_dst = torch.cat(cross_dst_list, dim=0)
                cross_batch = sub_batch_ids.index_select(0, cross_src)
            else:
                cross_src = empty
                cross_dst = empty
                cross_batch = empty

        edges[("substructure", "sub_binds_pocket", "pocket")] = (cross_src, cross_dst)
        edges[("pocket", "pocket_bound_by_sub", "substructure")] = (cross_dst, cross_src)

        # --- Build num_nodes_dict for 6 node types ---
        num_nodes_dict = {
            "atom": int(atom_emb.shape[0]),
            "substructure": total_subs,
            "drug": bsz,
            "residue": int(residue_emb.shape[0]),
            "pocket": total_pockets,
            "protein": bsz,
        }

        # --- Pack relations for HGT ---
        edge_type_to_canonical = {
            edge_type: (src_type, edge_type, dst_type)
            for src_type, edge_type, dst_type in edges
        }

        # Compute cross edge weight for warmup
        cross_weight_val = max(1e-6, float(cross_edge_weight))
        cross_weight_tensor = atom_emb.new_tensor(cross_weight_val, dtype=torch.float32)
        if cross_src.numel() > 0:
            cross_ew = cross_weight_tensor.expand(cross_src.shape[0]).unsqueeze(-1)
        else:
            cross_ew = atom_emb.new_zeros((0, 1), dtype=torch.float32)

        packed_relations: List[Dict[str, Any]] = []
        for edge_type, _ in sorted(self.joint_etype_dict.items(), key=lambda x: x[1]):
            canonical = edge_type_to_canonical.get(edge_type, None)
            if canonical is None:
                continue
            src_type, _, dst_type = canonical
            src_idx, dst_idx = edges[canonical]
            if edge_type in {"sub_binds_pocket", "pocket_bound_by_sub"}:
                edge_weight = None
                message_scale = cross_ew
            elif edge_type in protein_homo_ew:
                edge_weight = protein_homo_ew[edge_type]
                message_scale = None
            else:
                edge_weight = None
                message_scale = None
            packed_relations.append(
                {
                    "src_type": src_type,
                    "edge_type": edge_type,
                    "dst_type": dst_type,
                    "src_idx": src_idx,
                    "dst_idx": dst_idx,
                    "edge_weight": edge_weight,
                    "message_scale": message_scale,
                }
            )

        drug_h = drug_super_init + self.drug_super_node_emb
        protein_h = protein_super_init + self.protein_super_node_emb

        _zero = atom_emb.new_zeros((), dtype=torch.float32)
        return {
            "packed_relations": packed_relations,
            "num_nodes_dict": num_nodes_dict,
            "atom_batch_ids": atom_batch_ids,
            "res_batch_ids": res_batch_ids,
            "sub_batch_ids": sub_batch_ids,
            "pocket_batch_ids": pocket_batch_ids,
            "drug_ids": drug_ids,
            "protein_ids": protein_ids_t,
            "atom_emb": atom_emb,
            "residue_emb": residue_emb,
            "sub_emb": sub_emb,
            "pocket_emb": pocket_emb,
            "drug_h": drug_h,
            "protein_h": protein_h,
            "protein_other_drop_ratio": protein_other_drop_ratio,
            "cross_edge_weight": cross_weight_tensor,
            "protein_homo_ew": protein_homo_ew,
            "sub_pocket_edge_src": cross_src,
            "sub_pocket_edge_dst": cross_dst,
            "sub_pocket_edge_batch": cross_batch,
        }

    @staticmethod
    def _segmented_softmax_1d(
        logits: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: int,
        eps: float,
    ) -> torch.Tensor:
        if logits.ndim != 1 or segment_ids.ndim != 1:
            raise ValueError(
                f"Expected 1D logits/segment_ids, got {tuple(logits.shape)} and {tuple(segment_ids.shape)}."
            )
        if int(logits.shape[0]) != int(segment_ids.shape[0]):
            raise ValueError(
                f"Segmented-softmax shape mismatch: logits={tuple(logits.shape)}, segment_ids={tuple(segment_ids.shape)}."
            )
        if int(num_segments) <= 0:
            raise ValueError(f"num_segments must be > 0, got {num_segments}.")
        if logits.numel() == 0:
            return logits
        if scatter_add is None or scatter_max is None:
            raise RuntimeError("torch_scatter is required for token segmented softmax.")

        seg_idx = segment_ids.to(dtype=torch.long)
        score_fp32 = logits.float()
        max_per_seg, _ = scatter_max(score_fp32, seg_idx, dim=0, dim_size=int(num_segments))
        score_shifted = score_fp32 - max_per_seg.gather(0, seg_idx)
        exp_score = torch.exp(score_shifted)
        sum_per_seg = scatter_add(exp_score, seg_idx, dim=0, dim_size=int(num_segments))
        denom = sum_per_seg.gather(0, seg_idx).clamp_min(float(eps))
        prob = exp_score / denom
        return prob.to(dtype=logits.dtype)

    def _interact_token(
        self,
        rel_out: Dict[str, torch.Tensor],
        drug_graph: dgl.DGLGraph,
        protein_graph: dgl.DGLGraph,
    ) -> Dict[str, torch.Tensor]:
        """Token-level interaction using substructure ↔ pocket HGT outputs.

        Instead of router-selected atom-residue cross edges, we use
        sub_tok_r_all and pocket_tok_r_all from HGT as the cross tokens.
        Each (sub, pocket) pair from the same sample forms a token pair.
        """
        sub_tok = rel_out["sub_tok_r_all"]       # [total_subs, d]
        pocket_tok = rel_out["pocket_tok_r_all"]  # [total_pockets, d]
        sub_batch_ids = rel_out["sub_batch_ids"]  # [total_subs]
        pocket_batch_ids = rel_out["pocket_batch_ids"]  # [total_pockets]
        device = sub_tok.device

        atom_counts_t = self._batch_num_nodes_tensor(drug_graph, device=device)
        res_counts_t = self._batch_num_nodes_tensor(protein_graph, device=device)
        batch_size = int(atom_counts_t.shape[0])

        z_token_pair = sub_tok.new_zeros((batch_size, self.hidden_dim))
        token_valid_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)

        # Build sub-pocket token pairs per sample
        # For each sample b: all (sub_i, pocket_j) pairs → bilinear features → weighted aggregate
        total_subs = int(sub_tok.shape[0])
        total_pockets = int(pocket_tok.shape[0])

        if total_subs > 0 and total_pockets > 0:
            # Build full bipartite (sub, pocket) token pairs per sample
            # Use scatter to efficiently gather sub and pocket indices per sample
            pair_sub_list = []
            pair_pocket_list = []
            pair_batch_list = []

            # We use a vectorized approach:
            # For each sub s with batch b, pair with all pockets in batch b
            for b in range(batch_size):
                sub_mask = (sub_batch_ids == b)
                pocket_mask = (pocket_batch_ids == b)
                sub_idx_b = sub_mask.nonzero(as_tuple=True)[0]
                pocket_idx_b = pocket_mask.nonzero(as_tuple=True)[0]
                n_sub = sub_idx_b.shape[0]
                n_pocket = pocket_idx_b.shape[0]
                if n_sub > 0 and n_pocket > 0:
                    # Cartesian product
                    s_rep = sub_idx_b.repeat_interleave(n_pocket)
                    p_rep = pocket_idx_b.repeat(n_sub)
                    pair_sub_list.append(s_rep)
                    pair_pocket_list.append(p_rep)
                    pair_batch_list.append(torch.full((n_sub * n_pocket,), b, device=device, dtype=torch.long))
                    token_valid_mask[b] = True

            if pair_sub_list:
                pair_sub = torch.cat(pair_sub_list, dim=0)
                pair_pocket = torch.cat(pair_pocket_list, dim=0)
                pair_batch = torch.cat(pair_batch_list, dim=0)

                sub_edge = sub_tok.index_select(0, pair_sub)
                pocket_edge = pocket_tok.index_select(0, pair_pocket)

                s_proj = self.token_atom_proj(sub_edge)
                p_proj = self.token_res_proj(pocket_edge)
                bilinear_scale = 1.0 / math.sqrt(float(self.predictor_token_proj_dim))
                edge_logits = (s_proj * p_proj).sum(dim=-1) * bilinear_scale

                edge_weight = self._segmented_softmax_1d(
                    logits=edge_logits.float(),
                    segment_ids=pair_batch,
                    num_segments=batch_size,
                    eps=self.packed_attn_eps,
                )

                edge_feat = torch.cat([sub_edge, pocket_edge, sub_edge * pocket_edge, torch.abs(sub_edge - pocket_edge)], dim=-1)
                edge_h = self.token_pair_mlp(edge_feat)
                edge_msg = edge_h.to(dtype=z_token_pair.dtype) * edge_weight.to(dtype=z_token_pair.dtype).unsqueeze(-1)
                z_token_pair.index_add_(0, pair_batch, edge_msg)

        logits_token = self.predictor_token(z_token_pair).squeeze(-1)
        token_valid_ratio = token_valid_mask.float().mean()
        token_fallback_ratio = 1.0 - token_valid_ratio
        return {
            "z_token_pair": z_token_pair,
            "logits_token": logits_token,
            "token_valid_mask": token_valid_mask,
            "token_valid_ratio": token_valid_ratio,
            "token_fallback_ratio": token_fallback_ratio,
        }

    def encode_rel(
        self,
        drug_graph: dgl.DGLGraph,
        protein_graph: dgl.DGLGraph,
        atom_emb: Optional[torch.Tensor] = None,
        residue_emb: Optional[torch.Tensor] = None,
        typed_edge_batch: Optional[Dict[str, Any]] = None,
        cross_edge_weight: float = 1.0,
        return_affinity: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if atom_emb is None:
            drug_feats = self._get_graph_node_feat(drug_graph, self.drug_input_dim, "drug_graph")
            atom_emb = self._shared_encode(drug_graph, drug_feats, self.atom_encoder)

        if residue_emb is None:
            protein_feats = self._get_graph_node_feat(protein_graph, self.protein_input_dim, "protein_graph")
            residue_emb = self._shared_encode(protein_graph, protein_feats, self.residue_encoder)

        with drug_graph.local_scope():
            drug_graph.ndata["_tmp"] = atom_emb
            drug_super_init = dgl.mean_nodes(drug_graph, "_tmp")
        with protein_graph.local_scope():
            protein_graph.ndata["_tmp"] = residue_emb
            protein_super_init = dgl.mean_nodes(protein_graph, "_tmp")

        if typed_edge_batch is None:
            raise ValueError("encode_rel requires typed_edge_batch.")

        packed_batch = self._build_joint_relations_batch(
            drug_graph=drug_graph,
            protein_graph=protein_graph,
            atom_emb=atom_emb,
            residue_emb=residue_emb,
            drug_super_init=drug_super_init,
            protein_super_init=protein_super_init,
            typed_edge_batch=typed_edge_batch,
            cross_edge_weight=cross_edge_weight,
        )

        # Build h_init for 6 node types
        drug_h_init = {
            "atom": atom_emb,
            "substructure": packed_batch["sub_emb"],
            "drug": packed_batch["drug_h"],
        }
        protein_h_init = {
            "residue": residue_emb,
            "pocket": packed_batch["pocket_emb"],
            "protein": packed_batch["protein_h"],
        }

        # Cache pre-HGT representations for analysis hooks
        self._pre_hgt_pocket = packed_batch["pocket_emb"].detach()
        self._pre_hgt_protein = packed_batch["protein_h"].detach()

        flow_gate_drug = None
        flow_gate_protein = None
        flow_gate_drug_direct = None
        flow_gate_protein_direct = None
        flow_gate_drug_full = None
        flow_gate_protein_full = None

        # Apply custom edge-type masking (ablation: permanently zero-out specific relations)
        if self.custom_mask_edge_types:
            packed_batch["packed_relations"] = self._mask_relations(
                packed_relations=packed_batch["packed_relations"],
                masked_edge_types=self.custom_mask_edge_types,
                device=atom_emb.device,
            )

        if self.hgt_stage1 is not None:
            # ── Mid-aggregation: two-stage HGT ──
            # Stage 1: zero out sub/pocket edges, only atom/residue/drug/protein propagate
            stage1_relations = self._mask_relations(
                packed_relations=packed_batch["packed_relations"],
                masked_edge_types=self._hierarchy_edge_types(),
                device=atom_emb.device,
            )

            # Stage 1 forward: sub/pocket get zero messages, atom/residue/drug/protein update
            stage1_out = self.hgt_stage1(
                packed_relations=stage1_relations,
                num_nodes_by_type=packed_batch["num_nodes_dict"],
                drug_h_init=drug_h_init,
                protein_h_init=protein_h_init,
            )

            # Mid-aggregation: re-aggregate sub/pocket from updated atom/residue
            atom_to_sub = typed_edge_batch["atom_to_sub"].to(device=atom_emb.device, dtype=torch.long)
            total_subs = int(typed_edge_batch["total_subs"])
            res_to_pocket = typed_edge_batch["res_to_pocket"].to(device=atom_emb.device, dtype=torch.long)
            total_pockets = int(typed_edge_batch["total_pockets"])

            mid_sub_emb = self.mid_atom_to_sub_agg(stage1_out["atom"], atom_to_sub, total_subs)
            mid_sub_emb = mid_sub_emb + self.sub_super_node_emb
            mid_pocket_emb = self.mid_res_to_pocket_agg(stage1_out["residue"], res_to_pocket, total_pockets)
            mid_pocket_emb = mid_pocket_emb + self.pocket_super_node_emb

            # Stage 2 h_init: overwrite sub/pocket, keep updated atom/residue/drug/protein
            stage2_drug_h_init = {
                "atom": stage1_out["atom"],
                "substructure": mid_sub_emb,
                "drug": stage1_out["drug"],
            }
            stage2_protein_h_init = {
                "residue": stage1_out["residue"],
                "pocket": mid_pocket_emb,
                "protein": stage1_out["protein"],
            }

            target_relations = {"sub_binds_pocket"} if bool(return_affinity) else set()
            for layer in self.hgt_stage2.layers:
                layer.return_attn_relations = target_relations

            out = self.hgt_stage2(
                packed_relations=packed_batch["packed_relations"],
                num_nodes_by_type=packed_batch["num_nodes_dict"],
                drug_h_init=stage2_drug_h_init,
                protein_h_init=stage2_protein_h_init,
            )

            # For affinity, use stage2's last layer
            _hgt_last_layer = self.hgt_stage2.layers[-1]
        else:
            # ── Original: single multi-layer HGT ──
            target_relations = {"sub_binds_pocket"} if bool(return_affinity) else set()
            if self.flow_gate_enabled:
                direct_relations = self._mask_relations(
                    packed_relations=packed_batch["packed_relations"],
                    masked_edge_types=self._hierarchy_edge_types(),
                    device=atom_emb.device,
                )
                rng_state = None
                if self.training:
                    rng_state = self._capture_rng_state(atom_emb.device)
                direct_out, _ = self._run_single_stage_hgt(
                    packed_relations=direct_relations,
                    num_nodes_dict=packed_batch["num_nodes_dict"],
                    drug_h_init=drug_h_init,
                    protein_h_init=protein_h_init,
                    target_relations=set(),
                )
                if rng_state is not None:
                    self._restore_rng_state(*rng_state, atom_emb.device)
                out, _hgt_last_layer = self._run_single_stage_hgt(
                    packed_relations=packed_batch["packed_relations"],
                    num_nodes_dict=packed_batch["num_nodes_dict"],
                    drug_h_init=drug_h_init,
                    protein_h_init=protein_h_init,
                    target_relations=target_relations,
                )
                flow_gate_drug_direct = direct_out["drug"].detach()
                flow_gate_protein_direct = direct_out["protein"].detach()
                flow_gate_drug_full = out["drug"].detach()
                flow_gate_protein_full = out["protein"].detach()
                out["drug"], flow_gate_drug = self._compute_hierarchical_gate(
                    direct_h=direct_out["drug"],
                    full_h=out["drug"],
                )
                out["protein"], flow_gate_protein = self._compute_hierarchical_gate(
                    direct_h=direct_out["protein"],
                    full_h=out["protein"],
                )
            else:
                out, _hgt_last_layer = self._run_single_stage_hgt(
                    packed_relations=packed_batch["packed_relations"],
                    num_nodes_dict=packed_batch["num_nodes_dict"],
                    drug_h_init=drug_h_init,
                    protein_h_init=protein_h_init,
                    target_relations=target_relations,
                )

        affinity_sparse = None
        if return_affinity:
            last_relation_attns = getattr(_hgt_last_layer, "_last_relation_attns", {})
            raw_sub_pocket_attn = last_relation_attns.get("sub_binds_pocket", None)
            num_cross_edges = int(packed_batch["sub_pocket_edge_src"].shape[0])
            if raw_sub_pocket_attn is None:
                raw_sub_pocket_attn = atom_emb.new_zeros((num_cross_edges,), dtype=torch.float32)
            if int(raw_sub_pocket_attn.shape[0]) != num_cross_edges:
                # Safety fallback for malformed relation-attention cache.
                raw_sub_pocket_attn = atom_emb.new_zeros((num_cross_edges,), dtype=torch.float32)

            affinity_sparse = self.affinity_map_generator(
                raw_attn=raw_sub_pocket_attn,
                edge_src=packed_batch["sub_pocket_edge_src"],
                edge_dst=packed_batch["sub_pocket_edge_dst"],
                edge_batch=packed_batch["sub_pocket_edge_batch"],
            )
        protein_other_drop_ratio = packed_batch["protein_other_drop_ratio"].mean()
        _zero = atom_emb.new_zeros((), dtype=torch.float32)
        flow_gate_drug_mean = flow_gate_drug.detach().mean() if flow_gate_drug is not None else _zero
        flow_gate_protein_mean = flow_gate_protein.detach().mean() if flow_gate_protein is not None else _zero
        flow_gate_drug_min = flow_gate_drug.detach().min() if flow_gate_drug is not None else _zero
        flow_gate_protein_min = flow_gate_protein.detach().min() if flow_gate_protein is not None else _zero
        flow_gate_drug_max = flow_gate_drug.detach().max() if flow_gate_drug is not None else _zero
        flow_gate_protein_max = flow_gate_protein.detach().max() if flow_gate_protein is not None else _zero
        relation_msg_gate_stats = self._collect_relation_msg_gate_tensors()
        return {
            "atom_tok_r_all": out["atom"],
            "sub_tok_r_all": out["substructure"],
            "res_tok_r_all": out["residue"],
            "pocket_tok_r_all": out["pocket"],
            "drug_rel_global": out["drug"],
            "protein_rel_global": out["protein"],
            "drug_rel_global_direct": flow_gate_drug_direct,
            "protein_rel_global_direct": flow_gate_protein_direct,
            "drug_rel_global_full": flow_gate_drug_full,
            "protein_rel_global_full": flow_gate_protein_full,
            "hier_flow_gate_drug": flow_gate_drug,
            "hier_flow_gate_protein": flow_gate_protein,
            "hier_flow_gate_drug_mean": flow_gate_drug_mean,
            "hier_flow_gate_protein_mean": flow_gate_protein_mean,
            "hier_flow_gate_drug_min": flow_gate_drug_min,
            "hier_flow_gate_protein_min": flow_gate_protein_min,
            "hier_flow_gate_drug_max": flow_gate_drug_max,
            "hier_flow_gate_protein_max": flow_gate_protein_max,
            "relation_budget_entropy": relation_msg_gate_stats["relation_budget_entropy"],
            "relation_msg_gate_l1": relation_msg_gate_stats["relation_msg_gate_l1"],
            "relation_msg_gate_mean": relation_msg_gate_stats["relation_msg_gate_mean"],
            "relation_msg_gate_min": relation_msg_gate_stats["relation_msg_gate_min"],
            "relation_msg_gate_max": relation_msg_gate_stats["relation_msg_gate_max"],
            "relation_msg_gate_hierarchy_mean": relation_msg_gate_stats["relation_msg_gate_hierarchy_mean"],
            "relation_msg_gate_hierarchy_min": relation_msg_gate_stats["relation_msg_gate_hierarchy_min"],
            "relation_msg_gate_hierarchy_max": relation_msg_gate_stats["relation_msg_gate_hierarchy_max"],
            "protein_other_drop_ratio": protein_other_drop_ratio,
            "cross_edge_weight": packed_batch["cross_edge_weight"],
            "sub_batch_ids": packed_batch["sub_batch_ids"],
            "pocket_batch_ids": packed_batch["pocket_batch_ids"],
            "affinity_sparse": affinity_sparse,
        }

    def interact_rel(
        self,
        rel_out: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        d_r = rel_out["drug_rel_global"]
        p_r = rel_out["protein_rel_global"]
        pairfeat_r = torch.cat([d_r, p_r, d_r * p_r, torch.abs(d_r - p_r)], dim=-1)
        z_r_pair = self.pair_mlp_r(pairfeat_r)
        return {"z_r_pair": z_r_pair}

    @staticmethod
    def _init_linear_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_parameters(self):
        init_roots = [
            self.pair_mlp_r,
            self.predictor_rel,
            self.token_atom_proj,
            self.token_res_proj,
            self.token_pair_mlp,
            self.predictor_token,
            self.fusion_gate_mlp,
            self.atom_to_sub_agg,
            self.res_to_pocket_agg,
            self.affinity_map_generator,
        ]
        if self.hier_flow_gate is not None:
            init_roots.append(self.hier_flow_gate)
        for root in init_roots:
            root.apply(self._init_linear_weights)

        nn.init.xavier_uniform_(self.drug_super_node_emb)
        nn.init.xavier_uniform_(self.protein_super_node_emb)
        nn.init.xavier_uniform_(self.sub_super_node_emb)
        nn.init.xavier_uniform_(self.pocket_super_node_emb)
        if self.hier_flow_gate is not None and self.hier_flow_gate[-1].bias is not None:
            nn.init.constant_(self.hier_flow_gate[-1].bias, self.flow_gate_init_bias)

    def forward(
        self,
        drug_graph,
        protein_graph,
        typed_edge_batch: Optional[Dict[str, Any]] = None,
        cross_edge_weight: float = 1.0,
        return_affinity: bool = False,
    ):
        model_device = self._model_device()
        if drug_graph.device != model_device:
            drug_graph = drug_graph.to(model_device)
        if protein_graph.device != model_device:
            protein_graph = protein_graph.to(model_device)

        drug_feats = self._get_graph_node_feat(drug_graph, self.drug_input_dim, "drug_graph")
        protein_feats = self._get_graph_node_feat(protein_graph, self.protein_input_dim, "protein_graph")

        atom_emb = self._shared_encode(drug_graph, drug_feats, self.atom_encoder)
        residue_emb = self._shared_encode(protein_graph, protein_feats, self.residue_encoder)

        if typed_edge_batch is None:
            raise ValueError("forward requires typed_edge_batch.")

        rel_out = self.encode_rel(
            drug_graph,
            protein_graph,
            atom_emb=atom_emb,
            residue_emb=residue_emb,
            typed_edge_batch=typed_edge_batch,
            cross_edge_weight=cross_edge_weight,
            return_affinity=return_affinity,
        )
        rel_interaction_out = self.interact_rel(rel_out)

        z_r_pair = rel_interaction_out["z_r_pair"]
        logits_global = self.predictor_rel(z_r_pair).squeeze(-1)

        logits_token: Optional[torch.Tensor] = None
        z_token_pair: Optional[torch.Tensor] = None
        token_valid_mask: Optional[torch.Tensor] = None
        token_valid_ratio: Optional[torch.Tensor] = None
        token_fallback_ratio: Optional[torch.Tensor] = None
        fusion_alpha_mean: Optional[torch.Tensor] = None
        fusion_alpha_std: Optional[torch.Tensor] = None
        fusion_alpha_min: Optional[torch.Tensor] = None
        fusion_alpha_max: Optional[torch.Tensor] = None

        if self.predictor_mode in {"single_token", "fusion"}:
            token_out = self._interact_token(rel_out=rel_out, drug_graph=drug_graph, protein_graph=protein_graph)
            logits_token = token_out["logits_token"]
            z_token_pair = token_out["z_token_pair"]
            token_valid_mask = token_out["token_valid_mask"]
            token_valid_ratio = token_out["token_valid_ratio"]
            token_fallback_ratio = token_out["token_fallback_ratio"]

        if self.predictor_mode == "single_global":
            logits = logits_global
            token_valid_ratio = logits_global.new_ones(())
            token_fallback_ratio = logits_global.new_zeros(())
        elif self.predictor_mode == "single_token":
            if logits_token is None or token_valid_mask is None:
                raise RuntimeError("single_token mode requires token branch outputs.")
            logits = torch.where(token_valid_mask, logits_token, logits_global)
        elif self.predictor_mode == "fusion":
            if logits_token is None or z_token_pair is None or token_valid_mask is None:
                raise RuntimeError("fusion mode requires token branch outputs.")

            fusion_feat = torch.cat(
                [z_r_pair, z_token_pair, z_r_pair * z_token_pair, torch.abs(z_r_pair - z_token_pair)],
                dim=-1,
            )
            alpha = torch.sigmoid(self.fusion_gate_mlp(fusion_feat).squeeze(-1))
            alpha = alpha * token_valid_mask.to(dtype=alpha.dtype)
            if self.training and self.predictor_fusion_branch_dropout > 0.0:
                keep_prob = 1.0 - float(self.predictor_fusion_branch_dropout)
                keep_token = (torch.rand_like(alpha) < keep_prob).to(dtype=alpha.dtype)
                alpha = alpha * keep_token
            logits = logits_global + alpha * logits_token
            alpha_detached = alpha.detach()
            fusion_alpha_mean = alpha_detached.mean()
            fusion_alpha_std = alpha_detached.std(unbiased=False)
            fusion_alpha_min = alpha_detached.min()
            fusion_alpha_max = alpha_detached.max()
        else:
            raise ValueError(f"Unsupported predictor_mode: {self.predictor_mode}")

        result = {
            "logits": logits,
            "logits_rel": logits_global,
            "logits_global": logits_global,
            "logits_token": logits_token,
            "z_r_pair": z_r_pair,
            "z_token_pair": z_token_pair,
            "predictor_mode": self.predictor_mode,
            "token_valid_ratio": token_valid_ratio,
            "token_fallback_ratio": token_fallback_ratio,
            "fusion_alpha_mean": fusion_alpha_mean,
            "fusion_alpha_std": fusion_alpha_std,
            "fusion_alpha_min": fusion_alpha_min,
            "fusion_alpha_max": fusion_alpha_max,
            "drug_rel_global": rel_out["drug_rel_global"],
            "protein_rel_global": rel_out["protein_rel_global"],
            "drug_rel_global_direct": rel_out["drug_rel_global_direct"],
            "protein_rel_global_direct": rel_out["protein_rel_global_direct"],
            "drug_rel_global_full": rel_out["drug_rel_global_full"],
            "protein_rel_global_full": rel_out["protein_rel_global_full"],
            "hier_flow_gate_drug": rel_out["hier_flow_gate_drug"],
            "hier_flow_gate_protein": rel_out["hier_flow_gate_protein"],
            "hier_flow_gate_drug_mean": rel_out["hier_flow_gate_drug_mean"],
            "hier_flow_gate_protein_mean": rel_out["hier_flow_gate_protein_mean"],
            "hier_flow_gate_drug_min": rel_out["hier_flow_gate_drug_min"],
            "hier_flow_gate_protein_min": rel_out["hier_flow_gate_protein_min"],
            "hier_flow_gate_drug_max": rel_out["hier_flow_gate_drug_max"],
            "hier_flow_gate_protein_max": rel_out["hier_flow_gate_protein_max"],
            "relation_budget_entropy": rel_out["relation_budget_entropy"],
            "relation_msg_gate_l1": rel_out["relation_msg_gate_l1"],
            "relation_msg_gate_mean": rel_out["relation_msg_gate_mean"],
            "relation_msg_gate_min": rel_out["relation_msg_gate_min"],
            "relation_msg_gate_max": rel_out["relation_msg_gate_max"],
            "relation_msg_gate_hierarchy_mean": rel_out["relation_msg_gate_hierarchy_mean"],
            "relation_msg_gate_hierarchy_min": rel_out["relation_msg_gate_hierarchy_min"],
            "relation_msg_gate_hierarchy_max": rel_out["relation_msg_gate_hierarchy_max"],
            "protein_other_drop_ratio": rel_out["protein_other_drop_ratio"],
            "cross_edge_weight": rel_out["cross_edge_weight"],
            "affinity_sparse": rel_out["affinity_sparse"],
        }
        return result

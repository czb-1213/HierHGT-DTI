"""
Packed HGT implementations for Dual-View DTI.
"""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_add, scatter_max

    _TORCH_SCATTER_AVAILABLE = True
except Exception:
    scatter_add = None
    scatter_max = None
    _TORCH_SCATTER_AVAILABLE = False


def is_torch_scatter_available() -> bool:
    return bool(_TORCH_SCATTER_AVAILABLE)


_HIERARCHY_EDGE_TYPES = {
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


class PackedHGTConv(nn.Module):
    """
    Relation-aware multi-head attention on packed edge tensors.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        num_ntypes: int,
        num_etypes: int,
        dropout: float = 0.2,
        use_norm: bool = True,
        eps: float = 1e-8,
        require_torch_scatter: bool = True,
        return_attn_relations: Optional[List[str]] = None,
        relation_gate_init_bias: float = 0.0,
        relation_gate_temperature: float = 1.0,
        relation_gate_degree_scale: float = 1.0,
        relation_gate_freeze: bool = False,
    ):
        super().__init__()

        self.require_torch_scatter = bool(require_torch_scatter)
        self._use_torch_scatter = bool(_TORCH_SCATTER_AVAILABLE)
        if self.require_torch_scatter and (not self._use_torch_scatter):
            raise ImportError(
                "PackedHGTConv requires torch_scatter. Install torch-scatter to use packed backend."
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_ntypes = num_ntypes
        self.num_etypes = num_etypes
        self.dropout = float(dropout)
        self.use_norm = bool(use_norm)
        self.eps = float(eps)
        self.return_attn_relations = set(return_attn_relations or [])
        self.relation_gate_init_bias = float(relation_gate_init_bias)
        # Relation-budget controls used by relation-level gated aggregation.
        self.relation_gate_temperature = float(relation_gate_temperature)
        self.relation_gate_degree_scale = float(relation_gate_degree_scale)
        self.relation_gate_freeze = bool(relation_gate_freeze)
        self._last_relation_attns: Dict[str, torch.Tensor] = {}
        self._last_relation_gates: Dict[str, torch.Tensor] = {}
        self._last_relation_gate_stats: Dict[str, Any] = {}
        self._last_relation_budget_entropy: Dict[str, torch.Tensor] = {}
        self._last_relation_hierarchy_mass: Dict[str, Any] = {}

        self.head_dim = out_dim // num_heads
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}.")

        self.Q = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_ntypes)])
        self.K = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_ntypes)])
        self.V = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_ntypes)])

        self.edge_W = nn.ParameterList(
            [nn.Parameter(torch.Tensor(num_heads, self.head_dim, self.head_dim)) for _ in range(num_etypes)]
        )

        self.O = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(num_ntypes)])
        if self.use_norm:
            self.norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(num_ntypes)])
        self.relation_gate_logits = nn.Parameter(torch.empty(num_etypes))
        if self.relation_gate_freeze:
            self.relation_gate_logits.requires_grad_(False)

        self._init_parameters()

    def _init_parameters(self):
        for w in self.edge_W:
            nn.init.xavier_uniform_(w)
        nn.init.constant_(self.relation_gate_logits, self.relation_gate_init_bias)

    def _segmented_softmax(
        self,
        score: torch.Tensor,
        dst_idx: torch.Tensor,
        dst_num_nodes: int,
    ) -> torch.Tensor:
        """
        score: [E, H], dst_idx: [E] (0..dst_num_nodes-1)
        """
        if score.numel() == 0:
            return score

        num_edges, num_heads = score.shape
        device = score.device
        dtype = score.dtype

        # Build (head, dst) segment ids on flattened scores.
        head_offset = (
            torch.arange(num_heads, device=device, dtype=torch.long).view(1, num_heads) * int(dst_num_nodes)
        )
        seg_idx = (dst_idx.view(-1, 1) + head_offset).reshape(-1)  # [E * H]
        flat_score = score.reshape(-1)

        seg_size = int(dst_num_nodes) * int(num_heads)
        if self._use_torch_scatter:
            max_per_seg, _ = scatter_max(flat_score, seg_idx, dim=0, dim_size=seg_size)
        else:
            max_per_seg = flat_score.new_full((seg_size,), torch.finfo(flat_score.dtype).min)
            if hasattr(max_per_seg, "scatter_reduce_"):
                max_per_seg.scatter_reduce_(0, seg_idx, flat_score, reduce="amax", include_self=True)
            else:
                # Last-resort fallback for very old torch versions without scatter_reduce_.
                for offset in range(int(seg_idx.numel())):
                    seg_id = int(seg_idx[offset].item())
                    max_per_seg[seg_id] = torch.maximum(max_per_seg[seg_id], flat_score[offset])
        max_per_edge = max_per_seg.gather(0, seg_idx)
        exp_score = torch.exp(flat_score - max_per_edge)
        if self._use_torch_scatter:
            sum_per_seg = scatter_add(exp_score, seg_idx, dim=0, dim_size=seg_size)
        else:
            sum_per_seg = flat_score.new_zeros((seg_size,))
            sum_per_seg.scatter_add_(0, seg_idx, exp_score)
        denom = sum_per_seg.gather(0, seg_idx).clamp_min(self.eps)

        weight = (exp_score / denom).view(num_edges, num_heads).to(dtype=dtype)
        return weight

    def forward(
        self,
        packed_relations: List[Dict[str, Any]],
        h_dict: Dict[str, torch.Tensor],
        ntype_dict: Dict[str, int],
        etype_dict: Dict[str, int],
        num_nodes_by_type: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(packed_relations, list):
            raise TypeError("packed_relations must be a list of relation dicts.")

        sample_tensor = next(iter(h_dict.values()))
        device = sample_tensor.device
        # Attention cache from the latest forward pass; used by affinity-map extraction.
        self._last_relation_attns = {}
        self._last_relation_gates = {}
        self._last_relation_gate_stats = {}
        self._last_relation_budget_entropy = {}
        self._last_relation_hierarchy_mass = {}
        relation_budget_temp = max(float(self.relation_gate_temperature), 1e-4)
        relation_budget_degree_scale = float(self.relation_gate_degree_scale)

        h_base = {}
        h_msg = {}
        relation_budget_state = {}
        relation_budget_meta = {}
        for ntype, h in h_dict.items():
            n_nodes = int(num_nodes_by_type[ntype])
            if h.shape[-1] == self.out_dim:
                base = h
            else:
                base = torch.zeros((n_nodes, self.out_dim), device=device, dtype=h.dtype)
            h_base[ntype] = base
            h_msg[ntype] = torch.zeros_like(base)
            rel_dtype = base.dtype
            neg_fill = torch.finfo(rel_dtype).min
            relation_budget_state[ntype] = {
                "m": torch.full((n_nodes, 1), neg_fill, device=device, dtype=rel_dtype),
                "s": torch.zeros((n_nodes, 1), device=device, dtype=rel_dtype),
                "z": torch.zeros((n_nodes, self.out_dim), device=device, dtype=rel_dtype),
                "has_any": torch.zeros((n_nodes, 1), device=device, dtype=torch.bool),
            }
            relation_budget_meta[ntype] = []

        src_types_active = set()
        dst_types_active = set()
        for rel in packed_relations:
            if int(rel["src_idx"].numel()) <= 0:
                continue
            src_types_active.add(rel["src_type"])
            dst_types_active.add(rel["dst_type"])

        q_cache: Dict[str, torch.Tensor] = {}
        for ntype in dst_types_active:
            ntype_id = ntype_dict[ntype]
            q_cache[ntype] = self.Q[ntype_id](h_dict[ntype]).view(-1, self.num_heads, self.head_dim)

        k_cache: Dict[str, torch.Tensor] = {}
        v_cache: Dict[str, torch.Tensor] = {}
        for ntype in src_types_active:
            ntype_id = ntype_dict[ntype]
            h_type = h_dict[ntype]
            k_cache[ntype] = self.K[ntype_id](h_type).view(-1, self.num_heads, self.head_dim)
            v_cache[ntype] = self.V[ntype_id](h_type).view(-1, self.num_heads, self.head_dim)

        for rel in packed_relations:
            src_type = rel["src_type"]
            edge_type = rel["edge_type"]
            dst_type = rel["dst_type"]
            src_idx = rel["src_idx"]
            dst_idx = rel["dst_idx"]
            edge_weight = rel.get("edge_weight", None)
            message_scale = rel.get("message_scale", None)

            if src_idx.numel() == 0:
                continue

            dst_type_id = ntype_dict[dst_type]
            edge_type_id = etype_dict[edge_type]
            dst_nodes = int(num_nodes_by_type[dst_type])

            q = q_cache[dst_type].index_select(0, dst_idx)  # [E, H, Dh]
            k = k_cache[src_type].index_select(0, src_idx)  # [E, H, Dh]
            v = v_cache[src_type].index_select(0, src_idx)  # [E, H, Dh]

            w = self.edge_W[edge_type_id]
            k_w = torch.einsum("ehd,hdk->ehk", k, w)
            attn_score = torch.sum(q * k_w, dim=2) / math.sqrt(self.head_dim)  # [E, H]

            # Pre-softmax attention bias for per-edge varying weights
            # (e.g. protein contact probabilities).
            if isinstance(edge_weight, torch.Tensor) and edge_weight.numel() > 0:
                ew = edge_weight.to(device=device, dtype=torch.float32).reshape(-1, 1)
                ew = torch.clamp(ew, min=max(float(self.eps), 1e-12))
                attn_score = attn_score + torch.log(ew).to(dtype=attn_score.dtype)

            attn_prob = self._segmented_softmax(attn_score, dst_idx, dst_nodes)
            if edge_type in self.return_attn_relations:
                rel_attn = attn_prob.mean(dim=1)  # [E], averaged over heads
                prev = self._last_relation_attns.get(edge_type)
                self._last_relation_attns[edge_type] = rel_attn if prev is None else torch.cat([prev, rel_attn], dim=0)
            attn_weight = F.dropout(attn_prob, p=self.dropout, training=self.training)

            weighted_v = v * attn_weight.unsqueeze(-1)
            # Post-softmax message scaling for per-relation constant weights
            # (e.g. cross-edge warmup). Unlike edge_weight (attention bias),
            # a constant added to all logits cancels in softmax, so we scale
            # the message directly.
            if isinstance(message_scale, torch.Tensor) and message_scale.numel() > 0:
                ms = message_scale.to(device=device, dtype=weighted_v.dtype).reshape(-1, 1, 1)
                ms = torch.clamp(ms, min=max(float(self.eps), 1e-12))
                weighted_v = weighted_v * ms
            dst_msg = torch.zeros((dst_nodes, self.num_heads, self.head_dim), device=device, dtype=weighted_v.dtype)
            dst_msg.index_add_(0, dst_idx, weighted_v)
            dst_msg = dst_msg.reshape(dst_nodes, self.out_dim)

            h_dst_updated = self.O[dst_type_id](dst_msg)
            rel_degree = torch.zeros((dst_nodes, 1), device=device, dtype=h_dst_updated.dtype)
            ones = torch.ones((dst_idx.shape[0], 1), device=device, dtype=h_dst_updated.dtype)
            rel_degree.index_add_(0, dst_idx, ones)
            rel_active = rel_degree > 0
            prior_logit = self.relation_gate_logits[edge_type_id].to(device=device, dtype=h_dst_updated.dtype)
            score = prior_logit + relation_budget_degree_scale * torch.log1p(rel_degree)
            score = torch.where(rel_active, score, torch.full_like(score, torch.finfo(score.dtype).min))

            state = relation_budget_state[dst_type]
            m_prev = state["m"]
            s_prev = state["s"]
            z_prev = state["z"]
            new_m = torch.maximum(m_prev, score)

            scale_prev = torch.exp((m_prev - new_m) / relation_budget_temp)
            exp_score = torch.exp((score - new_m) / relation_budget_temp) * rel_active.to(dtype=score.dtype)

            state["z"] = z_prev * scale_prev + h_dst_updated * exp_score
            state["s"] = s_prev * scale_prev + exp_score
            state["m"] = new_m
            state["has_any"] = state["has_any"] | rel_active
            relation_budget_meta[dst_type].append((edge_type, edge_type_id, dst_idx))

        for dst_type, rel_meta in relation_budget_meta.items():
            state = relation_budget_state[dst_type]
            s = state["s"]
            denom = s.clamp_min(self.eps)
            has_any = state["has_any"]
            zero_msg = torch.zeros_like(state["z"])
            h_msg[dst_type] = torch.where(has_any, state["z"] / denom, zero_msg)

            if not rel_meta:
                continue

            valid_mask = has_any.squeeze(-1)
            valid_count = int(valid_mask.sum().item())
            # Accumulate per-node negative entropy: sum_r p_r * log(p_r)
            neg_entropy_acc = torch.zeros_like(s)  # [N_dst, 1]
            hierarchy_mass_sum = s.new_zeros(())
            for edge_type, edge_type_id, dst_idx in rel_meta:
                rel_degree = torch.zeros_like(s)
                ones = torch.ones((dst_idx.shape[0], 1), device=device, dtype=s.dtype)
                rel_degree.index_add_(0, dst_idx, ones)
                rel_active = rel_degree > 0
                prior_logit = self.relation_gate_logits[edge_type_id].to(device=device, dtype=s.dtype)
                rel_score = prior_logit + relation_budget_degree_scale * torch.log1p(rel_degree)
                rel_score = torch.where(rel_active, rel_score, torch.full_like(rel_score, torch.finfo(rel_score.dtype).min))
                rel_exp = torch.exp((rel_score - state["m"]) / relation_budget_temp) * rel_active.to(dtype=s.dtype)
                rel_weight = rel_exp / denom
                if valid_count > 0:
                    rel_budget_sum = rel_weight[valid_mask].sum()
                    rel_mean_budget = rel_budget_sum / float(valid_count)
                else:
                    rel_budget_sum = rel_weight.new_zeros(())
                    rel_mean_budget = rel_weight.new_zeros(())
                self._last_relation_gates[edge_type] = rel_mean_budget.detach()
                self._last_relation_gate_stats[edge_type] = (rel_budget_sum.detach(), valid_count)
                if edge_type in _HIERARCHY_EDGE_TYPES:
                    hierarchy_mass_sum = hierarchy_mass_sum + rel_budget_sum.detach()
                # Accumulate p_r * log(p_r) for entropy (only where active)
                log_w = torch.where(rel_active, torch.log(rel_weight.clamp_min(self.eps)), torch.zeros_like(rel_weight))
                neg_entropy_acc = neg_entropy_acc + rel_weight * log_w
            # neg_entropy_acc = sum_r p_r * log(p_r) per node, which is -H
            # Store (sum_of_neg_entropy, valid_count) for proper weighted averaging across dst_types.
            if valid_count > 0:
                neg_ent_sum = neg_entropy_acc[valid_mask].sum()
                self._last_relation_budget_entropy[dst_type] = (neg_ent_sum, valid_count)
                self._last_relation_hierarchy_mass[dst_type] = (hierarchy_mass_sum, valid_count)
            else:
                self._last_relation_budget_entropy[dst_type] = (neg_entropy_acc.new_zeros(()), 0)
                self._last_relation_hierarchy_mass[dst_type] = (s.new_zeros(()), 0)

        h_out = {}
        for ntype, base in h_base.items():
            msg = h_msg[ntype]
            out = base + msg if base.shape == msg.shape else msg
            if self.use_norm and out.shape[-1] == self.out_dim:
                out = self.norms[ntype_dict[ntype]](out)
            h_out[ntype] = F.dropout(out, p=self.dropout, training=self.training)
        return h_out


class PackedJointHGT(nn.Module):
    """
    Joint HGT on packed edge tensors.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        use_norm: bool,
        joint_ntype_dict: Dict[str, int],
        joint_etype_dict: Dict[str, int],
        eps: float = 1e-8,
        require_torch_scatter: bool = True,
        return_attn_relations: Optional[List[str]] = None,
        relation_gate_init_bias: float = 0.0,
        relation_gate_temperature: float = 1.0,
        relation_gate_degree_scale: float = 1.0,
        relation_gate_freeze: bool = False,
    ):
        super().__init__()
        self.joint_ntype_dict = joint_ntype_dict
        self.joint_etype_dict = joint_etype_dict

        self.joint_hgt_layers = nn.ModuleList(
            [
                PackedHGTConv(
                    in_dim=in_dim if i == 0 else hidden_dim,
                    out_dim=hidden_dim,
                    num_heads=num_heads,
                    num_ntypes=len(joint_ntype_dict),
                    num_etypes=len(joint_etype_dict),
                    dropout=dropout,
                    use_norm=use_norm,
                    eps=eps,
                    require_torch_scatter=require_torch_scatter,
                    return_attn_relations=return_attn_relations,
                    relation_gate_init_bias=relation_gate_init_bias,
                    relation_gate_temperature=relation_gate_temperature,
                    relation_gate_degree_scale=relation_gate_degree_scale,
                    relation_gate_freeze=relation_gate_freeze,
                )
                for i in range(num_layers)
            ]
        )
        # Compatibility alias used by model-side access patterns.
        self.layers = self.joint_hgt_layers

    def forward(
        self,
        packed_relations: List[Dict[str, Any]],
        num_nodes_by_type: Dict[str, int],
        drug_h_init: Dict[str, torch.Tensor],
        protein_h_init: Dict[str, torch.Tensor],
        joint_h_seed: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        h_joint = {}
        # Merge drug-side and protein-side h_init dicts
        for k, v in drug_h_init.items():
            h_joint[k] = v
        for k, v in protein_h_init.items():
            h_joint[k] = v
        if joint_h_seed is not None:
            for k, v in joint_h_seed.items():
                if k not in h_joint:
                    h_joint[k] = v

        for layer in self.joint_hgt_layers:
            h_joint = layer(
                packed_relations=packed_relations,
                h_dict=h_joint,
                ntype_dict=self.joint_ntype_dict,
                etype_dict=self.joint_etype_dict,
                num_nodes_by_type=num_nodes_by_type,
            )

        return h_joint

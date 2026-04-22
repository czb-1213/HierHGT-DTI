import os
import sys
import pandas as pd
import torch
import logging
from tqdm import tqdm
import esm
import numpy as np
import hashlib
import math
import dgl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SubPocket: Louvain pocket detection on contact map
# ---------------------------------------------------------------------------
def detect_pockets(contact_map, resolution=1.0, min_pocket_size=3):
    """Detect binding pockets via Louvain community detection on the contact map.

    Args:
        contact_map: numpy array [L, L] of contact probabilities.
        resolution: Louvain resolution parameter. Higher = more, smaller communities.
        min_pocket_size: minimum residues per pocket; smaller groups get merged.

    Returns:
        res_to_pocket: list of int, length = L. res_to_pocket[i] = pocket id for residue i.
        num_pockets: int, number of pockets.

    Fallback: if detection fails, put all residues in one pocket (id=0).
    """
    import networkx as nx
    import numpy as np

    L = contact_map.shape[0]
    if L == 0:
        return [0], 1
    if L <= min_pocket_size:
        return [0] * L, 1

    try:
        # Build weighted graph from contact map (vectorized)
        G = nx.Graph()
        G.add_nodes_from(range(L))

        # Vectorized: extract upper-triangle contact edges with prob > 0.5
        rows, cols = np.where(np.triu(contact_map, k=1) > 0.5)
        weights = contact_map[rows, cols]
        G.add_weighted_edges_from(zip(rows.tolist(), cols.tolist(),
                                      weights.tolist()))

        # Always add sequence neighbor edges where missing (weight=0.3)
        # Ensures chain connectivity without overriding strong contact weights
        seq_weight = 0.3
        for i in range(L - 1):
            if not G.has_edge(i, i + 1):
                G.add_edge(i, i + 1, weight=seq_weight)

        # Run Louvain community detection
        communities = nx.community.louvain_communities(
            G, weight='weight', resolution=resolution, seed=42
        )

        # Build res_to_pocket mapping
        res_to_pocket = [0] * L
        pocket_id = 0
        pocket_sizes = []

        for community in communities:
            for node in community:
                if node < L:
                    res_to_pocket[node] = pocket_id
            pocket_sizes.append(len(community))
            pocket_id += 1

        num_pockets = pocket_id

        # Merge small pockets into nearest larger pocket
        if num_pockets > 1:
            small_pockets = [pid for pid, size in enumerate(pocket_sizes) if size < min_pocket_size]
            if small_pockets:
                # Find the largest pocket
                largest_pocket = max(range(num_pockets), key=lambda pid: pocket_sizes[pid])
                for pid in small_pockets:
                    for i in range(L):
                        if res_to_pocket[i] == pid:
                            res_to_pocket[i] = largest_pocket

                # Re-index pockets to be contiguous
                unique_pockets = sorted(set(res_to_pocket))
                remap = {old: new for new, old in enumerate(unique_pockets)}
                res_to_pocket = [remap[p] for p in res_to_pocket]
                num_pockets = len(unique_pockets)

        if num_pockets == 0:
            return [0] * L, 1

        return res_to_pocket, num_pockets

    except Exception:
        return [0] * L, 1


# ESM模型层数映射
ESM_MODEL_LAYERS = {
    "esm2_t6_8M_UR50D": 6,
    "esm2_t12_35M_UR50D": 12,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
    "esm2_t36_3B_UR50D": 36,
    "esm2_t48_15B_UR50D": 48,
}

class ESMFeatureCache:
    def __init__(self, esm_model_name="esm2_t6_8M_UR50D", esm_layer=-1, 
                 cache_base_dir="./esm_cache", device=None, 
                 max_seq_len=1000, chunk_overlap=50, batch_size=1,
                 protein_window_size=3, build_graph=True,
                 use_contact_edges=True, contact_top_k=10,
                 contact_min_prob=0.3, contact_prob_bins=None,
                 prefer_predicted_contact=True):
        """
        ESM特征缓存器
        
        参数:
            esm_model_name: ESM模型名称
            esm_layer: 使用的ESM层（-1表示最后一层）
            cache_base_dir: 缓存基础目录
            device: 计算设备
            max_seq_len: 最大序列长度
            chunk_overlap: 分块重叠长度
            batch_size: 批处理大小（减少内存占用）
            protein_window_size: 蛋白质残基图的窗口大小
            build_graph: 是否构建残基图
        """
        self.esm_model_name = esm_model_name
        self.esm_layer = esm_layer
        self.cache_base_dir = cache_base_dir
        self.max_seq_len = max_seq_len
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.protein_window_size = protein_window_size
        self.build_graph = build_graph
        self.use_contact_edges = use_contact_edges
        self.contact_top_k = int(contact_top_k)
        self.contact_min_prob = float(contact_min_prob)
        if contact_prob_bins is None:
            contact_prob_bins = [0.5, 0.7, 0.9]
        self.contact_prob_bins = sorted([float(x) for x in contact_prob_bins])
        self.prefer_predicted_contact = prefer_predicted_contact
        self.edge_type_dim = 3
        # 0 is reserved for non-sequence edges; 1..protein_window_size represent |i-j| bins
        self.seq_gap_bin_dim = max(1, int(self.protein_window_size)) + 1
        self.contact_bin_dim = len(self.contact_prob_bins) + 2
        self.edge_feat_dim = self.edge_type_dim + self.seq_gap_bin_dim + self.contact_bin_dim
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"批处理大小: {self.batch_size}")
        logger.info(f"构建残基图: {self.build_graph}")
        
        # 加载ESM模型
        logger.info(f"use_contact_edges: {self.use_contact_edges}")
        if self.use_contact_edges:
            logger.info(f"contact_top_k: {self.contact_top_k}, contact_min_prob: {self.contact_min_prob}")
            logger.info(f"contact_prob_bins: {self.contact_prob_bins}")
            logger.info(f"prefer_predicted_contact: {self.prefer_predicted_contact}")

        self._load_esm_model()
    
    def _load_esm_model(self):
        """加载ESM模型"""
        try:
            logger.info(f"加载ESM模型: {self.esm_model_name}")
            
            # 兼容不同版本的fair-esm API
            if hasattr(esm.pretrained, 'load_model'):
                esm_model, alphabet = esm.pretrained.load_model(self.esm_model_name)
            else:
                esm_model, alphabet = esm.pretrained.load_model_and_alphabet(self.esm_model_name)
            
            self.esm_model = esm_model.to(self.device)
            self.esm_model.eval()
            
            self.batch_converter = alphabet.get_batch_converter()
            
            # 自动检测模型层数
            if hasattr(self.esm_model, 'num_layers'):
                model_num_layers = self.esm_model.num_layers
            elif hasattr(self.esm_model, 'layers'):
                model_num_layers = len(self.esm_model.layers)
            else:
                model_num_layers = ESM_MODEL_LAYERS.get(self.esm_model_name, 33)
                logger.warning(f"无法自动检测模型层数，使用默认值: {model_num_layers}")
            
            logger.info(f"ESM模型总层数: {model_num_layers}")
            
            # 设置ESM层
            if self.esm_layer is None or self.esm_layer == -1:
                self.esm_layer = model_num_layers
                logger.info(f"使用最后一层特征: layer {self.esm_layer}")
            elif self.esm_layer > model_num_layers:
                logger.warning(f"请求的层 {self.esm_layer} 超出模型层数 {model_num_layers}，使用最后一层")
                self.esm_layer = model_num_layers
            elif self.esm_layer < 0:
                self.esm_layer = model_num_layers + self.esm_layer + 1
                logger.info(f"使用倒数第 {-self.esm_layer + model_num_layers} 层特征: layer {self.esm_layer}")
            
            logger.info("ESM模型加载成功")
            
        except Exception as e:
            logger.error(f"加载ESM模型失败: {e}")
            raise
    
    def _compute_esm_features(self, protein_seq, need_contacts=False):
        """计算蛋白质序列的ESM特征

        Args:
            protein_seq: 蛋白质序列字符串
            need_contacts: 是否同时返回 contact map（避免二次 forward pass）

        Returns:
            need_contacts=False: features [L, d]
            need_contacts=True:  (features [L, d], contact_map [L, L] or None)

        注意: 长序列（> max_seq_len）不请求 contacts（显存限制），contacts 返回 None。
        """
        try:
            seq_len = len(protein_seq)
            # 长序列不请求 contacts（return_contacts 需要保存所有层 attention，显存 O(L²) 会 OOM）
            actual_need_contacts = need_contacts and (seq_len <= self.max_seq_len)
            result = self._compute_single_sequence_features(
                protein_seq, return_contacts=actual_need_contacts)
            if need_contacts and not actual_need_contacts:
                # 长序列：补上 None contacts
                return (result, None)
            return result
        except Exception as e:
            logger.error(f"计算ESM特征失败 (len={len(protein_seq)}): {e}")
            return (None, None) if need_contacts else None

    def _compute_single_sequence_features(self, protein_seq, return_contacts=False):
        """计算单个序列的ESM特征（返回完整序列特征）

        Args:
            protein_seq: 蛋白质序列字符串
            return_contacts: 是否同时返回 contact map

        Returns:
            return_contacts=False: features [L, d]
            return_contacts=True:  (features [L, d], contact_map [L, L] or None)
        """
        with torch.no_grad():
            data = [("protein", protein_seq)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            outputs = self.esm_model(
                batch_tokens,
                repr_layers=[self.esm_layer],
                return_contacts=return_contacts,
            )
            representations = outputs["representations"][self.esm_layer]

            # 移除batch和CLS token，返回完整序列特征 [L, d]
            seq_len = len(protein_seq)
            token_representations = representations[0, 1:seq_len+1, :]
            features = token_representations.cpu()

            if not return_contacts:
                return features

            # 提取 contact map
            contact_map = None
            if isinstance(outputs, dict):
                contacts = outputs.get("contacts", None)
                if contacts is not None:
                    cm = contacts[0]
                    if cm.dim() == 2:
                        if cm.shape[0] == seq_len + 2 and cm.shape[1] == seq_len + 2:
                            cm = cm[1:seq_len + 1, 1:seq_len + 1]
                        elif cm.shape[0] != seq_len or cm.shape[1] != seq_len:
                            m = min(seq_len, cm.shape[0], cm.shape[1])
                            fixed = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=cm.device)
                            fixed[:m, :m] = cm[:m, :m]
                            cm = fixed
                        contact_map = cm.detach().float().cpu().clamp(0.0, 1.0)

            return features, contact_map
    
    def _compute_batch_features(self, protein_seqs, return_contacts=False):
        """批量计算多条短序列的ESM特征（真正的 batch 推理）

        所有序列必须 <= max_seq_len。batch_converter 自动 pad + mask，
        非 pad token 的 representation 与单条推理完全一致。

        Args:
            protein_seqs: list of str，蛋白质序列列表
            return_contacts: 是否同时返回 contact maps

        Returns:
            return_contacts=False: list of Tensor [Li, d]
            return_contacts=True:  list of (Tensor [Li, d], contact_map [Li, Li] or None)
        """
        if not protein_seqs:
            return []

        with torch.no_grad():
            data = [(f"protein_{i}", seq) for i, seq in enumerate(protein_seqs)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            outputs = self.esm_model(
                batch_tokens,
                repr_layers=[self.esm_layer],
                return_contacts=return_contacts,
            )
            representations = outputs["representations"][self.esm_layer]

            results = []
            for idx, seq in enumerate(protein_seqs):
                seq_len = len(seq)
                # 移除 CLS token: positions 1..seq_len
                features = representations[idx, 1:seq_len + 1, :].cpu()

                if not return_contacts:
                    results.append(features)
                else:
                    contact_map = None
                    if isinstance(outputs, dict):
                        contacts = outputs.get("contacts", None)
                        if contacts is not None:
                            cm = contacts[idx]
                            if cm.dim() == 2:
                                if cm.shape[0] == seq_len + 2 and cm.shape[1] == seq_len + 2:
                                    cm = cm[1:seq_len + 1, 1:seq_len + 1]
                                elif cm.shape[0] != seq_len or cm.shape[1] != seq_len:
                                    m = min(seq_len, cm.shape[0], cm.shape[1])
                                    fixed = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=cm.device)
                                    fixed[:m, :m] = cm[:m, :m]
                                    cm = fixed
                                contact_map = cm.detach().float().cpu().clamp(0.0, 1.0)
                    results.append((features, contact_map))

            return results

    def _compute_chunked_features(self, protein_seq):
        """计算分块序列的ESM特征（使用张量累加优化内存）"""
        seq_len = len(protein_seq)

        # 计算分块数量
        effective_chunk_size = self.max_seq_len - self.chunk_overlap
        num_chunks = math.ceil((seq_len - self.chunk_overlap) / effective_chunk_size)

        # 先提取第一个块的特征来确定维度
        first_chunk = protein_seq[:self.max_seq_len]
        first_features = self._compute_single_sequence_features(first_chunk, return_contacts=False)
        feature_dim = first_features.shape[1]

        # 预分配张量（内存优化：不使用列表存储中间结果）
        merged_features = torch.zeros((seq_len, feature_dim),
                                      dtype=first_features.dtype,
                                      device='cpu')
        coverage = torch.zeros(seq_len, dtype=torch.float32)

        # 已经处理了第一个块
        merged_features[:len(first_chunk)] += first_features
        coverage[:len(first_chunk)] += 1

        # 处理剩余的块
        for i in range(1, num_chunks):
            start = i * effective_chunk_size
            end = min(start + self.max_seq_len, seq_len)

            # 确保最后一个块包含序列末尾
            if i == num_chunks - 1:
                end = seq_len
                start = max(0, end - self.max_seq_len)

            chunk = protein_seq[start:end]
            features = self._compute_single_sequence_features(chunk, return_contacts=False)

            # 直接更新合并特征和覆盖计数（内存优化：不存储中间结果）
            seq_len_chunk = end - start
            merged_features[start:end] += features[:seq_len_chunk]
            coverage[start:end] += 1

        # 对重叠区域取平均
        epsilon = 1e-10
        coverage = coverage.unsqueeze(1)
        merged_features = merged_features / (coverage + epsilon)

        return merged_features

    def _build_edge_feature(self, edge_type_idx, seq_gap_bin_idx=0, contact_bin_idx=0):
        """Build edge_feat = edge_type(seq/contact/self) + seq_gap_bin + contact_bin."""
        edge_type = [0.0] * self.edge_type_dim
        edge_type[max(0, min(self.edge_type_dim - 1, int(edge_type_idx)))] = 1.0

        seq_gap_bin = [0.0] * self.seq_gap_bin_dim
        seq_gap_bin[max(0, min(self.seq_gap_bin_dim - 1, int(seq_gap_bin_idx)))] = 1.0

        contact_bin = [0.0] * self.contact_bin_dim
        contact_bin[max(0, min(self.contact_bin_dim - 1, int(contact_bin_idx)))] = 1.0
        return edge_type + seq_gap_bin + contact_bin

    def _seq_gap_to_bin_idx(self, abs_gap):
        """Map |i-j| to a sequence-gap bin index. 0 is reserved for non-sequence edges."""
        gap = max(0, int(abs_gap))
        if gap <= 0:
            return 0
        return min(gap, self.seq_gap_bin_dim - 1)

    def _contact_prob_to_bin_idx(self, prob):
        """Map contact probability to a bin index. 0 is reserved for non-contact."""
        idx = 1
        for boundary in self.contact_prob_bins:
            if prob >= boundary:
                idx += 1
            else:
                break
        return min(idx, self.contact_bin_dim - 1)

    def _predict_contact_probs(self, protein_seq, precomputed_contacts=None):
        """Get contact probabilities from pre-computed ESM contacts.

        Args:
            protein_seq: used only for length reference.
            precomputed_contacts: contact map already obtained during feature
                computation.  If None, returns None (caller falls back to
                similarity-based estimation).
        """
        if not self.prefer_predicted_contact:
            return None
        if precomputed_contacts is None:
            return None
        return precomputed_contacts

    def _predict_contact_probs_chunked(self, protein_seq):
        """对长序列分 chunk 获取 predicted contact map，拼成完整矩阵。

        每个 chunk 长度 <= max_seq_len，可以安全调用 return_contacts=True。
        重叠区域取平均，非重叠区域直接赋值。chunk 之间的跨 chunk 残基对没有
        predicted contact 信息（值为 0），这是合理的——ESM contact prediction
        本身就是局部的。
        """
        if not self.prefer_predicted_contact:
            return None

        seq_len = len(protein_seq)
        if seq_len <= self.max_seq_len:
            # 短序列直接拿（不应该走到这里，但保险起见）
            _, contact_map = self._compute_single_sequence_features(
                protein_seq, return_contacts=True)
            return contact_map

        effective_chunk_size = self.max_seq_len - self.chunk_overlap
        num_chunks = math.ceil((seq_len - self.chunk_overlap) / effective_chunk_size)

        merged_contacts = torch.zeros((seq_len, seq_len), dtype=torch.float32)
        coverage = torch.zeros((seq_len, seq_len), dtype=torch.float32)

        for ci in range(num_chunks):
            start = ci * effective_chunk_size
            end = min(start + self.max_seq_len, seq_len)
            if ci == num_chunks - 1:
                end = seq_len
                start = max(0, end - self.max_seq_len)

            chunk = protein_seq[start:end]
            try:
                _, chunk_contacts = self._compute_single_sequence_features(
                    chunk, return_contacts=True)
            except RuntimeError:
                # OOM: 跳过这个 chunk 的 contacts
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if chunk_contacts is None:
                continue

            chunk_len = end - start
            cm = chunk_contacts[:chunk_len, :chunk_len]
            merged_contacts[start:end, start:end] += cm
            coverage[start:end, start:end] += 1

        # 取平均（避免除零）
        mask = coverage > 0
        merged_contacts[mask] /= coverage[mask]

        return merged_contacts.clamp(0.0, 1.0)

    def _estimate_contact_probs_from_features(self, features):
        """Fallback: estimate contact confidence via residue feature similarity."""
        if features is None or features.shape[0] == 0:
            return None
        feats = features.float()
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)
        sim = torch.matmul(feats, feats.T)
        return ((sim + 1.0) * 0.5).clamp(0.0, 1.0).cpu()

    def _get_contact_probs(self, protein_seq, features, precomputed_contacts=None):
        """Prefer predicted contacts; fallback to similarity-derived contacts."""
        if not self.use_contact_edges:
            return None
        contact_probs = self._predict_contact_probs(protein_seq, precomputed_contacts)
        if contact_probs is None:
            contact_probs = self._estimate_contact_probs_from_features(features)
        if contact_probs is None:
            return None
        contact_probs = contact_probs.clone()
        contact_probs.fill_diagonal_(0.0)
        return contact_probs

    def _select_contact_edges(self, contact_probs):
        """Row-wise top-k contact edges with min-prob threshold, then symmetrize."""
        if contact_probs is None:
            return [], [], []
        num_residues = contact_probs.shape[0]
        if num_residues <= 1 or self.contact_top_k <= 0:
            return [], [], []

        k = min(self.contact_top_k, num_residues - 1)
        vals, idx = torch.topk(contact_probs, k=k, dim=1)
        mask = vals >= self.contact_min_prob
        if not mask.any():
            return [], [], []

        src = torch.arange(num_residues, dtype=torch.long).unsqueeze(1).expand(-1, k)
        src = src[mask].tolist()
        dst = idx[mask].tolist()
        prob = vals[mask].tolist()

        # De-duplicate directed edges and keep the max confidence for repeated pairs.
        edge_to_prob = {}
        for u, v, p in zip(src, dst, prob):
            key = (int(u), int(v))
            p_val = float(p)
            if key not in edge_to_prob or p_val > edge_to_prob[key]:
                edge_to_prob[key] = p_val

        # Symmetrize: if (i, j) is selected, auto-add (j, i).
        for (u, v), p_val in list(edge_to_prob.items()):
            rev_key = (v, u)
            if rev_key not in edge_to_prob:
                edge_to_prob[rev_key] = p_val

        ordered_items = sorted(edge_to_prob.items())
        src = [edge[0] for edge, _ in ordered_items]
        dst = [edge[1] for edge, _ in ordered_items]
        prob = [p_val for _, p_val in ordered_items]
        return src, dst, prob

    def _build_protein_graph(self, protein_seq, features, precomputed_contacts=None):
        """
        Build protein residue graph.

        Keep sequence-window edges, add contact edges (top-k + probability bins),
        and explicit self-loop edges.
        edge_feat = edge_type + seq_gap_bin + contact_bin.
        edge_weight = raw contact probability for contact edges, 1.0 for others.
        """
        num_residues = len(protein_seq)

        # --- Vectorized sequence-window edges ---
        window = self.protein_window_size
        offsets = list(range(-window, window + 1))
        offsets = [o for o in offsets if o != 0]  # exclude self-loops (added later)

        all_i = np.arange(num_residues)
        seq_src_list, seq_dst_list, seq_gap_list = [], [], []
        for offset in offsets:
            j = all_i + offset
            valid = (j >= 0) & (j < num_residues)
            src_arr = all_i[valid]
            dst_arr = j[valid]
            seq_src_list.append(src_arr)
            seq_dst_list.append(dst_arr)
            seq_gap_list.append(np.full(src_arr.shape[0], abs(offset), dtype=np.int64))

        seq_src = np.concatenate(seq_src_list)
        seq_dst = np.concatenate(seq_dst_list)
        seq_gaps = np.concatenate(seq_gap_list)

        # Vectorized edge_feat for sequence edges
        num_seq_edges = seq_src.shape[0]
        seq_edge_feats = np.zeros((num_seq_edges, self.edge_feat_dim), dtype=np.float32)
        # edge_type one-hot: index 0
        seq_edge_feats[:, 0] = 1.0
        # seq_gap_bin one-hot
        gap_bin_idx = np.clip(seq_gaps, 0, self.seq_gap_bin_dim - 1)
        seq_edge_feats[np.arange(num_seq_edges),
                       self.edge_type_dim + gap_bin_idx] = 1.0
        # contact_bin one-hot: index 0 (non-contact)
        seq_edge_feats[:, self.edge_type_dim + self.seq_gap_bin_dim] = 1.0

        seq_weights = np.ones(num_seq_edges, dtype=np.float32)

        # --- Contact edges ---
        # Always compute contact probs (needed for both contact edges and pocket detection)
        contact_probs = self._get_contact_probs(protein_seq, features, precomputed_contacts)

        # 判断是否拿到了 predicted contacts
        has_predicted_contacts = (contact_probs is not None and precomputed_contacts is not None)

        if has_predicted_contacts:
            # 短序列正常路径：predicted contacts 可用
            contact_probs_for_pockets = contact_probs
        elif precomputed_contacts is None and self.prefer_predicted_contact:
            # 长序列或补建图路径：分 chunk 获取 predicted contacts 用于 pocket detection
            chunked_contacts = self._predict_contact_probs_chunked(protein_seq)
            if chunked_contacts is not None:
                contact_probs_for_pockets = chunked_contacts.clone()
                contact_probs_for_pockets.fill_diagonal_(0.0)
                # 如果 contact_probs 还是 None（contact edges 也需要），用 chunked contacts
                if contact_probs is None and self.use_contact_edges:
                    contact_probs = contact_probs_for_pockets
            else:
                contact_probs_for_pockets = None
        else:
            contact_probs_for_pockets = None

        # 最终 fallback：contact edges 需要但仍然没有 contact_probs
        if self.use_contact_edges and contact_probs is None:
            contact_probs = self._estimate_contact_probs_from_features(features)
            if contact_probs is not None:
                contact_probs = contact_probs.clone()
                contact_probs.fill_diagonal_(0.0)

        contact_src = np.array([], dtype=np.int64)
        contact_dst = np.array([], dtype=np.int64)
        contact_edge_feats = np.zeros((0, self.edge_feat_dim), dtype=np.float32)
        contact_weights = np.array([], dtype=np.float32)

        if self.use_contact_edges and contact_probs is not None:
            c_src, c_dst, c_prob = self._select_contact_edges(contact_probs)
            if c_src:
                c_src_arr = np.array(c_src, dtype=np.int64)
                c_dst_arr = np.array(c_dst, dtype=np.int64)
                c_prob_arr = np.array(c_prob, dtype=np.float32)
                # Filter out edges within sequence window
                mask = np.abs(c_src_arr - c_dst_arr) > window
                if mask.any():
                    c_src_arr = c_src_arr[mask]
                    c_dst_arr = c_dst_arr[mask]
                    c_prob_arr = c_prob_arr[mask]

                    num_c = c_src_arr.shape[0]
                    c_feats = np.zeros((num_c, self.edge_feat_dim), dtype=np.float32)
                    # edge_type one-hot: index 1 (contact)
                    c_feats[:, 1] = 1.0
                    # seq_gap_bin one-hot: index 0 (non-sequence)
                    c_feats[:, self.edge_type_dim] = 1.0
                    # contact_bin one-hot: vectorized
                    bins_arr = np.array(self.contact_prob_bins, dtype=np.float32)
                    # count how many bins each prob exceeds → bin index = 1 + count
                    bin_idx = 1 + np.sum(c_prob_arr[:, None] >= bins_arr[None, :], axis=1)
                    bin_idx = np.clip(bin_idx, 0, self.contact_bin_dim - 1).astype(np.int64)
                    c_feats[np.arange(num_c),
                            self.edge_type_dim + self.seq_gap_bin_dim + bin_idx] = 1.0

                    contact_src = c_src_arr
                    contact_dst = c_dst_arr
                    contact_edge_feats = c_feats
                    contact_weights = c_prob_arr

        # --- Self-loop edges ---
        self_nodes = np.arange(num_residues, dtype=np.int64)
        self_feats = np.zeros((num_residues, self.edge_feat_dim), dtype=np.float32)
        # edge_type one-hot: index 2 (self-loop)
        self_feats[:, 2] = 1.0
        # seq_gap_bin one-hot: index 0
        self_feats[:, self.edge_type_dim] = 1.0
        # contact_bin one-hot: index 0
        self_feats[:, self.edge_type_dim + self.seq_gap_bin_dim] = 1.0
        self_weights = np.ones(num_residues, dtype=np.float32)

        # --- Concatenate all edges ---
        all_src = np.concatenate([seq_src, contact_src, self_nodes])
        all_dst = np.concatenate([seq_dst, contact_dst, self_nodes])
        all_feats = np.concatenate([seq_edge_feats, contact_edge_feats, self_feats], axis=0)
        all_weights = np.concatenate([seq_weights, contact_weights, self_weights])

        if all_src.shape[0] > 0:
            g = dgl.graph((torch.from_numpy(all_src), torch.from_numpy(all_dst)),
                          num_nodes=num_residues)
            g.edata['edge_feat'] = torch.from_numpy(all_feats)
            g.edata['edge_weight'] = torch.from_numpy(all_weights)
        else:
            empty = torch.tensor([], dtype=torch.int64)
            g = dgl.graph((empty, empty), num_nodes=num_residues)
            g.edata['edge_feat'] = torch.zeros((0, self.edge_feat_dim), dtype=torch.float32)
            g.edata['edge_weight'] = torch.zeros((0,), dtype=torch.float32)

        g.ndata['h'] = features

        # --- SubPocket: detect binding pockets ---
        if contact_probs_for_pockets is not None:
            contact_np = (contact_probs_for_pockets.cpu().numpy()
                          if isinstance(contact_probs_for_pockets, torch.Tensor)
                          else np.array(contact_probs_for_pockets))
            res_to_pocket, num_pockets = detect_pockets(contact_np)
        else:
            # 没有 predicted contacts 时，Louvain 无意义，所有残基归为单个 pocket
            res_to_pocket = [0] * num_residues
            num_pockets = 1
        g.ndata['res_to_pocket'] = torch.tensor(res_to_pocket, dtype=torch.long)
        g.num_pockets = num_pockets

        return g
    @staticmethod
    def _normalize_dataset_name(dataset_name):
        """标准化数据集名称，确保大小写一致"""
        DATASET_NAME_MAP = {
            'biosnap': 'BioSnap',
            'drugbank': 'DrugBank',
            'bindingdb': 'BindingDB',
            'davis': 'Davis',
            'kiba': 'KIBA',
        }
        base = dataset_name.split('_')[0].lower()
        return DATASET_NAME_MAP.get(base, dataset_name.split('_')[0])

    def _get_esm_cache_path(self, protein_seq, dataset_name):
        """获取ESM特征缓存路径"""
        seq_hash = hashlib.md5(protein_seq.encode()).hexdigest()
        base_dataset = self._normalize_dataset_name(dataset_name)
        cache_dir = os.path.join(self.cache_base_dir, base_dataset)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{seq_hash}.pt")

    def _get_graph_cache_path(self, protein_seq, dataset_name):
        """获取图缓存路径"""
        seq_hash = hashlib.md5(protein_seq.encode()).hexdigest()
        base_dataset = self._normalize_dataset_name(dataset_name)
        cache_dir = os.path.join(self.cache_base_dir, f"{base_dataset}_graphs")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{seq_hash}.bin")
    
    def _save_graph(self, graph, cache_path):
        """保存图到缓存"""
        try:
            dgl.save_graphs(cache_path, [graph])
        except Exception as e:
            logger.warning(f"保存图到缓存失败: {e}")
    
    def cache_dataset(self, dataset_path, dataset_name, protein_col="Protein",
                      batch_size=None, save_progress=True):
        """
        缓存数据集中的蛋白质ESM特征和图

        参数:
            dataset_path: 数据集CSV文件路径
            dataset_name: 数据集名称（用于缓存子目录）
            protein_col: 蛋白质序列列名
            batch_size: 批处理大小（None表示使用默认值）
            save_progress: 是否保存进度
        """
        if batch_size is None:
            batch_size = self.batch_size

        logger.info(f"开始缓存数据集: {dataset_name}")
        logger.info(f"数据集路径: {dataset_path}")
        logger.info(f"批处理大小: {batch_size}")

        # 读取数据集
        df = pd.read_csv(dataset_path)
        logger.info(f"数据集包含 {len(df)} 个样本")

        # 获取唯一的蛋白质序列
        unique_proteins = df[protein_col].unique()
        logger.info(f"发现 {len(unique_proteins)} 个唯一蛋白质序列")

        # 创建数据集特定的缓存目录
        dataset_cache_dir = os.path.join(self.cache_base_dir, dataset_name)
        os.makedirs(dataset_cache_dir, exist_ok=True)
        logger.info(f"缓存目录: {dataset_cache_dir}")

        # 进度文件
        progress_file = os.path.join(dataset_cache_dir, "progress.txt")
        processed_proteins = set()

        # 加载已处理的蛋白质
        if save_progress and os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                for line in f:
                    seq_hash = line.strip()
                    processed_proteins.add(seq_hash)
            logger.info(f"已处理 {len(processed_proteins)} 个蛋白质")

        # --- Phase 1: 检查缓存状态，收集需要计算的序列 ---
        cached_count = 0
        skipped_count = 0
        need_compute = []        # 需要完整 ESM 计算的序列
        need_graph_only = []     # ESM 特征已存在，只需补建图

        need_contacts = self.build_graph and self.use_contact_edges and self.prefer_predicted_contact

        for protein_seq in unique_proteins:
            seq_hash = hashlib.md5(protein_seq.encode()).hexdigest()
            esm_cache_path = self._get_esm_cache_path(protein_seq, dataset_name)
            graph_cache_path = self._get_graph_cache_path(protein_seq, dataset_name) if self.build_graph else None

            esm_exists = os.path.exists(esm_cache_path)
            graph_exists = graph_cache_path and os.path.exists(graph_cache_path)
            need_graph = self.build_graph and not graph_exists

            # 进度文件脏数据检测
            if seq_hash in processed_proteins and not esm_exists:
                logger.warning(f"检测到进度文件脏数据（ESM文件缺失），将重算: {seq_hash}")
                processed_proteins.discard(seq_hash)

            # 进度文件中已记录为完成
            if seq_hash in processed_proteins:
                if need_graph:
                    need_graph_only.append((protein_seq, seq_hash, esm_cache_path, graph_cache_path))
                else:
                    skipped_count += 1
                continue

            # 两者都已存在
            if esm_exists and (not self.build_graph or graph_exists):
                skipped_count += 1
                continue

            # ESM 特征已存在但图缺失
            if esm_exists and need_graph:
                need_graph_only.append((protein_seq, seq_hash, esm_cache_path, graph_cache_path))
                continue

            # 需要完整计算
            need_compute.append((protein_seq, seq_hash, esm_cache_path, graph_cache_path))

        logger.info(f"跳过 {skipped_count} 个已完成, "
                    f"需补建图 {len(need_graph_only)} 个, "
                    f"需完整计算 {len(need_compute)} 个")

        # --- Phase 2: 补建图（不需要 ESM forward pass）---
        for protein_seq, seq_hash, esm_cache_path, graph_cache_path in tqdm(
                need_graph_only, desc=f"补建图 {dataset_name}"):
            try:
                features = torch.load(esm_cache_path, weights_only=True)
                protein_graph = self._build_protein_graph(protein_seq, features)
                self._save_graph(protein_graph, graph_cache_path)
                cached_count += 1
            except Exception as e:
                logger.warning(f"补建图失败，加入重算队列: {e}")
                try:
                    if os.path.exists(esm_cache_path):
                        os.remove(esm_cache_path)
                except OSError:
                    pass
                processed_proteins.discard(seq_hash)
                need_compute.append((protein_seq, seq_hash, esm_cache_path, graph_cache_path))

        # --- Phase 3: 完整 ESM 计算（batch 推理）---
        # 把短序列和长序列分开
        short_seqs = []  # (protein_seq, seq_hash, esm_cache_path, graph_cache_path)
        long_seqs = []
        for item in need_compute:
            if len(item[0]) <= self.max_seq_len:
                short_seqs.append(item)
            else:
                long_seqs.append(item)

        # 短序列按长度排序后分 batch（减少 padding 浪费）
        short_seqs.sort(key=lambda x: len(x[0]))

        logger.info(f"短序列 {len(short_seqs)} 条 (batch推理), 长序列 {len(long_seqs)} 条 (逐条推理)")

        # 处理短序列 batch（OOM 时自动减半重试，最终 fallback 到逐条）
        ptr = 0
        cur_bs = batch_size
        pbar = tqdm(total=len(short_seqs), desc=f"batch推理 {dataset_name}")
        while ptr < len(short_seqs):
            batch_items = short_seqs[ptr:ptr + cur_bs]
            batch_protein_seqs = [item[0] for item in batch_items]

            try:
                batch_results = self._compute_batch_features(
                    batch_protein_seqs, return_contacts=need_contacts)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and cur_bs > 1:
                    # OOM: 清理显存，减半 batch size，重试当前位置
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    cur_bs = max(1, cur_bs // 2)
                    logger.warning(f"OOM，batch_size 缩减为 {cur_bs}，重试")
                    continue
                else:
                    # 非 OOM 错误或 batch_size 已经是 1，逐条 fallback
                    logger.warning(f"Batch 推理失败，回退到逐条推理: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    batch_results = []
                    for seq in batch_protein_seqs:
                        try:
                            result = self._compute_esm_features(seq, need_contacts=need_contacts)
                            batch_results.append(result)
                        except Exception as e2:
                            logger.error(f"逐条推理也失败: {e2}")
                            batch_results.append((None, None) if need_contacts else None)

            for item, result in zip(batch_items, batch_results):
                protein_seq, seq_hash, esm_cache_path, graph_cache_path = item

                if need_contacts:
                    features, precomputed_contacts = result
                else:
                    features = result
                    precomputed_contacts = None

                if features is None:
                    continue

                torch.save(features, esm_cache_path)
                cached_count += 1
                processed_proteins.add(seq_hash)

                if self.build_graph:
                    try:
                        protein_graph = self._build_protein_graph(
                            protein_seq, features, precomputed_contacts)
                        self._save_graph(protein_graph, graph_cache_path)
                    except Exception as e:
                        logger.warning(f"构建图失败: {e}")

                if save_progress:
                    with open(progress_file, 'a') as f:
                        f.write(f"{seq_hash}\n")

            pbar.update(len(batch_items))
            ptr += len(batch_items)
        pbar.close()

        # 处理长序列（逐条推理，直接全长输入，不分 chunk）
        for protein_seq, seq_hash, esm_cache_path, graph_cache_path in tqdm(
                long_seqs, desc=f"长序列推理 {dataset_name}"):
            try:
                result = self._compute_esm_features(protein_seq, need_contacts=need_contacts)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.error(f"长序列推理失败 (len={len(protein_seq)}): {e}")
                continue

            if need_contacts:
                features, precomputed_contacts = result
            else:
                features = result
                precomputed_contacts = None

            if features is None:
                continue

            torch.save(features, esm_cache_path)
            cached_count += 1
            processed_proteins.add(seq_hash)

            if self.build_graph:
                try:
                    protein_graph = self._build_protein_graph(
                        protein_seq, features, precomputed_contacts)
                    self._save_graph(protein_graph, graph_cache_path)
                except Exception as e:
                    logger.warning(f"构建图失败: {e}")

            if save_progress:
                with open(progress_file, 'a') as f:
                    f.write(f"{seq_hash}\n")

        logger.info(f"缓存完成: {cached_count} 个新缓存, {skipped_count} 个已存在")
        logger.info(f"缓存目录: {dataset_cache_dir}")
        if self.build_graph:
            graph_cache_dir = os.path.join(self.cache_base_dir, f"{dataset_name}_graphs")
            logger.info(f"图缓存目录: {graph_cache_dir}")


def cache_all_datasets(datasets_root, cache_base_dir="./esm_cache", 
                      esm_model_name="esm2_t6_8M_UR50D", esm_layer=-1,
                      protein_col="Protein", batch_size=1, single_dataset=None,
                      protein_window_size=3, build_graph=True,
                      use_contact_edges=True, contact_top_k=10,
                      contact_min_prob=0.3, contact_prob_bins=None,
                      prefer_predicted_contact=True):
    """
    缓存所有数据集的ESM特征
    
    参数:
        datasets_root: 数据集根目录
        cache_base_dir: 缓存基础目录
        esm_model_name: ESM模型名称
        esm_layer: 使用的ESM层
        protein_col: 蛋白质序列列名
        batch_size: 批处理大小
        single_dataset: 只处理指定的数据集（格式：dataset_type_split，如 "bindingdb_random"）
    """
    # 创建ESM特征缓存器
    cache = ESMFeatureCache(
        esm_model_name=esm_model_name,
        esm_layer=esm_layer,
        cache_base_dir=cache_base_dir,
        batch_size=batch_size,
        protein_window_size=protein_window_size,
        build_graph=build_graph,
        use_contact_edges=use_contact_edges,
        contact_top_k=contact_top_k,
        contact_min_prob=contact_min_prob,
        contact_prob_bins=contact_prob_bins,
        prefer_predicted_contact=prefer_predicted_contact
    )
    
    # 查找所有full.csv文件（只处理完整数据集）
    full_dataset_files = []
    
    for root, dirs, files in os.walk(datasets_root):
        for file in files:
            if file == "full.csv":
                file_path = os.path.join(root, file)
                # 提取数据集名称
                rel_path = os.path.relpath(file_path, datasets_root)
                parts = rel_path.split(os.sep)
                
                if len(parts) >= 2:
                    # 格式：dataset/full.csv（如 BindingDB/full.csv）
                    dataset_name = parts[0]  # BindingDB, Davis, KIBA
                    full_dataset_files.append((file_path, dataset_name))
    
    logger.info(f"发现 {len(full_dataset_files)} 个完整数据集")
    for file_path, dataset_name in full_dataset_files:
        logger.info(f"  - {dataset_name}: {file_path}")
    
    # 如果指定了单个数据集，只处理该数据集
    if single_dataset:
        # 提取基础数据集名称
        base_dataset = single_dataset.split('_')[0]
        full_dataset_files = [(path, name) for path, name in full_dataset_files 
                             if name.lower() == base_dataset.lower()]
        logger.info(f"只处理指定数据集: {base_dataset}")
    
    # 逐个缓存完整数据集（避免内存占用过高）
    for idx, (dataset_path, dataset_name) in enumerate(full_dataset_files, 1):
        logger.info("=" * 80)
        logger.info(f"处理完整数据集 [{idx}/{len(full_dataset_files)}]: {dataset_name}")
        logger.info("=" * 80)
        
        try:
            cache.cache_dataset(dataset_path, dataset_name, protein_col=protein_col, 
                               batch_size=batch_size)
        except Exception as e:
            logger.error(f"缓存数据集 {dataset_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("=" * 80)
    logger.info("所有数据集缓存完成!")
    logger.info("=" * 80)


if __name__ == "__main__":
    import argparse

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="ESM feature cache builder")
    parser.add_argument("--datasets_root", type=str, default="/home/czb/Desktop/HierHGT-DTI/data",
                        help="datasets root directory")
    parser.add_argument("--cache_base_dir", type=str, default=os.path.join(SCRIPT_DIR, "esm_cache"),
                        help="cache output directory")
    parser.add_argument("--esm_model", type=str, default="esm2_t6_8M_UR50D",
                        help="ESM model name")
    parser.add_argument("--esm_layer", type=int, default=-1,
                        help="ESM layer index (-1 for last layer)")
    parser.add_argument("--protein_col", type=str, default="Protein",
                        help="protein sequence column name")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size for caching")
    parser.add_argument("--single_dataset", type=str, default=None,
                        help="only process one dataset, e.g. bindingdb_random")

    parser.add_argument("--protein_window_size", type=int, default=3,
                        help="window size for sequence edges")
    parser.add_argument("--no_build_graph", action="store_true",
                        help="only cache ESM features without building graphs")
    parser.add_argument("--disable_contact_edges", action="store_true",
                        help="disable contact edges; keep sequence+self-loop edges")
    parser.add_argument("--contact_top_k", type=int, default=10,
                        help="top-k contact edges per residue")
    parser.add_argument("--contact_min_prob", type=float, default=0.3,
                        help="minimum probability threshold for contact edges")
    parser.add_argument("--contact_prob_bins", type=str, default="0.5,0.7,0.9",
                        help="comma-separated contact-probability bin boundaries")
    parser.add_argument("--disable_predicted_contact", action="store_true",
                        help="disable ESM predicted contacts and use feature-similarity fallback")

    args = parser.parse_args()

    try:
        parsed_contact_bins = [float(x.strip()) for x in args.contact_prob_bins.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"Invalid --contact_prob_bins: {args.contact_prob_bins}") from e

    logger.info("=" * 80)
    logger.info("ESM feature batch cache")
    logger.info("=" * 80)
    logger.info(f"datasets_root: {args.data_root}")
    logger.info(f"cache_base_dir: {args.cache_base_dir}")
    logger.info(f"esm_model: {args.esm_model}")
    logger.info(f"esm_layer: {args.esm_layer}")
    logger.info(f"batch_size: {args.batch_size}")
    logger.info(f"protein_window_size: {args.protein_window_size}")
    logger.info(f"build_graph: {not args.no_build_graph}")
    logger.info(f"use_contact_edges: {not args.disable_contact_edges}")
    logger.info(f"contact_top_k: {args.contact_top_k}, contact_min_prob: {args.contact_min_prob}")
    logger.info(f"contact_prob_bins: {parsed_contact_bins}")
    logger.info(f"prefer_predicted_contact: {not args.disable_predicted_contact}")
    if args.single_dataset:
        logger.info(f"single_dataset: {args.single_dataset}")
    logger.info("=" * 80)

    cache_all_datasets(
        data_root=args.data_root,
        cache_base_dir=args.cache_base_dir,
        esm_model_name=args.esm_model,
        esm_layer=args.esm_layer,
        protein_col=args.protein_col,
        batch_size=args.batch_size,
        single_dataset=args.single_dataset,
        protein_window_size=args.protein_window_size,
        build_graph=not args.no_build_graph,
        use_contact_edges=not args.disable_contact_edges,
        contact_top_k=args.contact_top_k,
        contact_min_prob=args.contact_min_prob,
        contact_prob_bins=parsed_contact_bins,
        prefer_predicted_contact=not args.disable_predicted_contact,
    )

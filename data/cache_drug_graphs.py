import os
import sys
import pandas as pd
import torch
import logging
from tqdm import tqdm
import hashlib
import dgl

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit 不可用，无法构建药物图")

try:
    from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer
    DGLLIFE_AVAILABLE = True
except ImportError:
    DGLLIFE_AVAILABLE = False
    logging.warning("dgllife 不可用，将使用基本分子图构建功能")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_substructures(mol):
    """Extract BRICS substructures from a molecule.

    Returns:
        atom_to_sub: list of int, length = num_atoms. atom_to_sub[i] = substructure id for atom i.
        num_subs: int, number of substructures.

    Fallback: if BRICS finds 0 or 1 fragment, assign all atoms to a single substructure (id=0).
    """
    from rdkit.Chem import BRICS, AllChem

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return [0], 1

    # Try BRICS fragmentation
    try:
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        if len(brics_bonds) == 0:
            # No BRICS bonds found - all atoms in one substructure
            return [0] * num_atoms, 1

        # Get bond indices to cut
        bond_indices = []
        for (i, j), _ in brics_bonds:
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bond_indices.append(bond.GetIdx())

        if len(bond_indices) == 0:
            return [0] * num_atoms, 1

        # Fragment the molecule
        from rdkit.Chem import FragmentOnBonds, GetMolFrags
        fragmented = FragmentOnBonds(mol, bond_indices, addDummies=False)
        frag_atom_lists = GetMolFrags(fragmented, asMols=False)

        # Build atom_to_sub mapping
        atom_to_sub = [0] * num_atoms
        for sub_id, atom_list in enumerate(frag_atom_lists):
            for atom_idx in atom_list:
                if atom_idx < num_atoms:
                    atom_to_sub[atom_idx] = sub_id

        num_subs = len(frag_atom_lists)
        if num_subs == 0:
            return [0] * num_atoms, 1

        return atom_to_sub, num_subs

    except Exception:
        # Fallback: all atoms in one substructure
        return [0] * num_atoms, 1


class DrugGraphCache:
    def __init__(self, cache_base_dir="./drug_cache", use_dgllife=True, 
                 atom_featurizer=None, add_self_loop=True):
        """
        药物图缓存器
        
        参数:
            cache_base_dir: 缓存基础目录
            use_dgllife: 是否使用dgllife构建图
            atom_featurizer: 原子特征提取器
            add_self_loop: 是否添加自环
        """
        self.cache_base_dir = cache_base_dir
        self.use_dgllife = use_dgllife and DGLLIFE_AVAILABLE
        self.add_self_loop = add_self_loop
        self.bond_type_vocab = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        self.stereo_vocab = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
        ]
        self.edge_feat_dim = (
            len(self.bond_type_vocab) + 1 + 1 + len(self.stereo_vocab) + 1 + 1
        )
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit 不可用，无法构建药物图")
        
        if self.use_dgllife:
            if atom_featurizer is None:
                self.atom_featurizer = CanonicalAtomFeaturizer()
            else:
                self.atom_featurizer = atom_featurizer
            logger.info("使用dgllife构建药物图")
        else:
            logger.info("使用基本方法构建药物图")
        
        logger.info(f"缓存基础目录: {self.cache_base_dir}")
    
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

    def _get_drug_cache_path(self, smiles, dataset_name):
        """获取药物图缓存路径"""
        smiles_hash = hashlib.md5(smiles.encode()).hexdigest()
        base_dataset = self._normalize_dataset_name(dataset_name)
        cache_dir = os.path.join(self.cache_base_dir, base_dataset)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"drug_{smiles_hash}.bin")
    
    def _encode_bond_feature(self, bond, is_self_loop=False):
        """
        Encode edge feature:
        [bond_type_onehot(4), is_conjugated, is_aromatic, stereo_onehot(6), in_ring, is_self_loop]
        """
        if is_self_loop:
            return [0.0] * (self.edge_feat_dim - 1) + [1.0]

        if bond is None:
            return [0.0] * self.edge_feat_dim

        bond_type_one_hot = [0.0] * len(self.bond_type_vocab)
        bond_type = bond.GetBondType()
        if bond_type in self.bond_type_vocab:
            bond_type_one_hot[self.bond_type_vocab.index(bond_type)] = 1.0

        stereo_one_hot = [0.0] * len(self.stereo_vocab)
        stereo = bond.GetStereo()
        if stereo in self.stereo_vocab:
            stereo_one_hot[self.stereo_vocab.index(stereo)] = 1.0

        return bond_type_one_hot + [
            float(bond.GetIsConjugated()),
            float(bond.GetIsAromatic()),
        ] + stereo_one_hot + [
            float(bond.IsInRing()),
            0.0,
        ]

    def _attach_edge_features(self, graph, mol):
        """Attach edge_feat for all edges, including self loops."""
        src, dst = graph.edges()
        src = src.tolist()
        dst = dst.tolist()

        edge_features = []
        for u, v in zip(src, dst):
            if u == v:
                edge_features.append(self._encode_bond_feature(None, is_self_loop=True))
                continue

            bond = mol.GetBondBetweenAtoms(int(u), int(v))
            edge_features.append(self._encode_bond_feature(bond, is_self_loop=False))

        if edge_features:
            graph.edata['edge_feat'] = torch.tensor(edge_features, dtype=torch.float32)
        else:
            graph.edata['edge_feat'] = torch.zeros((0, self.edge_feat_dim), dtype=torch.float32)

    def _build_drug_graph_dgllife(self, smiles):
        """
        使用dgllife构建药物图
        
        参数:
            smiles: 药物分子的SMILES表示
            
        返回:
            dgl.DGLGraph: 药物图
            bool: 构建是否成功
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"RDKit无法解析SMILES: {smiles[:50]}...")
            return None, False
        
        try:
            # 构图但不自动加自环
            g = mol_to_bigraph(
                mol,
                node_featurizer=self.atom_featurizer,
                add_self_loop=False
            )
            
            # 统一使用'h'作为节点特征名
            if 'feat' in g.ndata:
                g.ndata['h'] = g.ndata['feat']
                del g.ndata['feat']
            elif 'h' not in g.ndata:
                logger.error(f"未找到节点特征，期望'h'或'feat'，实际得到 {list(g.ndata.keys())}")
                return None, False
            
            # 手动添加自环
            if self.add_self_loop:
                g = dgl.add_self_loop(g)
            self._attach_edge_features(g, mol)

            # --- SubPocket: extract BRICS substructures ---
            atom_to_sub, num_subs = extract_substructures(mol)
            g.ndata['atom_to_sub'] = torch.tensor(atom_to_sub, dtype=torch.long)
            g.num_subs = num_subs

            return g, True

        except Exception as e:
            logger.error(f"mol_to_bigraph错误，SMILES {smiles[:50]}...: {e}")
            return None, False
    
    def _build_drug_graph_basic(self, smiles):
        """
        使用基本方法构建药物图
        
        参数:
            smiles: 药物分子的SMILES表示
            
        返回:
            dgl.DGLGraph: 药物图
            bool: 构建是否成功
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"RDKit无法解析SMILES: {smiles[:50]}...")
            return None, False
        
        try:
            # 获取原子和键信息
            num_atoms = mol.GetNumAtoms()
            
            # 创建图
            g = dgl.graph(([], []))
            g.add_nodes(num_atoms)
            
            # 添加简单的原子特征（原子序数）
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append([atom.GetAtomicNum()])
            
            g.ndata['h'] = torch.tensor(atom_features, dtype=torch.float32)
            
            # 添加边（化学键）
            src_list = []
            dst_list = []
            
            for bond in mol.GetBonds():
                src = bond.GetBeginAtomIdx()
                dst = bond.GetEndAtomIdx()
                src_list.append(src)
                dst_list.append(dst)
                src_list.append(dst)
                dst_list.append(src)
            
            if src_list:
                g.add_edges(src_list, dst_list)
            
            # 添加自环
            if self.add_self_loop:
                g = dgl.add_self_loop(g)
            self._attach_edge_features(g, mol)

            # --- SubPocket: extract BRICS substructures ---
            atom_to_sub, num_subs = extract_substructures(mol)
            g.ndata['atom_to_sub'] = torch.tensor(atom_to_sub, dtype=torch.long)
            g.num_subs = num_subs

            return g, True

        except Exception as e:
            logger.error(f"构建药物图错误，SMILES {smiles[:50]}...: {e}")
            return None, False
    
    def _build_drug_graph(self, smiles):
        """
        构建药物图
        
        参数:
            smiles: 药物分子的SMILES表示
            
        返回:
            dgl.DGLGraph: 药物图
            bool: 构建是否成功
        """
        if self.use_dgllife:
            return self._build_drug_graph_dgllife(smiles)
        else:
            return self._build_drug_graph_basic(smiles)
    
    def _save_graph(self, graph, cache_path):
        """保存图到缓存"""
        try:
            dgl.save_graphs(cache_path, [graph])
        except Exception as e:
            logger.warning(f"保存图到缓存失败: {e}")
    
    def cache_dataset(self, dataset_path, dataset_name, smiles_col="SMILES",
                      batch_size=1, save_progress=True):
        """
        缓存数据集中的药物图
        
        参数:
            dataset_path: 数据集CSV文件路径
            dataset_name: 数据集名称（用于缓存子目录）
            smiles_col: SMILES列名
            batch_size: 批处理大小
            save_progress: 是否保存进度
        """
        logger.info(f"开始缓存数据集: {dataset_name}")
        logger.info(f"数据集路径: {dataset_path}")
        logger.info(f"批处理大小: {batch_size}")
        
        # 读取数据集
        df = pd.read_csv(dataset_path)
        logger.info(f"数据集包含 {len(df)} 个样本")
        
        # 获取唯一的SMILES
        unique_smiles = df[smiles_col].unique()
        logger.info(f"发现 {len(unique_smiles)} 个唯一药物分子")
        
        # 创建数据集特定的缓存目录（使用标准化名称，与 _get_drug_cache_path 一致）
        normalized_name = self._normalize_dataset_name(dataset_name)
        dataset_cache_dir = os.path.join(self.cache_base_dir, normalized_name)
        os.makedirs(dataset_cache_dir, exist_ok=True)
        logger.info(f"缓存目录: {dataset_cache_dir}")
        
        # 进度文件
        progress_file = os.path.join(dataset_cache_dir, "drug_progress.txt")
        processed_smiles = set()
        
        # 加载已处理的SMILES
        if save_progress and os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                for line in f:
                    smiles_hash = line.strip()
                    processed_smiles.add(smiles_hash)
            logger.info(f"已处理 {len(processed_smiles)} 个药物分子")
        
        # 缓存每个药物的图
        cached_count = 0
        skipped_count = 0
        failed_count = 0
        
        # 分批处理
        for i in tqdm(range(0, len(unique_smiles), batch_size), 
                     desc=f"缓存 {dataset_name}"):
            batch = unique_smiles[i:i+batch_size]
            
            for smiles in batch:
                smiles_hash = hashlib.md5(smiles.encode()).hexdigest()
                
                # 检查缓存
                cache_path = self._get_drug_cache_path(smiles, dataset_name)

                # 以真实缓存文件为准，进度文件仅作辅助。
                if os.path.exists(cache_path):
                    skipped_count += 1
                    continue
                if smiles_hash in processed_smiles:
                    logger.warning(f"检测到进度文件脏数据（缓存文件缺失），将重建: {smiles_hash}")
                    processed_smiles.discard(smiles_hash)

                # 构建药物图
                graph, success = self._build_drug_graph(smiles)
                
                if success and graph is not None:
                    # 保存到缓存
                    self._save_graph(graph, cache_path)
                    cached_count += 1
                    processed_smiles.add(smiles_hash)
                    
                    # 保存进度
                    if save_progress:
                        with open(progress_file, 'a') as f:
                            f.write(f"{smiles_hash}\n")
                else:
                    failed_count += 1
            
            # 定期清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"缓存完成: {cached_count} 个新缓存, {skipped_count} 个已存在, {failed_count} 个失败")
        logger.info(f"缓存目录: {dataset_cache_dir}")


def cache_all_datasets(datasets_root, cache_base_dir="./drug_cache", 
                      smiles_col="SMILES", batch_size=1, single_dataset=None,
                      use_dgllife=True):
    """
    缓存所有数据集的药物图
    
    参数:
        datasets_root: 数据集根目录
        cache_base_dir: 缓存基础目录
        smiles_col: SMILES列名
        batch_size: 批处理大小
        single_dataset: 只处理指定的数据集（格式：dataset_type_split，如 "bindingdb_random"）
        use_dgllife: 是否使用dgllife构建图
    """
    # 创建药物图缓存器
    cache = DrugGraphCache(
        cache_base_dir=cache_base_dir,
        use_dgllife=use_dgllife
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
    for idx, (file_path, dataset_name) in enumerate(full_dataset_files, 1):
        logger.info("=" * 80)
        logger.info(f"处理完整数据集 [{idx}/{len(full_dataset_files)}]: {dataset_name}")
        logger.info("=" * 80)
        
        try:
            cache.cache_dataset(file_path, dataset_name, smiles_col="SMILES", 
                               batch_size=batch_size)
        except Exception as e:
            logger.error(f"缓存数据集 {dataset_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info("=" * 80)
    logger.info("所有数据集缓存完成!")
    logger.info("=" * 80)


if __name__ == "__main__":
    import argparse

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="药物图批量缓存工具")
    parser.add_argument("--datasets_root", type=str, default="/home/czb/Desktop/HierHGT-DTI/data",
                        help="数据集根目录")
    parser.add_argument("--cache_base_dir", type=str, default=os.path.join(SCRIPT_DIR, "drug_cache"),
                        help="缓存基础目录")
    parser.add_argument("--smiles_col", type=str, default="SMILES",
                        help="SMILES列名")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批处理大小")
    parser.add_argument("--single_dataset", type=str, default=None,
                        help="只处理指定的数据集（格式：dataset_type_split，如 'bindingdb_random'）")
    parser.add_argument("--use_dgllife", action="store_true", default=True,
                        help="使用dgllife构建图（默认启用）")
    parser.add_argument("--no_dgllife", action="store_false", dest="use_dgllife",
                        help="不使用dgllife，使用基本方法构建图")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("药物图批量缓存工具")
    logger.info("=" * 80)
    logger.info(f"数据集根目录: {args.data_root}")
    logger.info(f"缓存基础目录: {args.cache_base_dir}")
    logger.info(f"SMILES列名: {args.smiles_col}")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info(f"使用dgllife: {args.use_dgllife}")
    if args.single_dataset:
        logger.info(f"只处理数据集: {args.single_dataset}")
    logger.info("=" * 80)
    
    # 缓存所有数据集
    cache_all_datasets(
        data_root=args.data_root,
        cache_base_dir=args.cache_base_dir,
        smiles_col=args.smiles_col,
        batch_size=args.batch_size,
        single_dataset=args.single_dataset,
        use_dgllife=args.use_dgllife
    )

# python cache_drug_graphs.py \
#     --datasets_root ../datesets \
#     --cache_base_dir ./drug_cache \
#     --batch_size 10

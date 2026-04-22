import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein


class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        self._drug_graph_cache = {}
        self._protein_cache = {}

    def _build_drug_graph(self, smiles):
        v_d = self.fc(smiles=smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        # 截断超过 max_drug_nodes 的分子
        if num_actual_nodes > self.max_drug_nodes:
            keep_ids = list(range(self.max_drug_nodes))
            v_d = v_d.subgraph(keep_ids)
            if '_ID' in v_d.ndata:
                v_d.ndata.pop('_ID')
            if '_ID' in v_d.edata:
                v_d.edata.pop('_ID')
            actual_node_feats = actual_node_feats[:self.max_drug_nodes]
            num_actual_nodes = self.max_drug_nodes
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        return v_d.add_self_loop()

    def _get_cached_drug_graph(self, smiles):
        graph = self._drug_graph_cache.get(smiles)
        if graph is None:
            graph = self._build_drug_graph(smiles)
            self._drug_graph_cache[smiles] = graph
        return graph.clone()

    def _get_cached_protein(self, protein_sequence):
        protein = self._protein_cache.get(protein_sequence)
        if protein is None:
            protein = integer_label_protein(protein_sequence)
            self._protein_cache[protein_sequence] = protein
        return protein.copy()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        smiles = self.df.iloc[index]['SMILES']
        protein_seq = self.df.iloc[index]['Protein']
        v_d = self._get_cached_drug_graph(smiles)
        v_p = self._get_cached_protein(protein_seq)
        y = self.df.iloc[index]["Y"]
        # y = torch.Tensor([y])
        return v_d, v_p, y


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches

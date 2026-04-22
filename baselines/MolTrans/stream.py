import numpy as np
import pandas as pd
import torch
from torch.utils import data
import json
import os

from sklearn.preprocessing import OneHotEncoder

from subword_nmt.apply_bpe import BPE
import codecs

# 使用相对于文件本身的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

vocab_path = os.path.join(current_dir, 'ESPF', 'protein_codes_uniprot.txt')
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv(os.path.join(current_dir, 'ESPF', 'subword_units_map_uniprot.csv'))

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

vocab_path = os.path.join(current_dir, 'ESPF', 'drug_codes_chembl.txt')
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv(os.path.join(current_dir, 'ESPF', 'subword_units_map_chembl.csv'))

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

max_d = 205
max_p = 545
 

def protein2emb_encoder(x):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)

    l = len(i1)
   
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
        
    return i, np.asarray(input_mask)

def drug2emb_encoder(x):
    max_d = 50
    #max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)
    
    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self._drug_emb_cache = {}
        self._protein_emb_cache = {}
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        #d = self.df.iloc[index]['DrugBank ID']
        d = self.df.iloc[index]['SMILES']
        p = self.df.iloc[index]['Target Sequence']
        
        #d_v = drug2single_vector(d)
        d_item = self._drug_emb_cache.get(d)
        if d_item is None:
            d_item = drug2emb_encoder(d)
            self._drug_emb_cache[d] = d_item
        p_item = self._protein_emb_cache.get(p)
        if p_item is None:
            p_item = protein2emb_encoder(p)
            self._protein_emb_cache[p] = p_item

        d_v, input_mask_d = d_item
        p_v, input_mask_p = p_item
        
        #print(d_v.shape)
        #print(input_mask_d.shape)
        #print(p_v.shape)
        #print(input_mask_p.shape)
        y = self.labels[index]
        return d_v, p_v, input_mask_d, input_mask_p, y

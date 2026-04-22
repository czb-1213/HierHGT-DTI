"""Adapted DO-GMA model for the HierHGT-DTI fixed-split benchmark."""

from __future__ import annotations

import math
from itertools import repeat
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn import GCN
from torch._jit_internal import Optional
from torch.nn import init
from torch.nn.parameter import Parameter


def _ntuple(n: int):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class DOConv2d(nn.Module):
    """Depthwise over-parameterized convolution from the official DO-GMA code."""

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
        "D_mul",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        D_mul=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError(f"Unsupported padding_mode={padding_mode}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))

        kernel_h, kernel_w = self.kernel_size
        self.D_mul = kernel_h * kernel_w if D_mul is None or kernel_h * kernel_w <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if kernel_h * kernel_w > 1:
            self.D = Parameter(torch.Tensor(in_channels, kernel_h * kernel_w, self.D_mul))
            init_zero = np.zeros([in_channels, kernel_h * kernel_w, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(kernel_h * kernel_w, dtype=torch.float32), (1, kernel_h * kernel_w, kernel_h * kernel_w))
            d_diag = eye.repeat((in_channels, 1, self.D_mul // (kernel_h * kernel_w)))
            if self.D_mul % (kernel_h * kernel_w) != 0:
                zeros = torch.zeros([in_channels, kernel_h * kernel_w, self.D_mul % (kernel_h * kernel_w)])
                self.d_diag = Parameter(torch.cat([d_diag, zeros], dim=2), requires_grad=False)
            else:
                self.d_diag = Parameter(d_diag, requires_grad=False)
        else:
            self.D = None
            self.d_diag = None

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def _conv_forward(self, input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input_tensor, self._padding_repeated_twice, mode=self.padding_mode),
                weight,
                self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(input_tensor, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        kernel_h, kernel_w = self.kernel_size
        weight_shape = (self.out_channels, self.in_channels // self.groups, kernel_h, kernel_w)
        if kernel_h * kernel_w > 1:
            D = self.D + self.d_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))
            weight = torch.reshape(torch.einsum("ims,ois->oim", D, W), weight_shape)
        else:
            weight = torch.reshape(self.W, weight_shape)
        return self._conv_forward(input_tensor, weight)


class GatedProjection(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_c = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention_c(self.attention_a(x) * self.attention_b(x))


class CrossAttentionFusion(nn.Module):
    def __init__(self, num_head: int = 2, num_hid: int = 256, dropout: float = 0.2):
        super().__init__()
        if num_hid % num_head != 0:
            raise ValueError(f"num_hid={num_hid} must be divisible by num_head={num_head}")
        self.num_hid = num_hid
        self.num_head = num_head
        self.head_dim = num_hid // num_head
        self.linear_v = nn.Linear(num_hid, num_hid)
        self.linear_k = nn.Linear(num_hid, num_hid)
        self.linear_q = nn.Linear(num_hid, num_hid)
        self.dropout = nn.Dropout(dropout)

    def _scaled_attention(self, key: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        return self.dropout(F.softmax(scores, dim=-1))

    def forward(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)

        value = self.linear_v(value).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)

        att_map = self._scaled_attention(key, query) + self._scaled_attention(value, query)
        fused_source = value + key
        attended = torch.matmul(att_map, fused_source)
        head_logits = (attended * query).sum(dim=2)
        return head_logits.reshape(batch_size, -1)


class MolecularGCN(nn.Module):
    def __init__(
        self,
        in_feats: int = 75,
        dim_embedding: int = 128,
        padding: bool = True,
        hidden_feats: Tuple[int, ...] = (128, 128, 128),
    ):
        super().__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=list(hidden_feats), activation=None)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph) -> torch.Tensor:
        node_feats = batch_graph.ndata.pop("h")
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        return node_feats.view(batch_graph.batch_size, -1, self.output_feats)


class ProteinConv2d(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        output_channels: int = 128,
        grid_height: int = 30,
        kernel_size: int = 3,
        padding: bool = True,
    ):
        super().__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        self.embedding_dim = embedding_dim
        self.grid_height = grid_height
        self.output_channels = output_channels
        self.conv = DOConv2d(in_channels=embedding_dim, out_channels=output_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens.long()).transpose(1, 2).contiguous()
        batch_size, channels, seq_len = x.shape
        if seq_len % self.grid_height != 0:
            raise ValueError(f"Protein sequence length {seq_len} is not divisible by {self.grid_height}")
        grid_width = seq_len // self.grid_height
        x = x.view(batch_size, channels, self.grid_height, grid_width)
        x = self.bn(F.relu(self.conv(x)))
        return x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.output_channels)


class DrugConv2d(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        output_channels: int = 32,
        grid_height: int = 290,
        grid_width: int = 4,
        padding: bool = True,
    ):
        super().__init__()
        if padding:
            self.embedding = nn.Embedding(65, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(65, embedding_dim)
        self.embedding_dim = embedding_dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.output_channels = output_channels
        self.conv = DOConv2d(
            in_channels=embedding_dim,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens.long()).transpose(1, 2).contiguous()
        batch_size, channels, seq_len = x.shape
        expected = self.grid_height * self.grid_width
        if seq_len != expected:
            raise ValueError(f"Drug sequence length {seq_len} does not match expected grid size {expected}")
        x = x.view(batch_size, channels, self.grid_height, self.grid_width)
        x = self.bn(F.relu(self.conv(x)))
        return x.permute(0, 2, 3, 1).contiguous().view(batch_size, self.grid_height, -1)


class MLPDecoder(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, out_dim: int = 128, binary: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        return self.fc4(x)


class DOGMAModel(nn.Module):
    """HierHGT-DTI-adapted DO-GMA with dynamic drug-length support."""

    def __init__(
        self,
        max_drug_nodes: int,
        protein_grid_height: int = 30,
        drug_grid_width: int = 4,
        drug_in_feats: int = 75,
        drug_embedding: int = 128,
        drug_sequence_embedding: int = 64,
        protein_embedding: int = 64,
        graph_hidden_feats: Tuple[int, ...] = (128, 128, 128),
        num_filters: int = 128,
        mlp_hidden_dim: int = 512,
        mlp_out_dim: int = 128,
        num_head: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.drug_graph_encoder = MolecularGCN(
            in_feats=drug_in_feats,
            dim_embedding=drug_embedding,
            padding=True,
            hidden_feats=graph_hidden_feats,
        )
        self.drug_sequence_encoder = DrugConv2d(
            embedding_dim=drug_sequence_embedding,
            output_channels=num_filters // 4,
            grid_height=max_drug_nodes,
            grid_width=drug_grid_width,
            padding=True,
        )
        self.protein_encoder = ProteinConv2d(
            embedding_dim=protein_embedding,
            output_channels=num_filters,
            grid_height=protein_grid_height,
            kernel_size=3,
            padding=True,
        )
        self.gate = GatedProjection(input_dim=128, hidden_dim=64, output_dim=256, dropout=dropout)
        self.attention = CrossAttentionFusion(num_head=num_head, num_hid=256, dropout=0.2)
        self.decoder = MLPDecoder(in_dim=256, hidden_dim=mlp_hidden_dim, out_dim=mlp_out_dim, binary=1)

    def forward(self, drug_graph, drug_tokens: torch.Tensor, protein_tokens: torch.Tensor) -> torch.Tensor:
        graph_repr = self.gate(self.drug_graph_encoder(drug_graph))
        drug_repr = self.gate(self.drug_sequence_encoder(drug_tokens))
        protein_repr = self.gate(self.protein_encoder(protein_tokens))
        fused = self.attention(graph_repr, drug_repr, protein_repr)
        return self.decoder(fused).squeeze(-1)

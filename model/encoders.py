import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GraphConv, GATConv, SAGEConv


class SharedNodeStem(nn.Module):
    """
    Shared node stem used by both modalities.
    Only aligns feature spaces, without any message passing.
    """

    def __init__(self, in_dim=74, out_dim=128, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.norm(feats)
        x = self.proj(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class _LegacyNodeEncoder(nn.Module):
    """
    Legacy GNN encoder kept for backward compatibility.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        n_layers=2,
        dropout=0.1,
        gnn_type="GCN",
        aggregator_type="mean",
        use_attention=True,
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.aggregator_type = aggregator_type
        self.layer_norms = nn.ModuleList()
        self.gnn_layers = nn.ModuleList()

        if gnn_type == "GCN":
            self.layer_norms.append(nn.LayerNorm(in_dim))
            for _ in range(1, n_layers):
                self.layer_norms.append(nn.LayerNorm(hidden_dim))

            self.gnn_layers.append(
                GraphConv(in_dim, hidden_dim, norm="both", activation=F.relu, allow_zero_in_degree=True)
            )
            for _ in range(1, n_layers - 1):
                self.gnn_layers.append(
                    GraphConv(hidden_dim, hidden_dim, norm="both", activation=F.relu, allow_zero_in_degree=True)
                )
            if n_layers > 1:
                self.gnn_layers.append(
                    GraphConv(hidden_dim, out_dim, norm="both", activation=None, allow_zero_in_degree=True)
                )
            else:
                self.gnn_layers.append(
                    GraphConv(in_dim, out_dim, norm="both", activation=None, allow_zero_in_degree=True)
                )
        elif gnn_type == "GAT":
            num_heads = 4
            head_dim = hidden_dim // num_heads
            self.layer_norms.append(nn.LayerNorm(in_dim))
            for _ in range(1, n_layers):
                self.layer_norms.append(nn.LayerNorm(hidden_dim))

            self.gnn_layers.append(
                GATConv(in_dim, head_dim, num_heads=num_heads, activation=F.relu, allow_zero_in_degree=True)
            )
            for _ in range(1, n_layers - 1):
                self.gnn_layers.append(
                    GATConv(hidden_dim, head_dim, num_heads=num_heads, activation=F.relu, allow_zero_in_degree=True)
                )
            if n_layers > 1:
                self.gnn_layers.append(
                    GATConv(
                        hidden_dim,
                        out_dim // num_heads,
                        num_heads=num_heads,
                        activation=None,
                        allow_zero_in_degree=True,
                    )
                )
            else:
                self.gnn_layers.append(
                    GATConv(
                        in_dim,
                        out_dim // num_heads,
                        num_heads=num_heads,
                        activation=None,
                        allow_zero_in_degree=True,
                    )
                )
        elif gnn_type == "GraphSAGE":
            self.layer_norms.append(nn.LayerNorm(in_dim))
            for _ in range(1, n_layers):
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.gnn_layers.append(SAGEConv(in_dim, hidden_dim, aggregator_type=aggregator_type, activation=F.relu))
            for _ in range(1, n_layers - 1):
                self.gnn_layers.append(
                    SAGEConv(hidden_dim, hidden_dim, aggregator_type=aggregator_type, activation=F.relu)
                )
            if n_layers > 1:
                self.gnn_layers.append(
                    SAGEConv(hidden_dim, out_dim, aggregator_type=aggregator_type, activation=None)
                )
            else:
                self.gnn_layers.append(SAGEConv(in_dim, out_dim, aggregator_type=aggregator_type, activation=None))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

    def forward(self, g, feats):
        device = feats.device
        if g.device != device:
            g = g.to(device)

        h = feats
        for i, layer in enumerate(self.gnn_layers):
            h = self.layer_norms[i](h)
            h = layer(g, h)
            if i < len(self.gnn_layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
            if self.gnn_type == "GAT" and i < len(self.gnn_layers) - 1:
                h = h.reshape(h.shape[0], -1)

        if self.gnn_type == "GAT":
            h = h.reshape(h.shape[0], -1)
        return h


class AtomEncoder(_LegacyNodeEncoder):
    def __init__(
        self,
        in_dim=74,
        hidden_dim=128,
        out_dim=64,
        n_layers=2,
        dropout=0.1,
        gnn_type="GCN",
        aggregator_type="mean",
        use_attention=True,
    ):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            aggregator_type=aggregator_type,
            use_attention=use_attention,
        )


class ResidueEncoder(_LegacyNodeEncoder):
    def __init__(
        self,
        in_dim=320,
        hidden_dim=128,
        out_dim=64,
        n_layers=2,
        dropout=0.1,
        gnn_type="GCN",
        aggregator_type="mean",
        use_attention=True,
    ):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            aggregator_type=aggregator_type,
            use_attention=use_attention,
        )

import torch
from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero


class GNNEncoder(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv_in = SAGEConv((-1, -1), hidden_channels, aggr="sum")
        self.conv_med = SAGEConv((-1, -1), hidden_channels, aggr="sum")
        self.conv_out = SAGEConv((-1, -1), out_channels, aggr="sum")
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.dropout(self.act(self.conv_in(x, edge_index)))
        for _ in range(2):
            x = self.dropout(self.act(self.conv_med(x, edge_index)))
        x = self.conv_out(x, edge_index)
        return x


class EdgeClassifier(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict["drug"][row], z_dict["protein"][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class GeNNiusModel(nn.Module):
    def __init__(self, hidden_channels: int, metadata, dropout: float = 0.2):
        super().__init__()
        encoder = GNNEncoder(hidden_channels, hidden_channels, dropout=dropout)
        self.encoder = to_hetero(encoder, metadata, aggr="sum")
        self.decoder = EdgeClassifier(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        out = self.decoder(z_dict, edge_label_index)
        return z_dict, out

"""
Spatio-Temporal Graph Neural Network for flight delay prediction.

Architecture:
- Spatial: 2-hop GCN to capture airport dependency propagation
- Readout: MLP head producing a per-node delay scalar

Input features (5-dim):
  [congestion, visibility_norm, wind_norm, flight_category_ordinal, precip_severity]

Output: log(1 + delay_minutes) when trained weights are loaded.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class STGNN(torch.nn.Module):
    def __init__(self, in_channels: int = 5, hidden_channels: int = 32, out_channels: int = 1):
        super(STGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

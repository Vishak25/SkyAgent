
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class STGNN(torch.nn.Module):
    """
    Spatio-Temporal Graph Neural Network for Flight Delay Prediction.

    Architecture:
    - Spatial: 2-hop GCN to capture airport dependency propagation.
    - Readout: MLP head producing a per-node delay scalar.

    Input features (5-dim):
      [congestion, visibility_norm, wind_norm, flight_category_ordinal, precip_severity]

    The 5th feature (precip_severity) encodes winter ops / precipitation impact:
      0.0 = clear, 0.3 = rain, 0.65 = snow, 0.95 = freezing rain, 1.0 = severe ice/TS.
    """

    def __init__(self, in_channels: int = 5, hidden_channels: int = 32, out_channels: int = 1):
        super(STGNN, self).__init__()
        # Spatial Layer 1
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Spatial Layer 2
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # Readout MLP
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index):
        # 1. Graph Convolution (Spatial Dependency)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # 2. Second Hop (Propagation)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # 3. Readout / Prediction
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x

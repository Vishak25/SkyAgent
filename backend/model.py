
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class STGNN(torch.nn.Module):
    """
    Spatio-Temporal Graph Neural Network for Flight Delay Prediction.
    
    Architecture:
    - Spatial: Graph Convolutional Network (GCN) or Graph Attention Network (GAT) to capture airport dependencies.
    - Temporal: Process sequential snapshots of the graph (simplified here to feature fusion for the MVP).
    
    Input:
    - x: Node features (Airport attributes: capacity, weather, congestion). Shape: [Num_Nodes, Num_Features]
    - edge_index: Adjacency list (Flight connections). Shape: [2, Num_Edges]
    - edge_attr: Edge features (Flight attributes: distance, aircraft type). Shape: [Num_Edges, Edge_Features]
    
    Output:
    - out: Predicted delay for each node (or specific target node).
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(STGNN, self).__init__()
        # Spatial Layer 1
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Spatial Layer 2
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Temporal/Readout Layer (Simplified for real-time inference on single snapshot)
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
        
        # Output is often a regression value (delay in minutes)
        return x

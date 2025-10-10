import torch.nn as nn
from torch_geometric.nn import GATConv


class TemporalGAT(nn.Module):
    """
    Single-layer Graph Attention Network using PyTorch Geometric.
    """
    def __init__(self, in_features, out_features, n_heads=1, dropout=0.1, concat=True):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            n_heads: Number of attention heads
            dropout: Dropout rate
            concat: Whether to concatenate (True) or average (False) attention heads
        """
        super(TemporalGAT, self).__init__()

        self.gat = GATConv(
            in_channels=in_features,
            out_channels=out_features,
            heads=n_heads,
            dropout=dropout,
            concat=concat
        )

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (n_nodes, in_features)
            edge_index: Graph connectivity in COO format (2, num_edges)

        Returns:
            out: Updated node features
            attention_weights: Attention weights (num_edges, n_heads) or (num_edges,)
        """
        out, (edge_index, attention_weights) = self.gat(x, edge_index, return_attention_weights=True)

        return out, (edge_index,attention_weights)

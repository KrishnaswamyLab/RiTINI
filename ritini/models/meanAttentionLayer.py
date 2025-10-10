import torch
import torch.nn as nn

##TODO: Verify this Layer
class MeanAttentionLayer(nn.Module):
    """
    Averages multiple attention heads instead of concatenating them.
    GAT outputs (num_nodes, n_heads * out_features), this layer reshapes and averages
    to produce (num_nodes, out_features).
    """
    def __init__(self, n_heads, out_features):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features

    def forward(self, x):
        # Handle tuple output from GAT
        if isinstance(x, tuple): #This is because the GAT returns a tuple of: (features, attention_weights)
            x = x[0]

        # x shape: (num_nodes, n_heads * out_features)
        # Reshape to (num_nodes, n_heads, out_features)
        num_nodes = x.shape[0]
        x = x.view(num_nodes, self.n_heads, self.out_features)

        # Average across heads: (num_nodes, n_heads, out_features) -> (num_nodes, out_features)
        x = x.mean(dim=1)

        return x
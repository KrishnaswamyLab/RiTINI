from torch_geometric.nn import GATConv
import torch.nn as nn

class GATConvWithAttention(GATConv):
    def __init__(self, in_features, out_features, n_heads, feat_dropout=0.0, attn_dropout=0.0, 
                 negative_slope=0.2, residual=False, activation=None, bias=True,concat=False):
        super(GATConvWithAttention, self).__init__(
            in_channels=in_features, 
            out_channels=out_features, 
            heads=n_heads,
            negative_slope=negative_slope, 
            dropout=attn_dropout,  # This is attention dropout in PyG
            bias=bias,
            concat=concat
        )
        
        
        # Store additional parameters that PyG doesn't handle directly
        self.feat_dropout = feat_dropout
        self.residual = residual
        self.activation = activation
        
        # Setup residual connection if needed
        if residual:
            if in_features != out_features * n_heads:
                self.residual_fc = nn.Linear(in_features, out_features * n_heads, bias=False)
            else:
                self.residual_fc = None
        
        # Setup feature dropout
        if feat_dropout > 0:
            self.feat_dropout = nn.Dropout(feat_dropout)
        else:
            self.feat_dropout = None
    
    def forward(self, x, edge_index):
        
        # Apply feature dropout if specified
        if self.feat_dropout is not None:
            x = self.feat_dropout(x)

        # Get GAT output with attention
        x_out, (edge_index_attn, attention_weights) = super().forward(
            x, edge_index, return_attention_weights=True
        )

        # Apply residual connection if specified
        if self.residual:
            if self.residual_fc is not None:
                x_out = x_out + self.residual_fc(x)
            else:
                x_out = x_out + x
            
        
        # Apply activation
        if self.activation is not None:
            x_out = self.activation(x_out)
        

        return x_out, (edge_index_attn, attention_weights)

    def _get_attention_weights(self):
        return self._last_attention
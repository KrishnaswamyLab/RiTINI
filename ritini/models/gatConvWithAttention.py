
from torch_geometric.nn import GATConv
import torch.nn as nn

class GATConvWithAttention(GATConv):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0, 
                 negative_slope=0.2, residual=False, activation=None, bias=True):
        super(GATConvWithAttention, self).__init__(
            in_channels=in_feats, 
            out_channels=out_feats, 
            heads=num_heads,
            negative_slope=negative_slope, 
            dropout=attn_drop,  # This is attention dropout in PyG
            bias=bias
        )
        
        
        # Store additional parameters that PyG doesn't handle directly
        self.feat_drop = feat_drop
        self.residual = residual
        self.activation = activation
        
        # Setup residual connection if needed
        if residual:
            if in_feats != out_feats * num_heads:
                self.residual_fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
            else:
                self.residual_fc = None
        
        # Setup feature dropout
        if feat_drop > 0:
            self.feat_dropout = nn.Dropout(feat_drop)
        else:
            self.feat_dropout = None

    # def forward(self, x, edge_index, get_attention=False):
    #     # Apply feature dropout if specified
    #     if self.feat_dropout is not None:
    #         x = self.feat_dropout(x)
        
    #     # Store input for residual connection
    #     residual_input = x if self.residual else None
        
    #     # Forward pass through GAT
    #     if get_attention:
    #         output, attention_weights = super().forward(x, edge_index, return_attention_weights=True)
    #     else:
    #         output = super().forward(x, edge_index, return_attention_weights=False)

    #     output = self.activation(output)
        
    #     # Return appropriate format
    #     if get_attention:
    #         return output, attention_weights
    #     else:
    #         return output

    
    def forward(self, x, edge_index, get_attention=False):
        print(f"Input x type: {type(x)}, shape: {x.shape}")
        
        # Get GAT output with attention
        x_out = super().forward(x, edge_index, return_attention_weights=True)
        
        # Handle tuple unpacking
        if isinstance(x_out, tuple):
            x_out, (edge_index_attn, attention_weights) = x_out
            self._last_attention = attention_weights
        else:
            self._last_attention = None
            
        print(f"After GAT x_out type: {type(x_out)}, shape: {x_out.shape}")
        
        # Apply activation to the tensor (not tuple!)
        print(f"About to apply activation to: {type(x_out)}")
        if self.activation is not None:
            x_activated = self.activation(x_out)
        else:
            x_activated = x_out
        print(f"After activation: {type(x_activated)}")
        
        # Get attention weights
        attention_weights = self._last_attention
        print(f"Attention type: {type(attention_weights)}")
        
        # Return based on get_attention flag
        if get_attention:
            result = (x_activated, attention_weights)
            print(f"Final result type (with attention): {type(result)}")
            return result
        else:
            # For your PygSequential, always return both
            result = (x_activated, attention_weights)
            print(f"Final result type (sequential): {type(result)}")
            return result
    

    def _get_attention_weights(self):
        return self._last_attention
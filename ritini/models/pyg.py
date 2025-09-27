__all__ = ['MeanAttentionLayer', 'SumAttentionLayer', 'PygSequential', 'PygSAGEConv', 'PygTAGConv', 'PygPNAConv', 'GODE']

import itertools
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, SAGEConv, TAGConv, PNAConv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree


from ..utils.utils import is_list_like

## TODO: THIS IS LIKELY WRONG
class MeanAttentionLayer(nn.Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis
        
    def forward(self, x, attention_weights=None):
        # Don't reduce feature dimension, just pass through
        return x

class SumAttentionLayer(nn.Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis
    def forward(self, x):
        return torch.sum(x, axis=self.axis)

class PygSequential(nn.Module):
    def __init__(self, *args):
        super(PygSequential, self).__init__()
        self.layers = nn.ModuleList(args)

    def forward(self, x, edge_index, **kwargs):
        # First layer returns x and attention
        x, attention_weights = self.layers[0](x, edge_index)
        
        # Second layer uses attention
        x = self.layers[1](x, attention_weights)
        
        return x

# class GATConvWithAttention(GATConv):
#     def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0, 
#                  negative_slope=0.2, residual=False, activation=None, bias=True):
#         super(GATConvWithAttention, self).__init__(in_feats, out_feats, num_heads, 
#                                          negative_slope=negative_slope, dropout=attn_drop, bias=bias)

#     def forward(self, x, edge_index, get_attention=False):
#         if get_attention:
#             return super().forward(x, edge_index, return_attention_weights=True)
#         else:
#             return super().forward(x, edge_index)

class PygSAGEConv(SAGEConv):
    def __init__(self, in_feats, out_feats, aggregator_type='mean', bias=True, normalize=False, **kwargs):
        aggr = aggregator_type if aggregator_type in ['mean', 'max', 'add'] else 'mean'
        super(SAGEConv, self).__init__(
            in_feats, out_feats, aggr=aggr, bias=bias, normalize=normalize, **kwargs
        )

    def forward(self, x, edge_index, **kwargs):
        return super().forward(x, edge_index, **kwargs)

class PygTAGConv(TAGConv):
    def __init__(self, in_feats, out_feats, k=2, bias=True, **kwargs):
         super(TAGConv, self).__init__(in_feats, out_feats, K=k, bias=bias, **kwargs)

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        return super().forward(x, edge_index, edge_weight, **kwargs)

class PygPNAConv(PNAConv):
    def __init__(
            self, in_size, out_size, aggregators, scalers, deg=None,
            dropout=0.0, num_towers=1, edge_feat_size=0, **kwargs
        ):
        super(PNAConv, self).__init__(
            in_channels=in_size, out_channels=out_size,
            aggregators=aggregators, scalers=scalers, deg=deg,
            dropout=dropout, towers=num_towers,
            edge_dim=edge_feat_size if edge_feat_size > 0 else None,
            **kwargs
        )

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        return super().forward(x, edge_index, edge_attr, **kwargs)

class GODE(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, activation=nn.Tanh):
        super(GODE, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads

        num_heads = num_heads if is_list_like(num_heads) else [num_heads]

        for idx, module in enumerate([
            PygGATConv(
                in_feats=in_feats, out_feats=out_feats,
                num_heads=n_heads,
                dropout=0.0
            ) for i, n_heads in enumerate(num_heads)
        ]):
            self.add_module(str(idx), module)

    def forward(self, x, edge_index, get_attention=False):
        attns = []
        for layer in self:
            if get_attention:
                result = layer(x, edge_index, get_attention)
                if isinstance(result, tuple):
                    x, attn = result
                    attns.append(attn)
                else:
                    x = result
            else:
                x = layer(x, edge_index)

        if get_attention:
            return x, attns
        return x
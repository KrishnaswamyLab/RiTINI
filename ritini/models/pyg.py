__all__ = ['MeanAttentionLayer', 'SumAttentionLayer', 'PyGSequential', 'PyGGATConv', 'PyGSAGEConv', 'PyGTAGConv', 'PyGPNAConv', 'GODE']

import itertools, math, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torchdiffeq

from torch_geometric.nn import GATConv, SAGEConv, TAGConv, PNAConv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree

from typing import Callable, Union, Literal

class MeanAttentionLayer(nn.Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis
    def forward(self, x):
        return torch.mean(x, axis=self.axis)

class SumAttentionLayer(nn.Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis
    def forward(self, x):
        return torch.sum(x, axis=self.axis)

class PyGSequential(nn.Module):
    def __init__(self, *args):
        super(PyGSequential, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, x, edge_index, **kwargs):
        for module in self:
            x = module(x, edge_index)
        return x

class PyGGATConv(GATConv):
    def __init__(
            self, in_feats, out_feats, num_heads,
            dropout=0.0, negative_slope=0.2, residual=False,
            bias=True, **kwargs
        ):
         super(PyGGATConv, self).__init__(
             in_feats, out_feats, heads=num_heads, dropout=dropout,
             negative_slope=negative_slope, add_self_loops=True, bias=bias, **kwargs
            )

    def forward(self, x, edge_index, get_attention=False, **kwargs):
        if get_attention:
            return super().forward(x, edge_index, return_attention_weights=True, **kwargs)
        else:
            return super().forward(x, edge_index, **kwargs)

class PyGSAGEConv(SAGEConv):
    def __init__(self, in_feats, out_feats, aggregator_type='mean', bias=True, normalize=False, **kwargs):
        aggr = aggregator_type if aggregator_type in ['mean', 'max', 'add'] else 'mean'
        super(PyGSAGEConv, self).__init__(
            in_feats, out_feats, aggr=aggr, bias=bias, normalize=normalize, **kwargs
        )

    def forward(self, x, edge_index, **kwargs):
        return super().forward(x, edge_index, **kwargs)

class PyGTAGConv(TAGConv):
    def __init__(self, in_feats, out_feats, k=2, bias=True, **kwargs):
         super(PyGTAGConv, self).__init__(in_feats, out_feats, K=k, bias=bias, **kwargs)

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        return super().forward(x, edge_index, edge_weight, **kwargs)

class PyGPNAConv(PNAConv):
    def __init__(
            self, in_size, out_size, aggregators, scalers, deg=None,
            dropout=0.0, num_towers=1, edge_feat_size=0, **kwargs
        ):
        super(PyGPNAConv, self).__init__(
            in_channels=in_size, out_channels=out_size,
            aggregators=aggregators, scalers=scalers, deg=deg,
            dropout=dropout, towers=num_towers,
            edge_dim=edge_feat_size if edge_feat_size > 0 else None,
            **kwargs
        )

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        return super().forward(x, edge_index, edge_attr, **kwargs)

from .utils import is_list_like

class GODE(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, activation=nn.Tanh):
        super(GODE, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads

        num_heads = num_heads if is_list_like(num_heads) else [num_heads]

        for idx, module in enumerate([
            PyGGATConv(
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
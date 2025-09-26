__all__ = ['GCNLayer', 'GCN']

import math, numpy as np
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .graph import pyg_norm
from typing import Callable


class GCNLayer(GCNConv):
    def __init__(
        self, in_feats:int, out_feats:int,
        activation:Callable[[torch.Tensor], torch.Tensor]=None,
        dropout:int=0, bias:bool=True, **kwargs
    ):
        super().__init__(in_feats, out_feats, bias=bias, **kwargs)

        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x, edge_index, edge_weight=None):
        if self.dropout:
            x = self.dropout(x)

        x = super().forward(x, edge_index, edge_weight)

        if self.activation:
            x = self.activation(x)

        return x

class GCN(nn.Module):
    def __init__(
        self,
        num_layers:int,
        in_feats:int, hidden_feats:int, out_feats:int,
        activation:Callable, dropout:int, bias=True
    ):

        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer(in_feats, hidden_feats, activation, dropout))

        for i in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_feats, hidden_feats, activation, dropout))

        self.layers.append(GCNLayer(hidden_feats, out_feats, None, 0.))

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        return x

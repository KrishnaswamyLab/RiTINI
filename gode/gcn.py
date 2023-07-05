# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_gcn.ipynb.

# %% auto 0
__all__ = ['GCNLayer', 'GCN']

# %% ../nbs/02_gcn.ipynb 4
import math, numpy as np
import dgl, dgl.function as fn
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .graph import dgl_norm
from typing import Callable


class GCNLayer(nn.Module):
    def __init__(
        self, g:dgl.DGLGraph, 
        in_feats:int, out_feats:int, 
        activation:Callable[[torch.Tensor], torch.Tensor],
        dropout:int, bias:bool=True, msg_fn:Callable=fn.sum
    ):
        super().__init__()
        self.g = g
        self.msg_fn = msg_fn
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.

        if 'norm' not in g.ndata:            
            self.g = dgl_norm(g)
            
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)

        # normalization by square root of src degree
        h = h * self.g.ndata['norm']
        self.g.ndata['h'] = h
        self.g.update_all(
            fn.copy_src(src='h', out='m'), 
            self.msg_fn(msg='m', out='h')
        )
        
        h = self.g.ndata.pop('h')
        
        # normalization by square root of dst degree
        h = h * self.g.ndata['norm']
        
        # bias
        if self.bias is not None:
            h = h + self.bias
        
        if self.activation:
            h = self.activation(h)
        
        return h

# %% ../nbs/02_gcn.ipynb 5
class GCN(nn.Module):
    def __init__(
        self, 
        num_layers:int, 
        g:dgl.DGLGraph, 
        in_feats:int, hidden_feats:int, out_feats:int, 
        activation:Callable, dropout:int, bias=True
    ):
        
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer(g, in_feats, hidden_feats, activation, dropout))

        for i in range(num_layers - 2):
            self.layers.append(GCNLayer(g, hidden_feats, hidden_feats, activation, dropout))

        self.layers.append(GCNLayer(g, hidden_feats, out_feats, None, 0.))

    def set_graph(self, g):
        for l in self.layers:
            l.g = g

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h

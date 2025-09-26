__all__ = ['GDEFunc', 'ControlledGDEFunc', 'PyGGDEFunc']

import torch_geometric
from torch_geometric.data import Data
import torch, torch.nn as nn
from .gcn import GCNLayer
from .data import augment_with_time
from typing import Callable


class GDEFunc(nn.Module):
    def __init__(
        self, 
        gnn:nn.Module, 
        augment:bool=False, 
        augment_size:int=2
    ):
        """General GDE function class. To be passed to an ODEBlock"""
        super().__init__()
        self.gnn = gnn
        
        # Number of function calls
        self.nfe = 0
        
        # Whether or not to augment input tensor x
        self.augment = augment
        
        # Dimensions of 0s to augment x with (as well as the time vector t)
        self.augment_size = augment_size
    
    def set_graph_data(self, edge_index, num_nodes=None):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
            
    def forward(self, t, x):
        self.nfe += 1
        
        x = augment_with_time(x, t, self.augment_size, self.augment)  
        
        # NOTE: technically dxdt
        if hasattr(self, 'edge_index'):
            x = self.gnn(x, self.edge_index)
        else:
            x = self.gnn(x)
        return x
class ControlledGDEFunc(GDEFunc):
    def __init__(self, gnn:nn.Module):
        """ Controlled GDE version. Input information is preserved longer via hooks to input node features X_0, 
            affecting all ODE function steps. Requires assignment of '.h0' before calling .forward"""
        super().__init__(gnn)
        self.nfe = 0
            
    def forward(self, t, x):
        self.nfe += 1
        x = torch.cat([x, self.h0], 1)
        x = self.gnn(x)
        return x
    

class PyGGDEFunc(nn.Module):
    def __init__(
        self, 
        gnn:nn.Module, 
        augment:bool=False, 
        augment_size:int=2
    ):
        """General GDE function class. To be passed to an ODEBlock"""
        super().__init__()
        self.gnn = gnn
        
        # Number of function calls
        self.nfe = 0
        
        # Whether or not to augment input tensor x
        self.augment = augment
        
        # Dimensions of 0s to augment x with (as well as the time vector t)
        self.augment_size = augment_size
    
    def set_graph_data(self, edge_index, num_nodes=None):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
            
    def forward(self, t, x):
        self.nfe += 1
        
        x = augment_with_time(x, t, self.augment_size, self.augment)  
        
        # NOTE: technically dxdt
        if hasattr(self, 'edge_index'):
            x = self.gnn(x, self.edge_index)
        else:
            x = self.gnn(x)
        return x

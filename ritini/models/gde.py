__all__ = ['GDEFunc', 'ControlledGDEFunc', 'PyGGDEFunc']

import torch_geometric
from torch_geometric.data import Data
import torch, torch.nn as nn
from .gcn import GCNLayer
from typing import Callable

def augment_with_time(
    x:torch.Tensor, 
    t:int, size:int=1, 
    augment:bool=True
) -> torch.Tensor:  
    '''
    Augment feature matrix x with zeros and time.

    Parameters
    ----------
    x
        The input features to augment.
    
    t
        Time to append to x.

    size
        Number of columns of zeros to add to x.

    augment
        Whether or not to augment x with zeroes and time. If `False` returns x unchanged
    
    Returns
    -------
    augmented
        The augmented tensor (x, t, zeros...).
    '''
    # Internally handle if / else statement
    if not augment:
        return x
    
    # Ensure t is wrapped as torch Tensor
    t = torch_t(t, device=x.device)
    
    # Augment with size number of 0s
    zeros = torch.zeros(x.size(dim=0), size).to(x.device)
    
    # Time is only concatenated once
    times = t.repeat(x.size(dim=0), 1)
        
    augmented = torch.cat((x, times, zeros), dim=1)
    return augmented


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

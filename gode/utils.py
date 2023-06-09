# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_utils.ipynb.

# %% auto 0
__all__ = ['PerformanceContainer', 'accuracy', 'MAPELoss', 'MAELoss', 'set_timestamps', 'get_indices', 'get_device',
           'is_list_like', 'torch_t', 'get_torchdiffeq_solver', 'to_np', 'make_imap', 'dict_diff', 'dict_if_in',
           'dict_imap_if_in', 'reverse_lookup', 'can_imap', 'invert_imap_lookup', 'all_imappable', 'generate_steps',
           'make_time_lambdas', 'aggregate_loss_over_time', 'dearray']

# %% ../nbs/01_utils.ipynb 4
import torch
import torch.nn as nn, torch.nn.functional as F

class PerformanceContainer(object):
    """ Simple data class for metrics logging."""
    def __init__(self, data:dict):
        self.data = data
        
    @staticmethod
    def deep_update(x, y):
        for key in y.keys():
            x.update({key: list(x[key] + y[key])})
        return x
    
def accuracy(y_hat:torch.Tensor, y:torch.Tensor):
    """ Standard percentage accuracy computation """
    preds = torch.max(y_hat, 1)[1]
    return torch.mean((y == preds).float())


class MAPELoss(nn.Module):
    
    def forward(self, estimation:torch.Tensor, target:torch.Tensor):
        AER = torch.abs((target - estimation) / (target + 1e-10))  # Absolute error ratio
        MAPE = AER.mean() * 100
        return MAPE

class MAELoss(nn.Module):
    
    def forward(self, estimation:torch.Tensor, target:torch.Tensor):
        AE = torch.abs(target - estimation)
        MAE = AE.mean()
        return MAE

# %% ../nbs/01_utils.ipynb 5
def set_timestamps(timestamps):
    return [timestamps[i].unique() for i in range(len(timestamps))]


def get_indices(timestamps, set_ts, bs):
    all_idx = []
    for i in range(len(set_ts)):
        idx = []
        for j in range(bs):
            idx.append((set_ts[i] == timestamps[i][j]).nonzero().item())
        all_idx.append(idx)
    return all_idx

# %% ../nbs/01_utils.ipynb 7
import itertools, math, numpy as np, pandas as pd
import dgl, dgl.function as fn

import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torchdiffeq

from typing import Callable, Union, Literal

def get_device() -> Literal['cuda', 'cpu']:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

# %% ../nbs/01_utils.ipynb 8
def is_list_like(obj) -> bool:
    '''
    Tests if obj is like a list

    Parameters
    ----------
    obj
        A python object.
    
    Returns
    -------
    result
        Whether or not `obj` is a list, numpy array, pandas series or pytorch tensor.
    '''
    list_like = (
        np.ndarray, pd.Series, torch.Tensor,
    )
    list_types = tuple(map(str, map(type, list_like)))
    if isinstance(obj, list_like) or isinstance(obj, list):
        return True
    return False




# %% ../nbs/01_utils.ipynb 9
def torch_t(
    t:Union[int, float, list, torch.Tensor], 
    device:Union[str, torch.device]=None, 
    append_zero:bool=False
) -> torch.Tensor:
    '''
    Creates the time tensor used with Neural ODEs.

    Parameters
    ----------
    t
        The time (index or list) to convert.
    
    device
        Device on which to put the tensor.

    append_zero
        Whether or not a 0 should come before `t` e.g. `tensor([0, 1])` if `t=1`.
    
    Returns
    -------
    time_tensor
        Given `t` as a tensor.
    '''
    # Make sure t is a torch Tensor
    if not is_list_like(t):
        # Just a single value, integrate [0, t]
        if append_zero:
            t = torch.tensor([0, t]).float()
        # Just a single value, integrate [t]
        else:
            t = torch.tensor(t)

    elif torch.is_tensor(t):
        pass
    # is a list, but not already a torch tensor
    else:
        if append_zero:
            t = torch.tensor([0, *t]).float()
        else:
            t = torch.tensor(t)

    # Put t on correct device    
    if device is not None:
        t = t.to(device)
    return t

# %% ../nbs/01_utils.ipynb 10
def get_torchdiffeq_solver(adjoint:bool)->Callable:
    '''
    Gets corresponding ode integration function depending on boolean flag

    Parameters
    ----------
    adjoint
        Whether to use adjoint or not.
    
    Returns
    -------
    solver
        Either torchdiffeq.odeint_adjoint or torchdiffeq.odeint
    '''
    func_str = 'odeint_adjoint' if adjoint else 'odeint'
    return getattr(torchdiffeq, func_str)

# %% ../nbs/01_utils.ipynb 11
def to_np(tensor):
    return tensor.detach().cpu().numpy()

# %% ../nbs/01_utils.ipynb 12
def make_imap(dictionary:dict) -> dict:
    return {v:k for k, v in dictionary.items()}

def dict_diff(dict_a:dict, dict_b:dict) -> dict:
    return {
        ka: va for i, (ka, va) in enumerate(dict_a.items())
        if ka not in dict_b
    }

def dict_if_in(dict_a:dict, dict_b:dict) -> dict:
    return {
        ka: va for i, (ka, va) in enumerate(dict_a.items())
        if ka in dict_b
    }

def dict_imap_if_in(dict_a:dict, dict_b:dict) -> dict:
    '''
    NOTE: equivalent to make_imap(dict_if_in(a, b))
    '''
    return {
        va: ka for ia, (ka, va) in enumerate(dict_a.items())
        if ka in dict_b
    }


def reverse_lookup(dictionary, value):
    imap = make_imap(dictionary)
    if value in imap:
        return imap[value]
    return None

def can_imap(imap, key, omap):
    return imap[key] in omap

def invert_imap_lookup(imap, key, omap):
    if can_imap(imap, key, omap):
        return omap[imap[key]]
    return None

def all_imappable(arr, imap, omap):
    return all([can_imap(imap, e, omap) for e in arr])

# %% ../nbs/01_utils.ipynb 13
def generate_steps(groups):
    return list(zip(groups[:-1], groups[1:]))

# %% ../nbs/01_utils.ipynb 14
def make_time_lambdas(time_bins):
    lambdas = np.array([
        (i + 1) / len(time_bins)
        for i in time_bins
    ])
    lambdas /= lambdas.sum()
    return lambdas

# %% ../nbs/01_utils.ipynb 15
def aggregate_loss_over_time(
    data_ti, 
    data_tp, 
    criterion:Callable,
    aggregation:Literal['mean', 'sum']='mean',
    lambdas:Union[np.ndarray, list]=None
):
    n_timepoints = data_tp.size(0)
    if lambdas is None:
        lambdas = np.ones_like(np.arange(n_timepoints))
    
    losses = sum([
        lambdas[i] * criterion(data_tp[i], data_ti[i]) 
        for i in range(1, n_timepoints)
    ])

    if aggregation == 'mean':
        losses /= n_timepoints
        
    return losses

# %% ../nbs/01_utils.ipynb 16
def dearray(arr):
    '''
    Parameters
    ----------
    arr
        the array like object (tensor, numpy array, list of elements) to cast itself 
            and all of its list-like elements back to a python list
    
    Returns
    -------
    arr
        the array as a python list with all of its list-like elements also as python lists    
    '''
    if is_list_like(arr):
        if torch.is_tensor(arr):
            arr = to_np(arr)#.detach().cpu().numpy()

        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        
        arr = [dearray(el) for el in arr]
    return arr

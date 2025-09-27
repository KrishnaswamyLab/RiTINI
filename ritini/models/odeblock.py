
__all__ = ['ODEBlock']

import torch
import torch.nn as nn, torchdiffeq

from ..utils.utils import torch_t

class ODEBlock(nn.Module):
    def __init__(
        self, 
        func:nn.Module, 
        method:str='dopri5', 
        rtol:float=1e-3, 
        atol:float=1e-4, 
        adjoint:bool=True,
    ):
        """ Standard ODEBlock class. Can handle all types of ODE functions
            :method:str = {'euler', 'rk4', 'dopri5', 'adams'}
        """
        super().__init__()
        self.func = func
        self.method = method
        self.adjoint = adjoint
        self.ode_function = 'odeint_adjoint' if adjoint else 'odeint'
        self.atol = atol
        self.rtol = rtol

    def forward(
        self, 
        x:torch.Tensor,
        edge_index: torch.Tensor, 
        t:torch.Tensor, 
        return_whole_sequence:bool=False
    ):
        self.func.edge_index = edge_index
        t = torch_t(t)
        t = t.to(x.device).type_as(x)
  
        
        solver = getattr(torchdiffeq, self.ode_function) 

        out = solver(
            self.func, x, t,
            rtol=self.rtol, atol=self.atol, method=self.method
        ) 
        
        if not return_whole_sequence:
            out = out[-1]
        
        return out
    
    def forward_batched(self, x:torch.Tensor, nn:int, indices:list, timestamps:set):
        """ Modified forward for ODE batches with different integration times """
        t = torch.Tensor(list(timestamps))
        out = self.forward(x, t, return_whole_sequence=True)        
        out = self._build_batch(out, nn, indices).reshape(x.shape)
        return out
    
    def _build_batch(self, odeout, nn, indices):
        b_out = []
        for i in range(len(indices)):
            b_out.append(odeout[indices[i], i*nn:(i+1)*nn])
        return torch.cat(b_out).to(odeout.device)              
        
    def trajectory(self, x:torch.Tensor, t_end:int, num_points:int):
        t = torch.linspace(0, t_end, num_points).type_as(x).to(x.device)
        out = self.forward(x, t, return_whole_sequence=True, adjoint=False)
        return out

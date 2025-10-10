import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

class ODEBlock(nn.Module):
    """
    Neural ODE Block that integrates the GDE over time.
    """
    def __init__(self, func, method='rk4', atol=1e-3, rtol=1e-4, adjoint=False):
        """
        Args:
            func: GDEFunc instance defining the dynamics
            method: ODE solver method ('euler', 'rk4', 'dopri5', etc.)
            atol: Absolute tolerance for adaptive solvers
            rtol: Relative tolerance for adaptive solvers
            adjoint: Whether to use adjoint method for backprop (memory efficient)
        """
        super(ODEBlock, self).__init__()
        self.func = func
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.adjoint = adjoint
        
        # Integration time span: from t=0 to t=1
        self.integration_time = torch.tensor([0.0, 1.0])
    
        # Fixed-grid solvers don't use atol/rtol
        self.fixed_grid_solvers = ['euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams']
        
        if method in self.fixed_grid_solvers:
            self.options = {}
        else:
            self.options = {'atol': atol, 'rtol': rtol}

    def forward(self, x, t=None):
        """
        Args:
            x: Initial state, shape (num_nodes, features)
            t: Integration time points. If None, uses [0, 1]
        Returns:
            Final state at t=1
        """
        if t is None:
            t = torch.tensor([0, 1], dtype=x.dtype, device=x.device)
        
        
        integrator = odeint_adjoint if self.adjoint else odeint #Choose the type of odeint
        
        out = integrator(
            self.func,
            x,
            t,
            method=self.method,
            **self.options          #options here is either {atol, rtol} or None
        )
        
        return out[-1]  # Return final time point
    
    def __repr__(self):
        return (f'ODEBlock(method={self.method}, atol={self.atol}, '
                f'rtol={self.rtol}, adjoint={self.adjoint})')
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
    
        # Fixed-grid solvers don't use atol/rtol
        self.fixed_grid_solvers = ['euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams']
        if method in self.fixed_grid_solvers:
            self.options = {}
        else:
            self.options = {'atol': atol, 'rtol': rtol}
    
    def forward(self, x, t_eval):
        """
        Args:
            x: Initial state, shape (num_nodes, features)
            t_eval: Integration time points, shape (num_steps,)
        Returns:
            Trajectory at all time points, shape (num_steps, num_nodes, features)
        """
        integrator = odeint_adjoint if self.adjoint else odeint
        out = integrator(
            self.func,
            x,
            t_eval,
            method=self.method,
            **self.options
        )
        return out  # Return full trajectory
    
    def __repr__(self):
        return (f'ODEBlock(method={self.method}, atol={self.atol}, '
                f'rtol={self.rtol}, adjoint={self.adjoint})')
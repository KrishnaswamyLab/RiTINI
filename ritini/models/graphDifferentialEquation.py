import torch
import torch.nn as nn

class GDEFunc(nn.Module):
    """
    Graph Differential Equation for latent node dynamics:
        dz/dt = MLP( GNN(z, edge_index) )
    """
    def __init__(self, gnn, latent_dim):
        """
        gnn: a GNN module mapping (N, latent_dim) -> (N, latent_dim)
        latent_dim: dimension of latent state per node
        """
        super().__init__()
        self.gnn = gnn
        self.latent_dim = latent_dim
        
        # small nonlinear MLP to give curvature to the vector field
        # this is f_theta in the paper
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.Tanh(),
            nn.Linear(4 * latent_dim, 4 * latent_dim),
            nn.Tanh(),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        
        # number of function evaluations (NFE)
        self.nfe = 0
        
        # will be set by RiTINI before each ODE solve
        self.edge_index = None
        
        # last attention info (optional)
        self.attention_output = None
    
    def set_graph(self, edge_index):
        """Set edge_index externally before ODE integration."""
        self.edge_index = edge_index
    
    def forward(self, t, z):
        """
        t: scalar time (ignored — autonomous ODE)
        z: (N, latent_dim)
        returns dz/dt: (N, latent_dim)
        """
        self.nfe += 1
        
        # GNN in latent space
        out = self.gnn(z, self.edge_index)
        
        # handle GAT returning (output, (edge_index, attn))
        if isinstance(out, tuple):
            out, self.attention_output = out
        
        # nonlinear mapping to get dz/dt
        dzdt = self.mlp(out)
        
        return dzdt
import torch
import torch.nn as nn

from ritini.models.gatConvwithAttention import GATConvWithAttention
from ritini.models.graphDifferentialEquation import GDEFunc
from ritini.models.ode import ODEBlock
from ritini.models.time_attention import TimeAttention

class RiTINI(nn.Module):
    def __init__(self, in_features, out_features, latent_dim=16, history_length=5,
                 n_heads=1, feat_dropout=0.0, attn_dropout=0.0, negative_slope=0.2,
                 activation=nn.Tanh(), residual=False, bias=True,
                 ode_method='rk4', atol=1e-3, rtol=1e-4, adjoint=False,
                 device='cpu'):
        super(RiTINI, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.latent_dim = latent_dim
        self.history_length = history_length
        self.device = device
        
        # History encoder: LSTM to process past trajectory per node
        self.history_encoder = nn.LSTM(
            input_size=in_features,
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=True
        ).to(device)
        
        # Readout: latent ODE state -> output features
        self.readout = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.Tanh(),
            nn.Linear(16, out_features),
        ).to(device)
        
        # Build the graph ode model
        self.graph_ode = self._build_model(
            latent_dim, n_heads, feat_dropout, attn_dropout, negative_slope, 
            activation, residual, bias, ode_method, atol, rtol, adjoint
        )
    
    def _build_model(self, latent_dim, n_heads, feat_dropout, attn_dropout, 
                     negative_slope, activation, residual, bias, 
                     ode_method, atol, rtol, adjoint):
        """Build the Graph Neural ODE model."""
        
        # Create GAT layer (operates in latent space)
        gat_layer = GATConvWithAttention(
            in_features=latent_dim,
            out_features=latent_dim,
            n_heads=n_heads,
            negative_slope=negative_slope,
            residual=residual,
            activation=activation,
            feat_dropout=feat_dropout,
            attn_dropout=attn_dropout,
            bias=bias,
            concat=False
        )

        # Wrap in GDE function
        gdefunc = GDEFunc(
            gnn=gat_layer,
            mlp_latent_dim=[latent_dim, 4*latent_dim]
        )
        
        # Create ODE block
        graph_ode = ODEBlock(
            func=gdefunc,
            method=ode_method,
            atol=atol,
            rtol=rtol,
            adjoint=adjoint
        ).to(self.device)
        
        return graph_ode
    
    def forward(self, x_history, edge_index, t_eval):
        """
        Args:
            x_history: Historical trajectory, shape (num_nodes, history_length, in_features)
            edge_index: Graph connectivity
            t_eval: Time points to evaluate, shape (num_steps,)
        Returns:
            x_traj: Trajectory of predictions, shape (num_steps, num_nodes, out_features)
            attention_output: Attention weights from GAT
        """
        x_history = x_history.to(self.device)
        edge_index = edge_index.to(self.device)
        t_eval = t_eval.to(self.device)
        
        # Set graph structure
        self.graph_ode.func.set_graph(edge_index)
        
        # Encode history with LSTM per node
        # x_history: (N, H, 1) where N=nodes, H=history_length
        _, (h_n, _) = self.history_encoder(x_history)  # h_n: (1, N, latent_dim)
        z0 = h_n.squeeze(0)  # (N, latent_dim) - initial latent state from history
        
        # Integrate ODE in latent space
        z_traj = self.graph_ode(z0, t_eval)  # (T, N, latent_dim)
        
        # Decode to output space
        x_traj = self.readout(z_traj)  # (T, N, out_features)
        
        return x_traj, self.graph_ode.func.attention_output

    def get_nfe(self):
        """Get number of function evaluations (computational cost indicator)."""
        return self.graph_ode.func.nfe
    
    def reset_nfe(self):
        """Reset function evaluation counter."""
        self.graph_ode.func.nfe = 0


# Example usage
if __name__ == "__main__":
    # Create a simple graph
    num_nodes = 10
    in_features = 16
    out_features = 16
    
    # Edge index for a simple chain graph: 0-1-2-3-...-9
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ], dtype=torch.long)
    # Make edges bidirectional
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Random node features
    x = torch.randn(num_nodes, in_features)
    
    # Create model
    model = RiTINI(
        in_features=in_features,
        out_features=out_features,
        n_heads=2,
        feat_dropout=0.1,
        attn_dropout=0.1,
        residual=False,
        ode_method='rk4',
        atol=1e-3,
        rtol=1e-4,
        adjoint=False,
        device='cpu'
    )
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of function evaluations: {model.get_nfe()}")
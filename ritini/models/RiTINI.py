import torch
import torch.nn as nn

from ritini.models.gatConvwithAttention import GATConvWithAttention
from ritini.models.meanAttentionLayer import MeanAttentionLayer
from ritini.models.graphDifferentialEquation import GDEFunc
from ritini.models.ode import ODEBlock
from ritini.models.time_attention import TimeAttention

class RiTINI(nn.Module):
    """
    RiTINI Graph Neural ODE model using PyTorch Geometric.
    """

    def __init__(self, in_features, out_features, n_heads=1, 
                 feat_dropout=0.0, attn_dropout=0.0, negative_slope = 0.2,
                 activation=nn.Tanh(), residual=False, bias = True,
                 ode_method='rk4', atol=1e-3, rtol=1e-4, adjoint=False,
                 device='cpu'):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension (per head)
            edge_index: Graph connectivity in COO format (2, num_edges)
            n_heads: Number of attention heads
            feat_dropout: Feature dropout rate
            attn_dropout: Attention dropout rate
            residual: Whether to use residual connections
            ode_method: ODE solver method
            atol: Absolute tolerance
            rtol: Relative tolerance
            adjoint: Use adjoint method for backprop
            device: Device to run on
        """
        super(RiTINI, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features

        self.device = device
        # TimeAttention module (implements l_delta across a time window)
        # The module will be used in forward() if the input x has a time dimension
        self.time_attention = TimeAttention(in_features)

        # Build the graph ode model
        self.graph_ode = self._build_model(
            in_features, out_features, n_heads, feat_dropout, attn_dropout,negative_slope, activation, residual,bias,
            ode_method, atol, rtol, adjoint
        )
    
    def _build_model(self, in_features, out_features, n_heads, feat_dropout, 
                     attn_dropout, negative_slope, activation, residual, bias, ode_method, atol, rtol, adjoint,augment=False, augment_size=2):
        """Build the Graph Neural ODE model."""
        
        # Create GAT layer
        gat_layer = GATConvWithAttention(
            in_features=in_features,
            out_features=out_features,
            n_heads=n_heads,
            negative_slope=negative_slope,
            residual=residual,
            activation=activation,
            feat_dropout=feat_dropout,
            attn_dropout=attn_dropout,
            bias=bias,
            concat=False # This will average the attention heads
        )

        # Build a small MLP (f_theta) that maps GAT outputs to the dynamics dx/dt
        # This creates the required MLP between the GAT and the Neural ODE
        mlp = nn.Sequential(
            nn.Linear(out_features, out_features),
            activation,
            nn.Dropout(feat_dropout) if feat_dropout > 0 else nn.Identity(),
            nn.Linear(out_features, out_features)
        )

        # Wrap in GDE function with both layers (GAT -> MLP produces dx/dt)
        gdefunc = GDEFunc(
            gnn=gat_layer,
            augment=augment,
            augment_size=augment_size,
            mlp=mlp
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
        
    
    def forward(self, x,edge_index):
        """
        TODO: Correct the time window pass. It is not clear what is the shape of X here.
        Args:
            x: Node features, shape (num_nodes, in_features)
        Returns:
            out: Node features after ODE integration, shape (num_nodes, out_features * n_heads)
        """
        # If input has a time window, apply time attention l_delta to aggregate
        if x is None:
            raise ValueError("Input x must be provided")

        # x here is (T, N)
        input_time = x.unsqueeze(-1)
        

        # Spatial attention (GAT) should be applied per time-slice first, then we apply time-attention
        T, N, F = input_time.shape
        # Ensure graph is known for the gnn calls
        self.graph_ode.func.set_graph(edge_index)

        spatial_outputs = []
        last_spatial_att = None
        gnn = self.graph_ode.func.gnn
        for t in range(T):
            x_t = input_time[t].to(self.device)
            feat_t, att = gnn(x_t, edge_index)
            spatial_outputs.append(feat_t)
            last_spatial_att = att
        # Stack across time (num_timepoints, num_nodes, features)
        stacked = torch.stack(spatial_outputs, dim=0)
        # Now apply time attention l_delta over the spatially-aggregated time series
        x0, time_attn_weights = self.time_attention(stacked)
        # spatial attentions: use the last computed spatial attention as a proxy
        spatial_attn = last_spatial_att
        # Pass graph structure to the GDE function
        # prefer set_graph API if present
        self.graph_ode.func.set_graph(edge_index)

        # Forward through ODE block (starting from aggregated initial state)
        out = self.graph_ode(x0)

        return out, {'time_attention': time_attn_weights, 'spatial_attention': spatial_attn}

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
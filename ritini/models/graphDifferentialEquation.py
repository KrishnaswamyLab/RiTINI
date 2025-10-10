from torch import nn

class GDEFunc(nn.Module):
    def __init__(
        self, 
        gnn: nn.Module,  # Can be any GNN (GAT, GCN, etc.)
        augment: bool = False, 
        augment_size: int = 2
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
        
        # Store edge_index for PyTorch Geometric
        self.edge_index = None
        
        # Optional: store attention weights if using GAT
        self.attention_output = None
    
    def set_graph(self, edge_index):
        """Set the graph structure (edge_index for PyTorch Geometric)"""
        self.edge_index = edge_index
            
    def forward(self, t, x):
        self.nfe += 1
        
        # Pass through GNN with edge_index
        out = self.gnn(x, self.edge_index)
        
        # Handle attention outputs if GNN returns tuple (e.g., GAT)
        if isinstance(out, tuple):
            out, self.attention_output = out
        
        return out
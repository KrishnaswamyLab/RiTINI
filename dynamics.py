from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

from tqdm import tqdm
from ritini.models.gatConvwithAttention import GATConvWithAttention
from ritini.data.temporal_graph import TemporalGraphDataset
from ritini.models.RiTINI import RiTINI
from ritini.train import train_epoch
from ritini.utils.attention_graphs import adjacency_to_edge_index

def diagnose_model(model, dataloader, device, prior_adjacency, time_window=5):
    """Check if model is learning at each stage"""
    from ritini.utils.attention_graphs import adjacency_to_edge_index
    
    model.eval()
    
    batch = next(iter(dataloader))
    history = batch['history'].to(device)  # (batch, history_length, n_genes)
    node_features = batch['node_features'].to(device)  # (batch, time_window, n_genes)
    
    # Get first sample
    x_history = history[0].T.unsqueeze(-1)  # (n_genes, history_length, 1)
    edge_index = adjacency_to_edge_index(prior_adjacency).to(device)
    dt = 0.1
    t_eval = torch.arange(1, time_window, device=device) * dt
    
    with torch.no_grad():
        # Stage 1: History Encoder (LSTM)
        _, (h_n, _) = model.history_encoder(x_history)
        z0 = h_n.squeeze(0)
        print(f"\n=== Diagnostics ===")
        print(f"Input history range: [{x_history.min():.3f}, {x_history.max():.3f}]")
        print(f"LSTM output z0 range: [{z0.min():.3f}, {z0.max():.3f}]")
        print(f"LSTM output z0 mean/std: {z0.mean():.3f} / {z0.std():.3f}")
        
        # Stage 2: ODE
        model.graph_ode.func.set_graph(edge_index)
        z_traj = model.graph_ode(z0, t_eval)
        print(f"ODE output range: [{z_traj.min():.3f}, {z_traj.max():.3f}]")
        print(f"ODE output std: {z_traj.std():.3f}")
        
        # Check if ODE changed anything
        z_change = (z_traj[-1] - z_traj[0]).abs().mean()
        print(f"ODE trajectory change: {z_change:.6f}")
        
        # Stage 3: Readout
        pred = model.readout(z_traj)
        print(f"Prediction range: [{pred.min():.3f}, {pred.max():.3f}]")
        
        # Compare to target
        target = node_features[0, 1:]
        error = (pred.squeeze(-1) - target).abs().mean()
        print(f"Mean prediction error: {error:.3f}")
    
    # Check gradients
    model.train()
    x_history = history[0].T.unsqueeze(-1)
    pred_traj, _ = model(x_history, edge_index, t_eval)
    loss = torch.nn.MSELoss()(pred_traj.squeeze(-1), node_features[0, 1:])
    loss.backward()
    
    print(f"\n=== Gradient Magnitudes ===")
    lstm_grad = model.history_encoder.weight_ih_l0.grad.abs().mean() if model.history_encoder.weight_ih_l0.grad is not None else 0
    gnn_grad = model.graph_ode.func.gnn.lin.weight.grad.abs().mean() if model.graph_ode.func.gnn.lin.weight.grad is not None else 0
    mlp_grad = model.graph_ode.func.mlp[0].weight.grad.abs().mean() if model.graph_ode.func.mlp[0].weight.grad is not None else 0
    readout_grad = model.readout[0].weight.grad.abs().mean() if model.readout[0].weight.grad is not None else 0
    
    print(f"LSTM encoder grad: {lstm_grad:.6f}")
    print(f"GNN grad: {gnn_grad:.6f}")
    print(f"MLP grad: {mlp_grad:.6f}")
    print(f"Readout grad: {readout_grad:.6f}")
    
    model.zero_grad()
    model.train()

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 16
    time_window = 5  # Length of time_window, set to None to use all timepoints

    # Training parameters
    n_epochs = 50
    learning_rate = 0.001
    latent_dim = 16
    n_heads = 1
    feat_dropout = 0.1
    attn_dropout = 0.1
    activation_func = nn.Tanh()
    residual = False
    negative_slope = 0.2
    history_length = 5  # number of past timepoints to use

    # Loss parameters
    graph_reg_weight = 0.0  # Currently unused for kuramoto

    # Scheduler configs
    lr_factor = 0.5
    lr_patience = 10

    # Load Kuramoto data
    print("Loading Kuramoto data...")
    signals = np.load('data/signals.npy')  # (n_timepoints, n_nodes)
    adjacency = np.load('data/adjacency.npy')  # (n_nodes, n_nodes)

    n_timepoints, n_genes = signals.shape
    prior_adjacency = torch.from_numpy(adjacency).float().to(device)
    train_node_features = torch.tensor(signals, dtype=torch.float32)

    # Normalize data
    mean = train_node_features.mean()
    std = train_node_features.std()
    train_node_features = (train_node_features - mean) / std

    print(f"  Data normalized: mean={mean:.3f}, std={std:.3f}")

    # Save normalization stats for viz
    np.save('data/norm_stats.npy', np.array([mean.item(), std.item()]))

    print(f"\nData loaded successfully:")
    print(f"  Number of nodes: {n_genes}")
    print(f"  Number of timepoints: {n_timepoints}")
    print(f"  Prior adjacency shape: {prior_adjacency.shape}")
    print(f"  Train node features shape: {train_node_features.shape}")

    # Create dataset and dataloader
    dataset = TemporalGraphDataset(
        node_features=train_node_features,
        time_window=time_window,
        history_length=history_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Graph density: {(prior_adjacency > 0).float().mean():.3f}")
    print(f"Graph edges: {(prior_adjacency > 0).sum()}")

    # Initialize the RiTINI model
    model = RiTINI(
        in_features=1,  # Each node has 1 feature (gene expression)
        out_features=1,  # Predict 1 feature per node
        latent_dim=latent_dim,
        history_length=history_length,
        n_heads=n_heads,
        feat_dropout = feat_dropout,
        attn_dropout=attn_dropout,
        negative_slope=negative_slope,
        residual=residual,
        activation=activation_func,
        bias=True,
        ode_method='rk4',
        device=device
    ).to(device)

    print(f"\nModel initialized:")
    print(f"  Input features: 1 (per node)")
    print(f"  Output features: 1 (per node)")
    print(f"  Number of nodes: {n_genes}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Feature Dropout: {attn_dropout}")
    print(f"  Attention Dropout: {attn_dropout}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Training configuration
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience)

    print(f"\nTraining configuration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: Adam")
    print(f"  Loss function: MSE")
    print(f"  Scheduler: ReduceLROnPlateau")

    # Training loop
    print(f"\nStarting training...")
    best_loss = float('inf')
    training_history = []

    for epoch in tqdm(range(n_epochs)):
        epoch_loss, epoch_feature_loss, epoch_graph_loss = train_epoch(
            model, dataloader, optimizer, criterion, device, n_genes, 
            prior_adjacency, graph_reg_weight
        )
        
        if epoch == 0:
            diagnose_model(model, dataloader, device, prior_adjacency, time_window)
        
        training_history.append({
            'loss': epoch_loss
        })
        
        scheduler.step(epoch_loss)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], "
                f"Loss: {epoch_loss:.6f}, "
                f"Best Loss: {best_loss:.6f}")

    print(f"\nTraining completed!")
    print(f"Best total loss: {best_loss:.6f}")

    # Save training history with all metrics
    with open('training_history.json', 'w') as f:
        json.dump({'history': training_history}, f, indent=2)

    print(f"Training history saved to training_history.json")
    print(f"Best model saved to best_model.pt")


if __name__ == "__main__":
    main()

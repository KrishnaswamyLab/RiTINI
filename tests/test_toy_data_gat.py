import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import json

from ritini.data_generation.toy_data import create_temporal_graph_data
from ritini.models.gat import TemporalGAT
from ritini.data.temporal_graph import TemporalGraphDataset
from ritini.train import train_epoch

def test_toy_data_gat():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data parameters
    n_timepoints = 20
    n_trajectories = 2
    n_genes = 10
    batch_size = 1
    time_window = 2  # Length of time_window, set to None to use all timepoints
    n_change = 5

    # Training parameters
    n_epochs = 1000
    learning_rate = 0.001
    n_heads = 1
    dropout = 0.1

    #Scheduler configs
    lr_factor = 0.5
    lr_patience = 10

    # Create temporal graph data with slow edge changes
    all_node_features, all_graphs = create_temporal_graph_data(
        n_timepoints=n_timepoints,
        n_trajectories=n_trajectories,
        n_genes=n_genes,
        edge_density=0.3,
        temporal_noise=0.1,
        n_change=n_change,
        save=True,
    )

    # Select a single trajectory type for training
    trajectory_idx = 0
    train_node_features = all_node_features[trajectory_idx]  # Shape: (n_timepoints, n_genes)
    train_graphs = all_graphs[trajectory_idx]  # List of n_timepoints graphs

    # Extract prior graph (initial graph at t=0)
    prior_graph = train_graphs[0]

    # Convert prior graph to adjacency matrix
    prior_adjacency = torch.zeros(n_genes, n_genes)
    for edge in prior_graph.edges():
        prior_adjacency[edge[0], edge[1]] = 1
        prior_adjacency[edge[1], edge[0]] = 1  # Symmetric

    print(f"\nPrior graph (t=0): {len(prior_graph.nodes())} nodes, {len(prior_graph.edges())} edges")
    print(f"Prior adjacency shape: {prior_adjacency.shape}\n")

    # Create dataset and dataloader
    dataset = TemporalGraphDataset(
        node_features=train_node_features,
        time_window=time_window
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # Initialize the GAT model
    model = TemporalGAT(
        in_features=1,  # Each node has 1 feature (gene expression)
        out_features=1,  # Predict 1 feature per node
        n_heads=n_heads,
        dropout=dropout,
        concat=False  # Average heads for output
    ).to(device)

    print(f"\nModel initialized:")
    print(f"  Input features: 1 (per node)")
    print(f"  Output features: 1 (per node)")
    print(f"  Number of nodes: {n_genes}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Dropout: {dropout}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

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

    for epoch in range(n_epochs):
        epoch_loss = train_epoch(model, dataloader, optimizer, criterion, device, n_genes, prior_adjacency)
        training_history.append(epoch_loss)

        # Update scheduler
        scheduler.step(epoch_loss)

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'toy_best_model.pt')

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.6f}, Best Loss: {best_loss:.6f}")

    print(f"\nTraining completed!")
    print(f"Best loss: {best_loss:.6f}")

    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump({'losses': training_history}, f)

    print(f"Training history saved to training_history.json")
    print(f"Best model saved to toy_best_model.pt")


if __name__ == "__main__":
    test_toy_data_gat()

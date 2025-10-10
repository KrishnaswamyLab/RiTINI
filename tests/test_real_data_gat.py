import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

import networkx as nx
from tqdm import tqdm
from ritini.data.trajectory_loader import prepare_trajectories_data
from ritini.models.gat import TemporalGAT
from ritini.data.temporal_graph import TemporalGraphDataset
from ritini.train import train_epoch

def test_real_data_gat():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data parameters
    trajectory_file = 'data/trajectories/traj_data.pkl' 
    prior_graph_file='data/trajectories/cancer_granger_prior_graph_nx_20.pkl'
    gene_names_file='data/trajectories/gene_names.txt'
    n_top_genes = 20  # Number of genes from prior graph to use
    batch_size = 4
    time_window = 5  # Length of time_window, set to None to use all timepoints

    # Training parameters
    n_epochs = 100
    learning_rate = 0.001
    n_heads = 1
    dropout = 0.1

    # Scheduler configs
    lr_factor = 0.5
    lr_patience = 10

    # Load real trajectory data
    print("Loading trajectory data...")

    data = prepare_trajectories_data(
        trajectory_file=trajectory_file,
        n_top_genes=n_top_genes,
        prior_graph_file=prior_graph_file,
        gene_names_file=gene_names_file,
        use_mean_trajectory=True
    )


    trajectories = data['trajectories']  # Shape: (n_timepoints, n_trajectories=1, n_genes)
    gene_names = data['gene_names']
    prior_adjacency = data['prior_adjacency']
    prior_graph = data['prior_graph']
    n_genes = data['n_genes']
    n_timepoints = data['n_timepoints']

    # Extract node features (remove trajectory dimension since we use mean)
    trajectory_idx = 0
    train_node_features = torch.tensor(trajectories[:, trajectory_idx, :], dtype=torch.float32)  # Shape: (n_timepoints, n_genes)

    print(f"\nData loaded successfully:")
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Number of genes: {n_genes}")
    print(f"  Number of timepoints: {n_timepoints}")
    print(f"  Prior graph: {len(prior_graph.nodes())} nodes, {len(prior_graph.edges())} edges")
    print(f"  Prior adjacency shape: {prior_adjacency.shape}\n")
    print(f"Train node features shape: {train_node_features.shape}")

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

    for epoch in tqdm(range(n_epochs)):
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
            }, 'best_model_real_gat.pt')

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.6f}, Best Loss: {best_loss:.6f}")

    print(f"\nTraining completed!")
    print(f"Best loss: {best_loss:.6f}")

    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump({'losses': training_history}, f)

    print(f"Training history saved to training_history.json")
    print(f"Best model saved to best_model_real_gat.pt")


if __name__ == "__main__":
    test_real_data_gat()

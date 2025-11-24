import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from ritini.data.trajectory_loader import prepare_trajectories_data
from ritini.data.temporal_graph import TemporalGraphDataset
from ritini.models.RiTINI import RiTINI
from ritini.train import train_epoch
from ritini.utils.preprocess import process_trajectory_data

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data parameters
    raw_trajectory_file = 'data/raw/traj_data.npy' 
    raw_gene_names_file='data/raw/gene_names.txt'
    interest_genes_file = 'data/raw/interest_genes.txt'

    # Data batching
    batch_size = 4
    time_window = 5  # Length of time_window, set to None to use all timepoints

    # Training parameters
    n_epochs = 200
    learning_rate = 0.001
    n_heads = 1
    feat_dropout = 0.1
    attn_dropout = 0.1
    activation_func = nn.Tanh()
    residual = False
    negative_slope = 0.2

    # Loss parameters
    graph_reg_weight = 0.1

    # Scheduler configs
    lr_factor = 0.5
    lr_patience = 10

    #Output directories
    output_dir = 'output'

    # Preprocess input data
    trajectory_file, prior_graph_adjacency_file, gene_names_file = process_trajectory_data(
        raw_trajectory_file,
        raw_gene_names_file,
        interest_genes_file)
    
    # Prepare input data
    data = prepare_trajectories_data(
        trajectory_file=trajectory_file,
        prior_graph_adjacency_file=prior_graph_adjacency_file,
        gene_names_file=gene_names_file)

    trajectories = data['trajectories']  # Shape: (n_timepoints, n_trajectories=1, n_genes)
    n_genes = data['n_genes']
    n_timepoints = data['n_timepoints']
    prior_adjacency = data['prior_adjacency']  # Shape: (n_genes, n_genes)

    #Convert prior adjacency to torch tensor
    prior_adjacency = torch.tensor(prior_adjacency, dtype=torch.float32).to(device)  # Shape: (n_genes, n_genes)

    # train_node_features is created shortly below; compute it now from data
    trajectory_idx = 0
    train_node_features = torch.tensor(trajectories[:, trajectory_idx, :], dtype=torch.float32)  # (n_timepoints, n_genes)

    # Normalize data
    mean = train_node_features.mean()
    std = train_node_features.std()
    train_node_features = (train_node_features - mean) / std

    print(f"\nData loaded successfully:")
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Number of genes: {n_genes}")
    print(f"  Number of timepoints: {n_timepoints}")
    print(f"  Prior adjacency shape: {prior_adjacency.shape}\n")
    print(f"Train node features shape: {train_node_features.shape}")
    print(f"  Data normalized: mean={mean:.3f}, std={std:.3f}")

    # Check prior graph properties
    print(f"  Graph density: {(prior_adjacency > 0).float().mean():.3f}")
    print(f"  Graph edges: {(prior_adjacency > 0).sum()}")

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

    # Initialize the RiTINI model
    model = RiTINI(
        in_features=1,  # Each node has 1 feature (gene expression)
        out_features=1,  # Predict 1 feature per node
        n_heads=n_heads,
        feat_dropout = feat_dropout,
        attn_dropout=attn_dropout,
        negative_slope=negative_slope,
        residual=residual,
        activation=activation_func,
        bias=True,
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
            model, dataloader, optimizer, criterion, device, n_genes, prior_adjacency,graph_reg_weight
        )
        
        # Store all loss components
        training_history.append({
            'total_loss': epoch_loss,
            'feature_loss': epoch_feature_loss,
            'graph_loss': epoch_graph_loss
        })

        # Update scheduler (use total loss)
        scheduler.step(epoch_loss)

        # Save best model (based on total loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'feature_loss': epoch_feature_loss,
                'graph_loss': epoch_graph_loss,
                'n_genes': n_genes,
                'mean': mean.item(),
                'std': std.item(),
            }, os.path.join(output_dir,'best_model.pt'))

        # Print progress with all loss components
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], "
                f"Total Loss: {epoch_loss:.6f}, "
                f"Feature Loss: {epoch_feature_loss:.6f}, "
                f"Graph Loss: {epoch_graph_loss:.6f}, "
                f"Best Loss: {best_loss:.6f}")

    print(f"\nTraining completed!")
    print(f"Best total loss: {best_loss:.6f}")

    # Save training history with all metrics
    with open(os.path.join(output_dir,'training_history.json'), 'w') as f:
        json.dump({'history': training_history}, f, indent=2)

    print(f"Training history saved to training_history.json")
    print(f"Best model saved to best_model.pt")


if __name__ == "__main__":
    main()

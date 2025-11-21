import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import numpy as np
import random

from tqdm import tqdm
from ritini.data.trajectory_loader import process_single_trajectory_data
from ritini.data.temporal_graph import TemporalGraphDataset
from ritini.models.RiTINI import RiTINI
from ritini.train import train_epoch

def main():
    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data parameters
    trajectory_file = 'data/trajectory_1_natalia/traj_data.npy'    
    gene_names_file = 'data/trajectory_1_natalia/gene_names.txt'    
    granger_p_val_file = 'data/cell_cycle_RG/granger_RGtoIPCtoNeuron_p.csv'    
    granger_coef_file = 'data/cell_cycle_RG/granger_RGtoIPCtoNeuron_c.csv'
    
    n_top_genes = 100  # Number of genes to use (subset from 3515)
    batch_size = 16
    time_window = 5
    history_length = 5

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
    ode_method = 'rk4'

    # Loss parameters
    graph_reg_weight = 0.0  # Disabled for now

    # Scheduler configs
    lr_factor = 0.5
    lr_patience = 10

    # Load gene expression trajectory data
    print("Loading gene expression trajectory data...")
    data = process_single_trajectory_data(
        trajectory_file=trajectory_file,
        granger_pval_file=granger_p_val_file,
        granger_coef_file=granger_coef_file,
        gene_names_file=gene_names_file
        #n_top_genes=n_top_genes
    )

    trajectories = data['trajectories']  # Shape: (n_timepoints, n_trajectories, n_genes)
    gene_names = data['gene_names']
    prior_adjacency = data['prior_adjacency'].to(device)
    n_genes = data['n_genes']
    n_timepoints = data['n_timepoints']

    # Extract node features (use mean trajectory)
    trajectory_idx = 0
    train_node_features = torch.tensor(trajectories[:, trajectory_idx, :], dtype=torch.float32)

    # Normalize data
    mean = train_node_features.mean()
    std = train_node_features.std()
    train_node_features = (train_node_features - mean) / std

    print(f"\nData loaded successfully:")
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Number of genes (selected): {n_genes}")
    print(f"  Number of timepoints: {n_timepoints}")
    print(f"  Prior adjacency shape: {prior_adjacency.shape}")
    print(f"  Train node features shape: {train_node_features.shape}")
    print(f"  Data normalized: mean={mean:.3f}, std={std:.3f}")
    
    # Check graph properties
    print(f"  Graph density: {(prior_adjacency > 0).float().mean():.3f}")
    print(f"  Graph edges: {(prior_adjacency > 0).sum()}")

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

    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # Initialize the RiTINI model
    model = RiTINI(
        in_features=1,
        out_features=1,
        latent_dim=latent_dim,
        history_length=history_length,
        n_heads=n_heads,
        feat_dropout=feat_dropout,
        attn_dropout=attn_dropout,
        negative_slope=negative_slope,
        residual=residual,
        activation=activation_func,
        bias=True,
        ode_method=ode_method,
        device=device
    ).to(device)

    print(f"\nModel initialized:")
    print(f"  Input features: 1 (per node)")
    print(f"  Output features: 1 (per node)")
    print(f"  Latent ODE dimension: {latent_dim}")
    print(f"  History length: {history_length}")
    print(f"  Number of nodes: {n_genes}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Feature Dropout: {feat_dropout}")
    print(f"  Attention Dropout: {attn_dropout}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Training configuration
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience
    )

    print(f"\nTraining configuration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: Adam")
    print(f"  Loss function: MSE")
    print(f"  Scheduler: ReduceLROnPlateau")
    print(f"  ODE method: {ode_method}")

    # Training loop
    print(f"\nStarting training...")
    best_loss = float('inf')
    training_history = []

    for epoch in tqdm(range(n_epochs)):
        epoch_loss, _, _ = train_epoch(
            model, dataloader, optimizer, criterion, device, n_genes, 
            prior_adjacency, graph_reg_weight
        )
        
        # Store loss
        training_history.append({
            'loss': epoch_loss
        })

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
                'n_genes': n_genes,
                'gene_names': gene_names,
                'mean': mean.item(),
                'std': std.item(),
            }, 'best_model_genes.pt')

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], "
                  f"Loss: {epoch_loss:.6f}, "
                  f"Best Loss: {best_loss:.6f}")

    print(f"\nTraining completed!")
    print(f"Best loss: {best_loss:.6f}")

    # Save training history
    with open('training_history_genes.json', 'w') as f:
        json.dump({'history': training_history}, f, indent=2)

    print(f"Training history saved to training_history_genes.json")
    print(f"Best model saved to best_model_genes.pt")


if __name__ == "__main__":
    main()
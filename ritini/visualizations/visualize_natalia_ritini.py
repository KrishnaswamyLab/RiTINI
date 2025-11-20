import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ritini.data.trajectory_loader import prepare_trajectories_data,process_single_trajectory_data
from ritini.models.RiTINI import RiTINI

def infer_graphs_at_timepoints(
    model,
    node_features,
    prior_adjacency,
    device,
    threshold=0.5
):
    """
    Infer graphs at each timepoint using the trained model.

    Args:
        model: Trained RiTINI model
        node_features: Node features (n_timepoints, n_genes)
        prior_adjacency: Prior adjacency matrix (n_genes, n_genes)
        device: Device to run inference on
        threshold: Attention weight threshold for edge inclusion

    Returns:
        list of attention weight matrices, one per timepoint
        predictions: predicted next timepoint values (n_timepoints-1, n_genes)
        targets: actual next timepoint values (n_timepoints-1, n_genes)
    """
    model.eval()

    # Convert prior adjacency to edge_index for first timepoint
    edge_index = torch.nonzero(prior_adjacency, as_tuple=False).t().to(device)
    n_genes = node_features.shape[1]

    with torch.no_grad():
        all_preds = []
        all_attention_matrices = []

        # Move all data to GPU at once
        x = node_features.unsqueeze(2).to(device)  # (T, n_genes, 1)

        for t in tqdm(range(node_features.shape[0])):
            # Forward pass
            pred, (edge_idx, attention_weights) = model(x[t], edge_index)
            
            # Store predictions
            if t < node_features.shape[0] - 1:
                all_preds.append(pred.squeeze())
            
            # Build adjacency matrix on GPU
            attention_matrix = torch.zeros(n_genes, n_genes, device=device)
            src, dst = edge_idx[0], edge_idx[1]
            attention_matrix[src, dst] = attention_weights.mean(dim=1) if attention_weights.dim() > 1 else attention_weights
            
            all_attention_matrices.append(attention_matrix)
            
            # Update edge_index for next timepoint based on attention weights
            # Option 1: Use edges above threshold
            if threshold is not None:
                mask = attention_weights.mean(dim=1) > threshold if attention_weights.dim() > 1 else attention_weights > threshold
                edge_index = edge_idx[:, mask]
            # Option 2: Use the returned edge_idx directly
            else:
                edge_index = edge_idx
        
        # Keep as tensors
        predictions = [p.cpu() for p in all_preds]
        targets = [node_features[t + 1].cpu() for t in range(node_features.shape[0] - 1)]
        attention_matrices = [a.cpu() for a in all_attention_matrices]

    return attention_matrices, predictions, targets

def visualize_gene_expression_trajectories(
    node_features,
    predictions,
    targets,
    gene_names=None,
    save_dir="visualizations",
    genes_to_show=None
):
    """
    Visualize expression evolution through time for each gene, comparing actual vs predicted.

    Args:
        node_features: Original node features (n_timepoints, n_genes)
        predictions: Predicted values (n_timepoints-1, n_genes)
        targets: Actual values (n_timepoints-1, n_genes)
        gene_names: Optional list of gene names
        save_dir: Directory to save visualizations
        genes_to_show: List of specific gene indices to visualize (None = all)
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    n_timepoints = node_features.shape[0]
    n_genes = node_features.shape[1]

    if genes_to_show is None:
        genes_to_show = range(n_genes)

    # Create individual plots for each gene
    for gene_idx in genes_to_show:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Actual expression values (all timepoints)
        actual_expression = node_features[:, gene_idx].numpy()
        timepoints = np.arange(n_timepoints)

        # Predicted values (starts from timepoint 1, predicting timepoint 2 onwards)
        # pred_expression = predictions[:, gene_idx].numpy()
        pred_expression = np.array(predictions)[:, gene_idx]
        pred_timepoints = np.arange(1, n_timepoints)

        # Plot actual expression
        ax.plot(timepoints, actual_expression, 'b-o', linewidth=2, 
                markersize=8, label='Actual Expression', alpha=0.7)

        # Plot predicted expression
        ax.plot(pred_timepoints, pred_expression, 'r--s', linewidth=2, 
                markersize=6, label='Predicted Expression', alpha=0.7)

        # Add vertical lines showing prediction errors
        for t in range(len(pred_expression)):
            actual_val = actual_expression[t + 1]
            pred_val = pred_expression[t]
            ax.plot([pred_timepoints[t], pred_timepoints[t]], 
                   [actual_val, pred_val], 
                   'gray', alpha=0.3, linewidth=1)

        # Calculate error metrics for this gene
        gene_errors = pred_expression - actual_expression[1:]
        gene_mse = np.mean(gene_errors ** 2)
        gene_mae = np.mean(np.abs(gene_errors))

        ax.set_xlabel('Timepoint', fontsize=12)
        ax.set_ylabel('Expression Level', fontsize=12)
        
        gene_name = gene_names[gene_idx] if gene_names is not None else f'Gene {gene_idx}'
        ax.set_title(f'{gene_name} Expression Over Time\nMSE: {gene_mse:.4f}, MAE: {gene_mae:.4f}', 
                    fontsize=14)
        
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / f'gene_{gene_idx}_trajectory.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(genes_to_show)} gene trajectory plots to {save_dir}/")

    #### PREDICTIONS HEATMAPS

    # Create heatmaps for target, predicted, and error values
    n_show = min(len(genes_to_show), len(genes_to_show))  # Show all genes or limit as needed
    selected_genes = list(genes_to_show)[:n_show]

    # Prepare data matrices
    n_genes = len(selected_genes)
    n_timepoints_pred = n_timepoints - 1

    # Matrix for actual expression (all timepoints)
    actual_matrix = np.zeros((n_genes, n_timepoints))
    # Matrix for predicted expression (excluding first timepoint)
    pred_matrix = np.zeros((n_genes, n_timepoints_pred))
    # Matrix for errors
    error_matrix = np.zeros((n_genes, n_timepoints_pred))

    # Gene names for y-axis
    gene_labels = []

    for idx, gene_idx in enumerate(selected_genes):
        # Actual expression for all timepoints
        actual_matrix[idx, :] = node_features[:, gene_idx].numpy()
        
        # Predicted expression (starts from timepoint 1)
        pred_matrix[idx, :] = np.array(predictions)[:, gene_idx]
        
        # Error (predicted - actual for corresponding timepoints)
        error_matrix[idx, :] = pred_matrix[idx, :] - actual_matrix[idx, 1:]
        
        # Gene label
        gene_name = gene_names[gene_idx] if gene_names is not None else f'Gene {gene_idx}'
        gene_labels.append(gene_name)

    # Create figure with three heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, n_genes * 0.3)))

    # Heatmap 1: Actual expression (all timepoints)
    im1 = axes[0].imshow(actual_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_title('Target Expression', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Timepoint', fontsize=12)
    axes[0].set_ylabel('Gene', fontsize=12)
    axes[0].set_yticks(range(n_genes))
    axes[0].set_yticklabels(gene_labels, fontsize=8)
    axes[0].set_xticks(range(n_timepoints))
    axes[0].set_xticklabels(range(n_timepoints))
    plt.colorbar(im1, ax=axes[0], label='Expression Level')

    # Heatmap 2: Predicted expression (timepoints 1 onwards)
    im2 = axes[1].imshow(pred_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1].set_title('Predicted Expression', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Timepoint', fontsize=12)
    axes[1].set_ylabel('Gene', fontsize=12)
    axes[1].set_yticks(range(n_genes))
    axes[1].set_yticklabels(gene_labels, fontsize=8)
    axes[1].set_xticks(range(n_timepoints_pred))
    axes[1].set_xticklabels(range(1, n_timepoints))
    plt.colorbar(im2, ax=axes[1], label='Expression Level')

    # Heatmap 3: Prediction errors
    im3 = axes[2].imshow(error_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                        vmin=-np.abs(error_matrix).max(), vmax=np.abs(error_matrix).max())
    axes[2].set_title('Prediction Error (Predicted - Target)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Timepoint', fontsize=12)
    axes[2].set_ylabel('Gene', fontsize=12)
    axes[2].set_yticks(range(n_genes))
    axes[2].set_yticklabels(gene_labels, fontsize=8)
    axes[2].set_xticks(range(n_timepoints_pred))
    axes[2].set_xticklabels(range(1, n_timepoints))
    plt.colorbar(im3, ax=axes[2], label='Error')

    plt.tight_layout()
    plt.savefig(save_path / 'gene_expression_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()

def visualize_attention_heatmaps(
    attention_matrices,
    gene_names=None,
    save_dir="visualizations",
    timepoints_to_show=None,
    vmin=None,
    vmax=None,
    cmap='viridis'
):
    """
    Visualize attention matrices as heatmaps over time.

    Args:
        attention_matrices: List of attention matrices (n_timepoints, n_genes, n_genes)
        gene_names: Optional list of gene names
        save_dir: Directory to save visualizations
        timepoints_to_show: List of specific timepoint indices to visualize (None = auto-select)
        vmin: Minimum value for colormap (None = auto)
        vmax: Maximum value for colormap (None = auto)
        cmap: Colormap to use
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    n_timepoints = len(attention_matrices)
    n_genes = attention_matrices[0].shape[0]

    # Auto-select timepoints to show if not specified
    if timepoints_to_show is None:
        n_show = min(9, n_timepoints)
        timepoints_to_show = np.linspace(0, n_timepoints-1, n_show, dtype=int)

    # Determine global min/max for consistent color scale across timepoints
    if vmin is None or vmax is None:
        all_values = torch.cat([am.flatten() for am in attention_matrices])
        if vmin is None:
            vmin = all_values.min().item()
        if vmax is None:
            vmax = all_values.max().item()

    # Create grid of heatmaps
    rows = int(np.ceil(np.sqrt(len(timepoints_to_show))))
    cols = int(np.ceil(len(timepoints_to_show) / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows))
    if len(timepoints_to_show) == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, t in enumerate(timepoints_to_show):
        if idx >= len(axes):
            break

        ax = axes[idx]
        
        # Convert tensor to numpy
        attn_matrix = attention_matrices[t].numpy()
        
        # Create heatmap
        im = ax.imshow(attn_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        ax.set_title(f'Timepoint {t}', fontsize=12)
        ax.set_xlabel('Target Gene', fontsize=10)
        ax.set_ylabel('Source Gene', fontsize=10)
        
        # Add gene names if available and not too many
        if gene_names is not None and n_genes <= 20:
            ax.set_xticks(range(n_genes))
            ax.set_yticks(range(n_genes))
            ax.set_xticklabels(gene_names, rotation=90, fontsize=8)
            ax.set_yticklabels(gene_names, fontsize=8)
        
        # Add colorbar for each subplot
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(len(timepoints_to_show), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path / 'attention_matrices_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved attention matrices heatmap to {save_dir}/attention_matrices_heatmaps.png")

    # Create individual high-resolution heatmaps for each selected timepoint
    for t in timepoints_to_show:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        attn_matrix = attention_matrices[t].numpy()
        
        im = ax.imshow(attn_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        ax.set_title(f'Attention Matrix at Timepoint {t}', fontsize=16)
        ax.set_xlabel('Target Gene', fontsize=14)
        ax.set_ylabel('Source Gene', fontsize=14)
        
        # Add gene names if available
        if gene_names is not None:
            ax.set_xticks(range(n_genes))
            ax.set_yticks(range(n_genes))
            ax.set_xticklabels(gene_names, rotation=90, fontsize=10)
            ax.set_yticklabels(gene_names, fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', fontsize=12)
        
        # Add grid for better readability
        ax.set_xticks(np.arange(n_genes) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_genes) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(save_path / f'attention_matrix_t{t}.png', dpi=200, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(timepoints_to_show)} individual attention matrix heatmaps")

    # Create an animation-style summary showing attention evolution
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Calculate average attention weights over time
    avg_attention_over_time = np.array([am.numpy().mean() for am in attention_matrices])
    max_attention_over_time = np.array([am.numpy().max() for am in attention_matrices])
    
    ax.plot(avg_attention_over_time, 'b-', linewidth=2, label='Average Attention')
    ax.plot(max_attention_over_time, 'r-', linewidth=2, label='Max Attention')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title('Attention Weights Evolution Over Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'attention_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved attention evolution plot to {save_dir}/attention_evolution.png")

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = "best_model.pt"
    save_dir = "visualizations_natalia_new"


    # Data parameters
    trajectory_file = 'data/data/traj_data.npy' 
    gene_names_file='data/data/gene_names.txt'

    granger_p_val_file = 'data/data/granger_RGtoIPCtoNeuron_p.csv'
    granger_coef_file = 'data/data/granger_RGtoIPCtoNeuron_c.csv'


    # Training parameters
    n_heads = 1
    feat_dropout = 0.1
    attn_dropout = 0.1
    activation_func = nn.Tanh()
    residual = False
    negative_slope = 0.2


    # Load real trajectory data
    print("Loading trajectory data...")

    data = process_single_trajectory_data(
        trajectory_file= trajectory_file,
        granger_pval_file= granger_p_val_file,
        granger_coef_file= granger_coef_file,
        gene_names_file= gene_names_file
    )


    trajectories = data['trajectories']
    gene_names = data['gene_names']

    prior_adjacency = data['prior_adjacency']
    n_genes = data['n_genes']

    # Extract node features
    node_features = torch.tensor(trajectories[:, 0, :], dtype=torch.float32)

    timepoints_to_show = [0, 24, 49, 74, 99]  # None = all timepoints, or specify list like [0, 2, 4, 6, 8]
    # genes_to_show =[0,5,10,20,30]
    genes_to_show = None
    attention_threshold = 0.01

    print(f"Data loaded:")
    print(f"  Node features shape: {node_features.shape}")
    print(f"  Number of genes: {n_genes}")
    print(f"  Gene names: {gene_names[:5]}..." if gene_names is not None else "  No gene names")

    # Load trained model
    # Training parameters

    n_heads = 1
    feat_dropout = 0.1
    attn_dropout = 0.1
    activation_func = nn.Tanh()
    residual = False
    negative_slope = 0.2

    print(f"\nLoading model from {model_path}...")
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
        ode_method='rk4', 
        atol=1e-3, 
        rtol=1e-4, 
        adjoint=False,
        device=device).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded (trained for {checkpoint['epoch']+1} epochs, loss: {checkpoint['loss']:.6f})")

    # Infer graphs at all timepoints
    print("\nInferring graphs at all timepoints...")
    attention_matrices, predictions, targets = infer_graphs_at_timepoints(
        model,
        node_features,
        prior_adjacency,
        device,
        threshold=attention_threshold
    )

    print(f"Inferred {len(attention_matrices)} attention_matrices")

    # Visualize predictions
    if predictions is not None and targets is not None:
        print("\nCreating prediction visualizations...")
        # Visualize gene expression trajectories
        print("\nCreating gene expression trajectory visualizations...")
        visualize_gene_expression_trajectories(
            node_features,
            predictions,
            targets,
            gene_names=gene_names,
            save_dir=save_dir,
            genes_to_show=genes_to_show  # Set to None for all genes, or specify list like [0, 1, 2, 5]
        )
    print("\nCreating attention matrix visualizations...")

    visualize_attention_heatmaps(
        attention_matrices,
        gene_names=gene_names,
        save_dir=save_dir,
        timepoints_to_show=timepoints_to_show,
        cmap='viridis'  # Try 'hot', 'plasma', 'coolwarm' for different color schemes
    )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

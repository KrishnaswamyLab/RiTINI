import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
from ritini.data.trajectory_loader import prepare_trajectories_data
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
        model: Trained TemporalGAT model
        node_features: Node features (n_timepoints, n_genes)
        prior_adjacency: Prior adjacency matrix (n_genes, n_genes)
        device: Device to run inference on
        threshold: Attention weight threshold for edge inclusion

    Returns:
        list of networkx graphs, one per timepoint
        list of attention weight matrices, one per timepoint
        predictions: predicted next timepoint values (n_timepoints-1, n_genes)
        targets: actual next timepoint values (n_timepoints-1, n_genes)
    """
    model.eval()
    graphs = []
    attention_matrices = []
    predictions = []
    targets = []

    # Convert prior adjacency to edge_index
    edge_index = torch.nonzero(prior_adjacency, as_tuple=False).t().to(device)
    n_genes = node_features.shape[1]

    with torch.no_grad():
        for t in range(node_features.shape[0]):
            # Get features at this timepoint
            x = node_features[t].unsqueeze(1).to(device)  # (n_genes, 1)

            # Forward pass to get predictions and attention weights
            pred, (edge_idx, attention_weights) = model(x, edge_index)

            # Store predictions and targets (if not last timepoint)
            if t < node_features.shape[0] - 1:
                predictions.append(pred.squeeze().cpu())
                targets.append(node_features[t + 1].cpu())

            # Build adjacency matrix from attention weights
            attention_matrix = torch.zeros(n_genes, n_genes)
            for i, (src, dst) in enumerate(edge_idx.t().cpu()):
                attention_matrix[src, dst] = attention_weights[i].mean().item()

            attention_matrices.append(attention_matrix.numpy())

            # Create networkx graph from thresholded attention weights
            G = nx.Graph()
            G.add_nodes_from(range(n_genes))

            for i, (src, dst) in enumerate(edge_idx.t().cpu()):
                weight = attention_weights[i].mean().item()
                if weight >= threshold:
                    G.add_edge(src.item(), dst.item(), weight=weight)

            graphs.append(G)

    predictions = torch.stack(predictions) if predictions else None
    targets = torch.stack(targets) if targets else None

    return graphs, attention_matrices, predictions, targets


def visualize_graphs_at_timepoints(
    graphs,
    attention_matrices,
    gene_names=None,
    save_dir="visualizations",
    timepoints_to_show=None
):
    """
    Visualize inferred graphs at multiple timepoints.

    Args:
        graphs: List of networkx graphs
        attention_matrices: List of attention weight matrices
        gene_names: Optional list of gene names for node labels
        save_dir: Directory to save visualizations
        timepoints_to_show: List of specific timepoint indices to visualize (None = all)
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    n_timepoints = len(graphs)

    if timepoints_to_show is None:
        timepoints_to_show = range(n_timepoints)

    # Visualize individual timepoint graphs
    for t in timepoints_to_show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot graph
        G = graphs[t]
        pos = nx.spring_layout(G, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=500, ax=ax1)

        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]

        if len(weights) > 0:
            nx.draw_networkx_edges(G, pos, width=weights,
                                  alpha=1, edge_color=weights,
                                  edge_cmap=plt.cm.Reds, ax=ax1)

        # Draw labels
        if gene_names is not None:
            labels = {i: gene_names[i] for i in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax1)
        else:
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)

        ax1.set_title(f'Inferred Graph at Timepoint {t}')
        ax1.axis('off')

        # Plot attention matrix heatmap
        im = ax2.imshow(attention_matrices[t], cmap='Reds', aspect='auto')
        ax2.set_title(f'Attention Weights at Timepoint {t}')
        ax2.set_xlabel('Gene Index')
        ax2.set_ylabel('Gene Index')
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        plt.savefig(save_path / f'graph_timepoint_{t:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(timepoints_to_show)} graph visualizations to {save_dir}/")

    # Create overview plot with multiple timepoints
    n_show = min(6, len(timepoints_to_show))
    selected_timepoints = np.linspace(0, n_timepoints-1, n_show, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, t in enumerate(selected_timepoints):
        if idx >= len(axes):
            break

        G = graphs[t]
        pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=300, ax=axes[idx])

        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]

        if len(weights) > 0:
            nx.draw_networkx_edges(G, pos, width=[w*2 for w in weights],
                                  alpha=0.6, edge_color=weights,
                                  edge_cmap=plt.cm.Reds, ax=axes[idx])

        if gene_names is not None:
            labels = {i: gene_names[i] for i in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=axes[idx])

        axes[idx].set_title(f'Timepoint {t}')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path / 'graphs_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved overview visualization to {save_dir}/graphs_overview.png")


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
        pred_expression = predictions[:, gene_idx].numpy()
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

    # Create overview plot with multiple genes
    n_show = min(9, len(genes_to_show))
    selected_genes = list(genes_to_show)[:n_show] if len(genes_to_show) <= n_show else \
                     [list(genes_to_show)[i] for i in np.linspace(0, len(genes_to_show)-1, n_show, dtype=int)]

    rows = int(np.ceil(np.sqrt(n_show)))
    cols = int(np.ceil(n_show / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_show == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, gene_idx in enumerate(selected_genes):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Actual expression
        actual_expression = node_features[:, gene_idx].numpy()
        timepoints = np.arange(n_timepoints)

        # Predicted expression
        pred_expression = predictions[:, gene_idx].numpy()
        pred_timepoints = np.arange(1, n_timepoints)

        ax.plot(timepoints, actual_expression, 'b-o', linewidth=2, 
                markersize=6, label='Actual', alpha=0.7)
        ax.plot(pred_timepoints, pred_expression, 'r--s', linewidth=2, 
                markersize=4, label='Predicted', alpha=0.7)

        # Calculate error metrics
        gene_errors = pred_expression - actual_expression[1:]
        gene_mse = np.mean(gene_errors ** 2)

        gene_name = gene_names[gene_idx] if gene_names is not None else f'Gene {gene_idx}'
        ax.set_title(f'{gene_name} (MSE: {gene_mse:.3f})', fontsize=10)
        ax.set_xlabel('Timepoint', fontsize=9)
        ax.set_ylabel('Expression', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(selected_genes), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path / 'gene_trajectories_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved gene trajectories overview to {save_dir}/gene_trajectories_overview.png")


def visualize_predictions(
    predictions,
    targets,
    gene_names=None,
    save_dir="visualizations",
    timepoints_to_show=None
):
    """
    Visualize predicted vs actual next timepoint values.

    Args:
        predictions: Predicted values (n_timepoints-1, n_genes)
        targets: Actual values (n_timepoints-1, n_genes)
        gene_names: Optional list of gene names
        save_dir: Directory to save visualizations
        timepoints_to_show: List of specific timepoint indices to visualize
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    n_timepoints = predictions.shape[0]
    n_genes = predictions.shape[1]

    # Compute errors
    errors = (predictions - targets).numpy()
    mse_per_timepoint = (errors ** 2).mean(axis=1)
    mae_per_timepoint = np.abs(errors).mean(axis=1)

    # Plot 1: Error metrics over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(mse_per_timepoint, 'b-', linewidth=2, label='MSE')
    ax1.set_xlabel('Timepoint')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Prediction MSE Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(mae_per_timepoint, 'r-', linewidth=2, label='MAE')
    ax2.set_xlabel('Timepoint')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Prediction MAE Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path / 'prediction_errors_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Predicted vs Actual scatter plots for selected timepoints
    if timepoints_to_show is None:
        n_show = min(6, n_timepoints)
        timepoints_to_show = np.linspace(0, n_timepoints-1, n_show, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, t in enumerate(timepoints_to_show):
        if idx >= len(axes):
            break

        ax = axes[idx]
        pred_t = predictions[t].numpy()
        target_t = targets[t].numpy()

        ax.scatter(target_t, pred_t, alpha=0.6, s=50)

        # Add diagonal line (perfect prediction)
        min_val = min(target_t.min(), pred_t.min())
        max_val = max(target_t.max(), pred_t.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

        ax.set_xlabel('Actual Expression')
        ax.set_ylabel('Predicted Expression')
        ax.set_title(f'Timepoint {t} (MSE: {mse_per_timepoint[t]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / 'predicted_vs_actual_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Per-gene error heatmap
    errors_array = errors.T  # (n_genes, n_timepoints)

    fig, ax = plt.subplots(figsize=(14, max(6, n_genes * 0.3)))
    im = ax.imshow(errors_array, cmap='RdBu_r', aspect='auto', vmin=-np.abs(errors).max(), vmax=np.abs(errors).max())

    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Gene')
    ax.set_title('Prediction Errors: Predicted - Actual (per gene, per timepoint)')

    if gene_names is not None and len(gene_names) <= 30:
        ax.set_yticks(range(n_genes))
        ax.set_yticklabels(gene_names, fontsize=8)

    plt.colorbar(im, ax=ax, label='Error')
    plt.tight_layout()
    plt.savefig(save_path / 'per_gene_errors_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f"\nPrediction Statistics:")
    print(f"  Overall MSE: {mse_per_timepoint.mean():.6f}")
    print(f"  Overall MAE: {mae_per_timepoint.mean():.6f}")
    print(f"  Best timepoint MSE: {mse_per_timepoint.min():.6f} at t={mse_per_timepoint.argmin()}")
    print(f"  Worst timepoint MSE: {mse_per_timepoint.max():.6f} at t={mse_per_timepoint.argmax()}")

    print(f"\nSaved prediction visualizations to {save_dir}/")


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data parameters (should match training)
    trajectory_file = 'data/trajectories/traj_data.pkl'
    prior_graph_file = 'data/trajectories/cancer_granger_prior_graph_nx_20.pkl'
    gene_names_file = 'data/trajectories/gene_names.txt'
    n_top_genes = 20
    model_path = 'best_model.pt'

    # Visualization parameters
    attention_threshold = 0.25
    save_dir = 'visualizations'
    timepoints_to_show = None  # None = visualize all, or specify list like [0, 5, 10, 15]

    # Load data
    print("Loading trajectory data...")
    data = prepare_trajectories_data(
        trajectory_file=trajectory_file,
        n_top_genes=n_top_genes,
        prior_graph_file=prior_graph_file,
        gene_names_file=gene_names_file,
        use_mean_trajectory=True
    )

    trajectories = data['trajectories']
    gene_names = data['gene_names']
    prior_adjacency = data['prior_adjacency']
    n_genes = data['n_genes']

    # Extract node features
    node_features = torch.tensor(trajectories[:, 0, :], dtype=torch.float32)

    print(f"Data loaded:")
    print(f"  Node features shape: {node_features.shape}")
    print(f"  Number of genes: {n_genes}")
    print(f"  Gene names: {gene_names[:5]}..." if gene_names is not None else "  No gene names")

    # Load trained model
    # Training parameters

    n_heads = 1
    feat_dropout = 0.1
    attn_dropout = 0.1
    activation_func = torch.nn.Tanh()
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
    graphs, attention_matrices, predictions, targets = infer_graphs_at_timepoints(
        model,
        node_features,
        prior_adjacency,
        device,
        threshold=attention_threshold
    )

    print(f"Inferred {len(graphs)} graphs")
    print(f"Average edges per graph: {np.mean([len(G.edges()) for G in graphs]):.1f}")

    # Visualize graphs
    print("\nCreating graph visualizations...")
    visualize_graphs_at_timepoints(
        graphs,
        attention_matrices,
        gene_names=gene_names,
        save_dir=save_dir,
        timepoints_to_show=timepoints_to_show
    )

    # Visualize predictions
    if predictions is not None and targets is not None:
        print("\nCreating prediction visualizations...")
        visualize_predictions(
            predictions,
            targets,
            gene_names=gene_names,
            save_dir=save_dir,
            timepoints_to_show=timepoints_to_show
        )

        # Visualize gene expression trajectories
        print("\nCreating gene expression trajectory visualizations...")
        visualize_gene_expression_trajectories(
            node_features,
            predictions,
            targets,
            gene_names=gene_names,
            save_dir=save_dir,
            genes_to_show=None  # Set to None for all genes, or specify list like [0, 1, 2, 5]
        )
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

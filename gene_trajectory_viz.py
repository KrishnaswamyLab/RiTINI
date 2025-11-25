import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import argparse
import os

from ritini.data.trajectory_loader import prepare_trajectories_data
from ritini.data.temporal_graph import TemporalGraphDataset
from ritini.utils.utils import load_config, get_device, load_trained_model
from ritini.utils.attention_graphs import adjacency_to_edge_index


def main(checkpoint_path: str, visualization_config_path: str):
    """Main visualization pipeline for gene expression predictions.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint (.pt file).
                        The config.yaml is expected to be in the same directory.
    """
    
    # Derive paths from checkpoint location
    output_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(output_dir, 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}. "
                               f"Expected config.yaml in the same directory as checkpoint.")
    
    # Load configuration
    config = load_config(config_path)
    
    # Device configuration
    device = get_device(config['device'])
    print(f"Using device: {device}")
    
    # Model config
    model_config = config['model']
    
    # Batching parameters
    time_window = config['batching']['time_window']
    history_length = config['batching'].get('history_length', time_window)
    batch_size = config['batching'].get('batch_size', 4)
    
    # Data paths (could also be in config)
    trajectory_file = config.get('data', {}).get('trajectory_file', 'data/processed/trajectory.npy')
    prior_graph_adjacency_file = config.get('data', {}).get('prior_adjacency_file', 'data/processed/prior_adjacency.npy')
    gene_names_file = config.get('data', {}).get('gene_names_file', 'data/processed/gene_names.txt')
    
    # Visualization config
    viz_config = load_config(visualization_config_path)
    dt = viz_config.get('dt', 0.1)
    n_genes_to_plot = viz_config.get('n_genes_to_plot', None)  # None = all genes
    
    # Output configuration
    plots_dir = Path(output_dir) / viz_config['plots_dir']
    plots_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data = prepare_trajectories_data(
        trajectory_file=trajectory_file,
        prior_graph_adjacency_file=prior_graph_adjacency_file,
        gene_names_file=gene_names_file
    )
    
    trajectories = data['trajectories']
    gene_names = data['gene_names']
    prior_adjacency = data['prior_adjacency']
    n_genes = data['n_genes']
    n_timepoints = data['n_timepoints']
    
    prior_adjacency = torch.tensor(prior_adjacency, dtype=torch.float32).to(device)
    
    # Extract trajectory
    trajectory_idx = 0
    signals = trajectories[:, trajectory_idx, :].astype(np.float32)
    
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Number of genes: {n_genes}")
    print(f"  Number of timepoints: {n_timepoints}")
    print(f"  Gene names: {gene_names[:5]}..." if len(gene_names) > 5 else f"  Gene names: {gene_names}")
    
    # Load trained model
    print("\nLoading trained model...")
    model, _, mean, std = load_trained_model(checkpoint_path, model_config, device)
    
    print(f"  Model loaded from {checkpoint_path}")
    print(f"  Normalization: mean={mean:.3f}, std={std:.3f}")
    
    # Normalize signals
    signals_normalized = (signals - mean) / std
    train_node_features = torch.tensor(signals_normalized, dtype=torch.float32)
    
    # Get edge_index
    edge_index = adjacency_to_edge_index(prior_adjacency).to(device)
    t_eval = torch.arange(1, time_window, device=device) * dt
    
    # Select genes to plot
    gene_variances = np.var(signals, axis=0)
    if n_genes_to_plot is None:
        # Plot all genes
        gene_indices = np.arange(n_genes)
    else:
        # Plot top n genes by variance
        gene_indices = np.argsort(gene_variances)[-n_genes_to_plot:][::-1]
    
    n_genes_plot = len(gene_indices)
    print(f"\nPlotting {n_genes_plot} genes (out of {n_genes} total)")
    
    time = np.arange(n_timepoints) * dt
    
    # ============================================================
    # PLOT 1: Autoregressive Rollout
    # ============================================================
    print("\nCreating Plot: Autoregressive rollout...")
    
    # Calculate grid dimensions
    n_cols = min(3, n_genes_plot)
    n_rows = (n_genes_plot + n_cols - 1) // n_cols
    fig_height = 5 * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, fig_height))
    
    # Handle case where there's only 1 subplot
    if n_genes_plot == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    current_t = history_length
    rollout_predictions = []
    rollout_times = []
    
    temp_features = train_node_features.clone()
    
    with torch.no_grad():
        while current_t < n_timepoints - time_window:
            x_history = temp_features[current_t-history_length:current_t].T.unsqueeze(-1).to(device)
            pred_traj, _ = model(x_history, edge_index, t_eval)
            pred_traj = pred_traj.squeeze(-1).cpu()
            
            for i in range(len(pred_traj)):
                rollout_predictions.append(pred_traj[i].numpy())
                rollout_times.append(current_t + 1 + i)
            
            temp_features[current_t + time_window - 1] = pred_traj[-1]
            current_t += time_window - 1
    
    rollout_predictions = np.array(rollout_predictions)
    rollout_predictions = rollout_predictions * std + mean
    rollout_times = np.array(rollout_times) * dt
    
    for plot_idx, gene_idx in enumerate(gene_indices):
        ax = axes_flat[plot_idx]
        
        ax.plot(time, signals[:, gene_idx], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(rollout_times, rollout_predictions[:, gene_idx], 'r-', linewidth=2, 
                label='Autoregressive', alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Expression', fontsize=11)
        ax.set_title(f'{gene_names[gene_idx]}', fontsize=12)
        ax.grid(alpha=0.3)
        if plot_idx == 0:
            ax.legend(fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_genes_plot, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle('Autoregressive Rollout: Long-Term Prediction', fontsize=16)
    plt.tight_layout()
    
    save_path = plots_dir / "prediction_autoregressive.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # ============================================================
    # PLOT 2: Mean ± Std
    # ============================================================
    print("Creating Plot: Mean ± std...")
    
    dataset = TemporalGraphDataset(
        node_features=train_node_features,
        time_window=time_window,
        history_length=history_length
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_start_indices = []
    
    with torch.no_grad():
        window_idx = 0
        for batch in dataloader:
            node_features = batch['node_features'].to(device)
            history = batch['history'].to(device)
            batch_size_actual = node_features.shape[0]
            
            for b in range(batch_size_actual):
                x_history = history[b].T.unsqueeze(-1)
                pred_traj, _ = model(x_history, edge_index, t_eval)
                pred_traj = pred_traj.squeeze(-1).cpu().numpy()
                
                all_predictions.append(pred_traj)
                all_start_indices.append(window_idx + history_length)
                window_idx += 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, fig_height))
    
    # Handle case where there's only 1 subplot
    if n_genes_plot == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    for plot_idx, gene_idx in enumerate(gene_indices):
        ax = axes_flat[plot_idx]
        
        ax.plot(time, signals[:, gene_idx], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
        
        pred_by_time = {t: [] for t in range(1, n_timepoints)}
        
        for pred, start_idx in zip(all_predictions, all_start_indices):
            for i, t in enumerate(range(start_idx+1, start_idx+1+len(pred))):
                if t < n_timepoints:
                    pred_by_time[t].append(pred[i, gene_idx] * std + mean)
        
        pred_times = []
        pred_means = []
        pred_stds = []
        
        for t in sorted(pred_by_time.keys()):
            if len(pred_by_time[t]) > 0:
                pred_times.append(time[t])
                pred_means.append(np.mean(pred_by_time[t]))
                pred_stds.append(np.std(pred_by_time[t]))
        
        pred_times = np.array(pred_times)
        pred_means = np.array(pred_means)
        pred_stds = np.array(pred_stds)
        
        ax.plot(pred_times, pred_means, 'r-', linewidth=2, label='Mean Prediction')
        ax.fill_between(pred_times, pred_means - pred_stds, pred_means + pred_stds, 
                        color='r', alpha=0.3, label='±1 Std')
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Expression', fontsize=11)
        ax.set_title(f'{gene_names[gene_idx]}', fontsize=12)
        ax.grid(alpha=0.3)
        if plot_idx == 0:
            ax.legend(fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_genes_plot, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle('Mean ± Std Across All Sliding Windows', fontsize=16)
    plt.tight_layout()
    
    save_path = plots_dir / "prediction_mean_std.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    print("\n" + "="*60)
    print(f"All visualizations saved to: {plots_dir}")
    print("="*60)
    
    return rollout_predictions, rollout_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize RiTINI gene expression predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--visualization-config', type=str, required=False, default='configs/visualization.yaml',
                        help='Path to the visualization config file (.yaml file) [default: configs/visualization.yaml]')
    args = parser.parse_args()
    
    main(checkpoint_path=args.checkpoint,visualization_config_path=args.visualization_config)

# Usage: uv run visualize_predictions.py --checkpoint /path/to/output/best_model.pt
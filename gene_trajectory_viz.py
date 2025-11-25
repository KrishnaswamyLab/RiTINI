import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from ritini.models.RiTINI import RiTINI
from ritini.data.trajectory_loader import prepare_trajectories_data
from ritini.data.temporal_graph import TemporalGraphDataset
from ritini.utils.attention_graphs import adjacency_to_edge_index

def visualize_predictions(model_path='output/best_model.pt', 
                          time_window=5, history_length=5, batch_size=4, device='cpu'):
    """
    Visualize gene expression predictions with history encoder.
    """
    print("Loading data and model...")
    
    # Data paths
    processed_trajectory_file = 'data/processed/trajectory.npy' 
    prior_adjacency_file = 'data/processed/prior_adjacency.npy'
    gene_names_file = 'data/processed/gene_names.txt'    
    
    # Load model checkpoint first to get n_genes
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    n_genes_from_checkpoint = checkpoint.get('n_genes', None)
    
    # Load gene data
    # Prepare input data
    data = prepare_trajectories_data(
        trajectory_file=processed_trajectory_file,
        prior_graph_adjacency_file=prior_adjacency_file,
        gene_names_file=gene_names_file)
    
    trajectories = data['trajectories']
    gene_names = data['gene_names']
    prior_adjacency = data['prior_adjacency']
    n_genes = data['n_genes']
    n_timepoints = data['n_timepoints']
    
    prior_adjacency = torch.tensor(prior_adjacency, dtype=torch.float32).to(device)  # Shape: (n_genes, n_genes)
    
    # Extract trajectory
    trajectory_idx = 0
    signals = trajectories[:, trajectory_idx, :].astype(np.float32)
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get normalization from checkpoint
    mean = checkpoint['mean']
    std = checkpoint['std']
    
    # Normalize signals
    signals_normalized = (signals - mean) / std
    train_node_features = torch.tensor(signals_normalized, dtype=torch.float32)
    
    # Load model
    model = RiTINI(
        in_features=1,
        out_features=1,
        latent_dim=16,
        history_length=history_length,
        n_heads=1,
        feat_dropout=0.0,
        attn_dropout=0.0,
        device=device
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded. Best training loss: {checkpoint['loss']:.6f}")
    print(f"Number of genes: {n_genes}")
    print(f"Number of timepoints: {n_timepoints}")
    
    # Get edge_index
    edge_index = adjacency_to_edge_index(prior_adjacency).to(device)
    dt = 0.1
    t_eval = torch.arange(1, time_window, device=device) * dt
    
    # Create plots directory
    plots_dir = Path("output/plots_genes")
    plots_dir.mkdir(exist_ok=True)
    
    # Select 6 genes to plot (choose most variable ones for interesting plots)
    gene_variances = np.var(signals, axis=0)
    top_gene_indices = np.argsort(gene_variances)[-6:][::-1]
    
    time = np.arange(n_timepoints) * 0.1
    
    # # ============================================================
    # # PLOT 1: Single Window Close-up
    # # ============================================================
    # print("Creating Plot 1: Single window close-up...")
    
    # start_t = 50  # Middle of trajectory
    # fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # axes = axes.flatten()
    
    # with torch.no_grad():
    #     x_history = train_node_features[start_t-history_length:start_t].T.unsqueeze(-1).to(device)
    #     pred_traj, _ = model(x_history, edge_index, t_eval)
    #     pred_traj = pred_traj.squeeze(-1).cpu().numpy()
    #     # Denormalize
    #     pred_traj = pred_traj * std + mean
    
    # for plot_idx, gene_idx in enumerate(top_gene_indices):
    #     ax = axes[plot_idx]
        
    #     window_times = time[start_t:start_t+time_window]
    #     window_truth = signals[start_t:start_t+time_window, gene_idx]
        
    #     ax.plot(window_times, window_truth, 'ko-', linewidth=2, markersize=8, 
    #             label='Ground Truth', alpha=0.8)
        
    #     pred_times = time[start_t+1:start_t+time_window]
    #     ax.plot(pred_times, pred_traj[:, gene_idx], 'ro-', linewidth=2, markersize=8,
    #             label='Prediction', alpha=0.8)
        
    #     ax.plot(window_times[0], window_truth[0], 'bs', markersize=12, 
    #             label='Initial Condition')
        
    #     ax.set_xlabel('Time (s)', fontsize=11)
    #     ax.set_ylabel('Expression', fontsize=11)
    #     ax.set_title(f'{gene_names[gene_idx]}', fontsize=12)
    #     ax.grid(alpha=0.3)
    #     if plot_idx == 0:
    #         ax.legend(fontsize=9)
    
    # plt.suptitle(f'Single Window: 4-Step Prediction from t={start_t*0.1:.1f}s', fontsize=16)
    # plt.tight_layout()
    
    # save_path = plots_dir / "prediction_single_window.png"
    # plt.savefig(save_path, dpi=200, bbox_inches='tight')
    # plt.close()
    # print(f"Saved: {save_path}")
    
    # # ============================================================
    # # PLOT 2: Multiple Non-Overlapping Windows
    # # ============================================================
    # print("Creating Plot 2: Multiple non-overlapping windows...")
    
    # window_starts = range(history_length, n_timepoints - time_window, 15)
    
    # fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # axes = axes.flatten()
    
    # with torch.no_grad():
    #     for plot_idx, gene_idx in enumerate(top_gene_indices):
    #         ax = axes[plot_idx]
            
    #         ax.plot(time, signals[:, gene_idx], 'k-', linewidth=0.5, alpha=0.3)
            
    #         for start_t in window_starts:
    #             x_history = train_node_features[start_t-history_length:start_t].T.unsqueeze(-1).to(device)
    #             pred_traj, _ = model(x_history, edge_index, t_eval)
    #             pred_traj = pred_traj.squeeze(-1).cpu().numpy()
    #             pred_traj = pred_traj * std + mean
                
    #             pred_times = time[start_t+1:start_t+time_window]
    #             ax.plot(pred_times, pred_traj[:, gene_idx], 'r-', linewidth=2, alpha=0.7)
                
    #             ax.plot(time[start_t], signals[start_t, gene_idx], 'bs', markersize=6)
            
    #         ax.set_xlabel('Time (s)', fontsize=11)
    #         ax.set_ylabel('Expression', fontsize=11)
    #         ax.set_title(f'{gene_names[gene_idx]}', fontsize=12)
    #         ax.grid(alpha=0.3)
    
    # plt.suptitle('Multiple Non-Overlapping 4-Step Predictions', fontsize=16)
    # plt.tight_layout()
    
    # save_path = plots_dir / "prediction_multiple_windows.png"
    # plt.savefig(save_path, dpi=200, bbox_inches='tight')
    # plt.close()
    # print(f"Saved: {save_path}")
    
    # ============================================================
    # PLOT 3: Autoregressive Rollout
    # ============================================================
    print("Creating Plot 3: Autoregressive rollout...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
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
    rollout_times = np.array(rollout_times) * 0.1
    
    for plot_idx, gene_idx in enumerate(top_gene_indices):
        ax = axes[plot_idx]
        
        ax.plot(time, signals[:, gene_idx], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
        
        ax.plot(rollout_times, rollout_predictions[:, gene_idx], 'r-', linewidth=2, 
                label='Autoregressive', alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Expression', fontsize=11)
        ax.set_title(f'{gene_names[gene_idx]}', fontsize=12)
        ax.grid(alpha=0.3)
        if plot_idx == 0:
            ax.legend(fontsize=10)
    
    plt.suptitle('Autoregressive Rollout: Long-Term Prediction', fontsize=16)
    plt.tight_layout()
    
    save_path = plots_dir / "prediction_autoregressive.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # ============================================================
    # PLOT 4: Mean ± Std
    # ============================================================
    print("Creating mean ± std plot...")
    
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
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for plot_idx, gene_idx in enumerate(top_gene_indices):
        ax = axes[plot_idx]
        
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
    
    plt.suptitle('Mean ± Std Across All Sliding Windows', fontsize=16)
    plt.tight_layout()
    
    save_path = plots_dir / "prediction_mean_std.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    print("\nAll visualizations complete!")
    print(f"Plots saved in: {plots_dir}/")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_predictions(device=device, history_length=5)
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from ritini.models.RiTINI import RiTINI
from ritini.data.temporal_graph import TemporalGraphDataset
from ritini.utils.attention_graphs import adjacency_to_edge_index

def visualize_predictions(model_path='best_model.pt', data_dir='data', time_window=5, 
                          history_length=5, batch_size=4, device='cpu'):
    """
    Visualize model predictions with history encoder.
    """
    print("Loading data and model...")
    
    # Load data
    signals = np.load(f'{data_dir}/signals.npy')
    adjacency = np.load(f'{data_dir}/adjacency.npy')
    
    # Load normalization stats
    norm_stats = np.load(f'{data_dir}/norm_stats.npy')
    mean, std = norm_stats[0], norm_stats[1]
    
    # Normalize for model input
    signals_normalized = (signals - mean) / std
    
    n_timepoints, n_genes = signals.shape
    prior_adjacency = torch.from_numpy(adjacency).float().to(device)
    train_node_features = torch.tensor(signals_normalized, dtype=torch.float32)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
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
    
    # Get edge_index
    edge_index = adjacency_to_edge_index(prior_adjacency).to(device)
    dt = 0.1
    t_eval = torch.arange(1, time_window, device=device) * dt
    
    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    nodes_to_plot = min(6, n_genes)
    time = np.arange(n_timepoints) * 0.1
    
    # # ============================================================
    # # PLOT 1: Single Window Close-up (starting at t=100)
    # # ============================================================
    # print("Creating Plot 1: Single window close-up...")
    
    # start_t = 100
    # fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # axes = axes.flatten()
    
    # with torch.no_grad():
    #     # Get history
    #     x_history = train_node_features[start_t-history_length:start_t].T.unsqueeze(-1).to(device)
    #     pred_traj, _ = model(x_history, edge_index, t_eval)
    #     pred_traj = pred_traj.squeeze(-1).cpu().numpy()
    #     # Denormalize
    #     pred_traj = pred_traj * std + mean
    
    # for node_idx in range(nodes_to_plot):
    #     ax = axes[node_idx]
        
    #     # Ground truth window
    #     window_times = time[start_t:start_t+time_window]
    #     window_truth = signals[start_t:start_t+time_window, node_idx]
        
    #     # Plot ground truth
    #     ax.plot(window_times, window_truth, 'ko-', linewidth=2, markersize=8, 
    #             label='Ground Truth', alpha=0.8)
        
    #     # Plot prediction
    #     pred_times = time[start_t+1:start_t+time_window]
    #     ax.plot(pred_times, pred_traj[:, node_idx], 'ro-', linewidth=2, markersize=8,
    #             label='Prediction', alpha=0.8)
        
    #     # Mark initial condition
    #     ax.plot(window_times[0], window_truth[0], 'bs', markersize=12, 
    #             label='Initial Condition')
        
    #     ax.set_xlabel('Time (s)', fontsize=11)
    #     ax.set_ylabel('Signal', fontsize=11)
    #     ax.set_title(f'Node {node_idx}', fontsize=12)
    #     ax.grid(alpha=0.3)
    #     ax.legend(fontsize=9)
    
    # plt.suptitle(f'Single Window: 4-Step Prediction from t={start_t*0.1:.1f}s', fontsize=16)
    # plt.tight_layout()
    
    # save_path = plots_dir / "prediction_single_window.png"
    # plt.savefig(save_path, dpi=200, bbox_inches='tight')
    # plt.close()
    # print(f"Saved: {save_path}")
    
    # ============================================================
    # PLOT 2: Multiple Non-Overlapping Windows
    # ============================================================
    print("Creating Plot 2: Multiple non-overlapping windows...")
    
    # Space windows every 50 timesteps (but start from history_length)
    window_starts = range(history_length, n_timepoints - time_window, 50)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    with torch.no_grad():
        for node_idx in range(nodes_to_plot):
            ax = axes[node_idx]
            
            # Plot full ground truth as faint background
            ax.plot(time, signals[:, node_idx], 'k-', linewidth=0.5, alpha=0.3)
            
            for start_t in window_starts:
                x_history = train_node_features[start_t-history_length:start_t].T.unsqueeze(-1).to(device)
                pred_traj, _ = model(x_history, edge_index, t_eval)
                pred_traj = pred_traj.squeeze(-1).cpu().numpy()
                pred_traj = pred_traj * std + mean
                
                # Plot prediction
                pred_times = time[start_t+1:start_t+time_window]
                ax.plot(pred_times, pred_traj[:, node_idx], 'r-', linewidth=2, alpha=0.7)
                
                # Mark initial condition
                ax.plot(time[start_t], signals[start_t, node_idx], 'bs', markersize=6)
            
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Signal', fontsize=11)
            ax.set_title(f'Node {node_idx}', fontsize=12)
            ax.grid(alpha=0.3)
    
    plt.suptitle('Multiple Non-Overlapping 4-Step Predictions', fontsize=16)
    plt.tight_layout()
    
    save_path = plots_dir / "prediction_multiple_windows.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # ============================================================
    # PLOT 3: Autoregressive Rollout
    # ============================================================
    print("Creating Plot 3: Autoregressive rollout...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Start from t=history_length and predict forward autoregressively
    current_t = history_length
    rollout_predictions = []
    rollout_times = []
    
    # Make a copy for autoregressive updates
    temp_features = train_node_features.clone()
    
    with torch.no_grad():
        while current_t < n_timepoints - time_window:
            x_history = temp_features[current_t-history_length:current_t].T.unsqueeze(-1).to(device)
            pred_traj, _ = model(x_history, edge_index, t_eval)
            pred_traj = pred_traj.squeeze(-1).cpu()
            
            # Store predictions
            for i in range(len(pred_traj)):
                rollout_predictions.append(pred_traj[i].numpy())
                rollout_times.append(current_t + 1 + i)
            
            # Use last prediction as next data point (stay normalized)
            temp_features[current_t + time_window - 1] = pred_traj[-1]
            current_t += time_window - 1
    
    rollout_predictions = np.array(rollout_predictions)
    # Denormalize
    rollout_predictions = rollout_predictions * std + mean
    rollout_times = np.array(rollout_times) * 0.1
    
    for node_idx in range(nodes_to_plot):
        ax = axes[node_idx]
        
        # Plot ground truth
        ax.plot(time, signals[:, node_idx], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
        
        # Plot autoregressive predictions
        ax.plot(rollout_times, rollout_predictions[:, node_idx], 'r-', linewidth=2, 
                label='Autoregressive Prediction', alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Signal', fontsize=11)
        ax.set_title(f'Node {node_idx}', fontsize=12)
        ax.grid(alpha=0.3)
        if node_idx == 0:
            ax.legend(fontsize=10)
    
    plt.suptitle('Autoregressive Rollout: Long-Term Prediction', fontsize=16)
    plt.tight_layout()
    
    save_path = plots_dir / "prediction_autoregressive.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    print("\nAll visualizations complete!")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_predictions(device=device, history_length=5)
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ritini.data.trajectory_loader import process_single_trajectory_data
from ritini.utils.attention_graphs import adjacency_to_edge_index
from ritini.models.RiTINI import RiTINI


def visualize_dynamics_prediction(
    model_path: str,
    trajectory_file: str,
    granger_p_val_file: str,
    granger_coef_file: str,
    gene_names_file: str,
    save_dir: str = 'visualizations_prediction',
    time_window: int = 4,
    genes_to_show: int = 9,
    device: str = 'cpu'
):
    """
    Load a trained RiTINI model and plot predicted vs actual node values for the
    next timepoint using sliding windows sampled the same way as training.

    Outputs:
      - per-gene predicted vs actual plots (for a subset of genes)
      - MSE over time (one value per predicted timepoint)
    """

    device = torch.device(device)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load data (uses the project's loader to keep behaviour consistent)
    data = process_single_trajectory_data(
        trajectory_file=trajectory_file,
        granger_pval_file=granger_p_val_file,
        granger_coef_file=granger_coef_file,
        gene_names_file=gene_names_file
    )

    node_features = data['node_features']  # (n_timepoints, n_genes)
    prior_adj = data['prior_adjacency']
    gene_names = data.get('gene_names', None)

    n_timepoints, n_genes = node_features.shape

    # Build model and load checkpoint
    model = RiTINI(
        in_features=1,
        out_features=1,
        device=device,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Prepare sliding windows exactly like training: windows of length `time_window`
    starts = list(range(0, n_timepoints - time_window + 1))
    if len(starts) == 0:
        raise ValueError('Time series too short for the requested window size')

    # edge_index uses the prior adjacency (static prior for visualization)
    edge_index = adjacency_to_edge_index(prior_adj).to(device)

    preds = []
    targets = []
    times = []

    with torch.no_grad():
        for s in starts:
            window = node_features[s:s + time_window]  # (time_window, n_genes)
            input_seq = window[:-1].to(device)  # (T_in, n_genes)
            target = window[-1].to(device)       # (n_genes,)

            pred, _ = model(input_seq, edge_index)
            # pred shape expected (n_genes, 1) or (n_genes,)
            pred = pred.squeeze(-1).cpu()
            preds.append(pred.numpy())
            targets.append(target.cpu().numpy())
            # associate prediction with the time index of the target (end of window)
            times.append(s + time_window - 1)

    preds = np.stack(preds, axis=0)    # (n_windows, n_genes)
    targets = np.stack(targets, axis=0)
    times = np.array(times)

    # Compute MSE per window (averaged across genes)
    mse_per_window = ((preds - targets) ** 2).mean(axis=1)

    # Plot MSE over prediction timepoints
    plt.figure(figsize=(10, 4))
    plt.plot(times, mse_per_window, '-o')
    plt.xlabel('Time index (target timepoint)')
    plt.ylabel('MSE (avg across genes)')
    plt.title('Prediction MSE over time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'prediction_mse_over_time.png', dpi=150)
    plt.close()

    # Choose genes to show
    if gene_names is None:
        gene_labels = [f'gene_{i}' for i in range(n_genes)]
    else:
        gene_labels = gene_names

    # Select indices for plotting (spread across gene set)
    n_show = min(genes_to_show, n_genes)
    if n_show <= 0:
        n_show = min(9, n_genes)

    if n_show == n_genes:
        selected = list(range(n_genes))
    else:
        selected = list(np.linspace(0, n_genes - 1, n_show, dtype=int))

    # Create per-gene timeseries plots (actual vs predicted)
    rows = int(np.ceil(np.sqrt(n_show)))
    cols = int(np.ceil(n_show / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i, g in enumerate(selected):
        ax = axes[i]
        ax.plot(times, targets[:, g], 'k-', label='actual')
        ax.plot(times, preds[:, g], 'r--', label='pred')
        ax.set_title(gene_labels[g] if g < len(gene_labels) else f'gene_{g}')
        ax.set_xlabel('Time index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path / 'predicted_vs_actual_genes.png', dpi=150)
    plt.close()

    print(f'Prediction visualizations saved to {save_path}')


if __name__ == '__main__':
    # Example usage (adjust paths as needed)
    visualize_dynamics_prediction(
        model_path='best_model.pt',
        trajectory_file='data/data/traj_data.npy',
        granger_p_val_file='data/data/granger_RGtoIPCtoNeuron_p.csv',
        granger_coef_file='data/data/granger_RGtoIPCtoNeuron_c.csv',
        gene_names_file='data/data/gene_names.txt',
        save_dir='visualizations_prediction',
        time_window=4,
        genes_to_show=9,
        device='cpu'
    )

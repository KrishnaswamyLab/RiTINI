import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
import seaborn as sns


def plot_training_history(
    train_history: Dict[str, List[float]],
    val_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """Plot training history."""
    metrics = list(train_history.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            epochs = range(1, len(train_history[metric]) + 1)

            ax.plot(epochs, train_history[metric], label=f'Train {metric.capitalize()}', linewidth=2)
            ax.plot(epochs, val_history[metric], label=f'Val {metric.capitalize()}', linewidth=2)

            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} over Training')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_predictions_vs_actual(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_samples: int = 5,
    n_genes: int = 10,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot predictions vs actual values for a subset of samples and genes.

    Args:
        predictions: Predicted values of shape (n_samples, n_genes)
        targets: Actual values of shape (n_samples, n_genes)
        n_samples: Number of samples to plot
        n_genes: Number of genes to plot per sample
        save_path: Path to save the plot
        figsize: Figure size
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    n_samples = min(n_samples, predictions.shape[0])
    n_genes = min(n_genes, predictions.shape[1])

    fig, axes = plt.subplots(n_samples, 1, figsize=(figsize[0], figsize[1] * n_samples / 5))
    if n_samples == 1:
        axes = [axes]

    for sample_idx in range(n_samples):
        ax = axes[sample_idx]

        # Select random genes to plot
        gene_indices = np.random.choice(predictions.shape[1], n_genes, replace=False)
        gene_indices = np.sort(gene_indices)

        x_pos = np.arange(n_genes)

        pred_values = predictions[sample_idx, gene_indices]
        target_values = targets[sample_idx, gene_indices]

        width = 0.35
        ax.bar(x_pos - width/2, target_values, width, label='Actual', alpha=0.7)
        ax.bar(x_pos + width/2, pred_values, width, label='Predicted', alpha=0.7)

        ax.set_xlabel('Gene Index')
        ax.set_ylabel('Expression Value')
        ax.set_title(f'Sample {sample_idx + 1}: Predictions vs Actual')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Gene {i}' for i in gene_indices])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels if there are many genes
        if n_genes > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_correlation_heatmap(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot correlation heatmap between predictions and targets for each gene.

    Args:
        predictions: Predicted values of shape (n_samples, n_genes)
        targets: Actual values of shape (n_samples, n_genes)
        save_path: Path to save the plot
        figsize: Figure size
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    n_genes = predictions.shape[1]
    correlations = np.zeros(n_genes)

    # Compute correlation for each gene
    for gene_idx in range(n_genes):
        pred_gene = predictions[:, gene_idx]
        target_gene = targets[:, gene_idx]

        if np.std(pred_gene) > 1e-8 and np.std(target_gene) > 1e-8:
            correlations[gene_idx] = np.corrcoef(pred_gene, target_gene)[0, 1]
        else:
            correlations[gene_idx] = 0.0

    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Plot correlation per gene
    ax1.bar(range(n_genes), correlations)
    ax1.set_xlabel('Gene Index')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Prediction Correlation per Gene')
    ax1.grid(True, alpha=0.3)

    # Plot correlation distribution
    ax2.hist(correlations, bins=20, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(correlations), color='red', linestyle='--',
                label=f'Mean: {np.mean(correlations):.3f}')
    ax2.set_xlabel('Correlation')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Gene Correlations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return correlations


def plot_temporal_predictions(
    data: torch.Tensor,
    predictions: torch.Tensor,
    cell_idx: int = 0,
    gene_indices: Optional[List[int]] = None,
    sequence_length: int = 5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 8)
):
    """
    Plot temporal sequence and predictions for specific cell and genes.

    Args:
        data: Original data of shape (n_cells, n_genes, n_time_points)
        predictions: Predictions of shape (n_samples, n_genes)
        cell_idx: Index of cell to plot
        gene_indices: List of gene indices to plot
        sequence_length: Length of input sequences
        save_path: Path to save the plot
        figsize: Figure size
    """
    data = data.cpu().numpy()
    predictions = predictions.cpu().numpy()

    if gene_indices is None:
        gene_indices = list(range(min(5, data.shape[1])))

    n_genes_plot = len(gene_indices)
    n_time_points = data.shape[2]

    fig, axes = plt.subplots(n_genes_plot, 1, figsize=(figsize[0], figsize[1] * n_genes_plot / 5))
    if n_genes_plot == 1:
        axes = [axes]

    for i, gene_idx in enumerate(gene_indices):
        ax = axes[i]

        # Plot actual time series
        time_points = np.arange(n_time_points)
        actual_values = data[cell_idx, gene_idx, :]
        ax.plot(time_points, actual_values, 'b-', label='Actual', linewidth=2)

        # Plot predictions for each possible sequence
        n_sequences = n_time_points - sequence_length
        for seq_idx in range(n_sequences):
            pred_time = sequence_length + seq_idx
            # Find corresponding prediction (this is simplified)
            if seq_idx < len(predictions):
                pred_value = predictions[seq_idx, gene_idx]
                ax.scatter(pred_time, pred_value, color='red', s=50, alpha=0.7)

        # Mark sequence boundaries
        for seq_start in range(0, n_time_points - sequence_length + 1, sequence_length):
            ax.axvline(x=seq_start, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Time Point')
        ax.set_ylabel('Expression Value')
        ax.set_title(f'Cell {cell_idx}, Gene {gene_idx}: Temporal Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Add legend for predictions
    axes[0].scatter([], [], color='red', s=50, alpha=0.7, label='Predictions')
    axes[0].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_loss_landscape(
    train_history: Dict[str, List[float]],
    val_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """Plot loss landscape with moving averages."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(train_history['loss']) + 1)

    # Plot raw losses
    ax1.plot(epochs, train_history['loss'], label='Train Loss', alpha=0.7)
    ax1.plot(epochs, val_history['loss'], label='Val Loss', alpha=0.7)

    # Plot moving averages
    window = max(1, len(epochs) // 10)
    if len(epochs) > window:
        train_ma = np.convolve(train_history['loss'], np.ones(window)/window, mode='valid')
        val_ma = np.convolve(val_history['loss'], np.ones(window)/window, mode='valid')
        ma_epochs = range(window, len(epochs) + 1)

        ax1.plot(ma_epochs, train_ma, '--', label=f'Train MA({window})', linewidth=2)
        ax1.plot(ma_epochs, val_ma, '--', label=f'Val MA({window})', linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot correlation
    ax2.plot(epochs, train_history['correlation'], label='Train Correlation', linewidth=2)
    ax2.plot(epochs, val_history['correlation'], label='Val Correlation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Prediction Correlation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
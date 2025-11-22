import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch


def prepare_trajectories_data(
    trajectory_file: str,
    gene_names_file: str,
    prior_graph_adjacency_file: str,
) -> Dict[str, any]:
    """
    Run compute_prior_graphs.py to precompute prior adjacency matrix.

    Load processed data and prepare trajectory data for RiTINI.
    Note: This function assumes that you have already precomputed the prior adjacency matrix
    and saved it to a file. It also assumes that the trajectories dimension is matching the prior adjacency.



    Args:
        trajectory_file: Path to trajectory file with shape (n_timepoints, n_trajectories, n_genes)
        n_top_genes: Number of genes to subsample for node features
        prior_graph_file: Path to prior graph file (NetworkX graph pickle)
        use_mean_trajectory: Whether to average across trajectories dimension

    Returns:
        Dictionary containing:
            - trajectories: Full trajectory data (n_timepoints, n_trajectories, n_genes)
            - prior_adjacency: Prior graph as adjacency matrix (n_top_genes, n_top_genes)
            - n_genes: Total number of genes in input
            - n_timepoints: Number of timepoints
            - node_features: Subsampled node features (n_timepoints, n_top_genes) tensor
    """
    # Load trajectory data from file
    trajectory_path = Path(trajectory_file)
    trajectories = np.load(trajectory_path)  # Shape: (n_timepoints, n_trajectories, n_genes)

    print(f"Trajectory shape: {trajectories.shape}")

    # Add trajectory dimension if needed: (n_timepoints, n_genes) -> (n_timepoints, 1, n_genes)
    if trajectories.ndim == 2:
        trajectories = trajectories[:, np.newaxis, :]

    gene_names_path = Path(gene_names_file)
    with open(gene_names_path, "r") as f:
        gene_names = [line.strip() for line in f.readlines()]
    
    # Load prior graph from file
    prior_graph_adjacency_path = Path(prior_graph_adjacency_file)
    with open(prior_graph_adjacency_path, "rb") as f:
        prior_adjacency = pickle.load(f)


    n_timepoints, n_trajectories, n_genes = trajectories.shape

    # Extract node features
    node_features = torch.tensor(
        trajectories,
        dtype=torch.float32
    )  # Shape: (n_timepoints, n_trajectories, n_top_genes)


    return {
        'trajectories': trajectories,
        'prior_adjacency': prior_adjacency,
        'gene_names': gene_names,
        'n_genes': n_genes,
        'n_timepoints': n_timepoints,
        'node_features': node_features
    }

import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import scanpy as sc
import anndata


def prepare_trajectories_data(
    trajectory_file: str,
    n_top_genes: int,
    prior_graph_file: str,
    gene_names_file: str,
    use_mean_trajectory: bool = True
) -> Dict[str, any]:
    """
    Load and prepare trajectory data for RiTINI.

    Args:
        trajectory_file: Path to trajectory file with shape (n_timepoints, n_trajectories, n_genes)
        n_top_genes: Number of genes to subsample for node features
        prior_graph_file: Path to prior graph file (NetworkX graph pickle)
        use_mean_trajectory: Whether to average across trajectories dimension

    Returns:
        Dictionary containing:
            - trajectories: Full trajectory data (n_timepoints, n_trajectories, n_genes)
            - prior_adjacency: Prior graph as adjacency matrix (n_top_genes, n_top_genes)
            - prior_graph: Prior graph as NetworkX object
            - n_genes: Total number of genes in input
            - n_timepoints: Number of timepoints
            - node_features: Subsampled node features (n_timepoints, n_top_genes) tensor
    """
    # Load trajectory data from file
    trajectory_path = Path(trajectory_file)
    with open(trajectory_path, "rb") as f:
        trajectories = pickle.load(f)

    print(trajectories.shape)

    # Load prior graph from file
    prior_graph_path = Path(prior_graph_file)
    with open(prior_graph_path, "rb") as f:
        prior_graph = pickle.load(f)

    gene_names_path = Path(gene_names_file)
    with open(gene_names_path, "r") as f:
        gene_names = [line.strip() for line in f.readlines()]
    

    n_timepoints, n_trajectories, n_genes = trajectories.shape

    # Select highly variable genes using scanpy
    # Reshape trajectories to (n_samples, n_genes) for scanpy
    # Combine timepoints and trajectories into samples dimension
    entire_trajectory = trajectories.reshape(-1, n_genes)

    # Create AnnData object for scanpy
    adata = anndata.AnnData(X=entire_trajectory)
    adata.var_names = gene_names
    print(f"Identifying {n_top_genes} highly variable genes.")
    # Identify highly variable genes
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
    )

    # Get indices of highly variable genes
    selected_genes = np.where(adata.var['highly_variable'])[0]

    # Filter trajectories to selected genes
    filtered_gene_names = adata.var_names[selected_genes]
    filtered_trajectories = trajectories[:, :, selected_genes]

    # Extract node features
    if use_mean_trajectory:
        node_features = torch.tensor(
            filtered_trajectories.mean(axis=1),
            dtype=torch.float32
        )  # Shape: (n_timepoints, n_top_genes)
    else:
        node_features = torch.tensor(
            filtered_trajectories,
            dtype=torch.float32
        )  # Shape: (n_timepoints, n_trajectories, n_top_genes)

    # Convert prior graph to adjacency matrix
    n_nodes = len(prior_graph.nodes())
    prior_adjacency = torch.zeros(n_nodes, n_nodes)
    for edge in prior_graph.edges():
        prior_adjacency[edge[0], edge[1]] = 1
        prior_adjacency[edge[1], edge[0]] = 1  # Symmetric

    return {
        'trajectories': filtered_trajectories,
        'prior_adjacency': prior_adjacency,
        'prior_graph': prior_graph,
        'gene_names': filtered_gene_names,
        'n_genes': n_genes,
        'n_timepoints': n_timepoints,
        'node_features': node_features
    }

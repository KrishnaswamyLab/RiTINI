import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import scanpy as sc
import anndata


def process_single_trajectory_data(
    trajectory_file: str,
    granger_pval_file: str,
    granger_coef_file: str,
    gene_names_file: str
) -> Dict[str, any]:
    """
    Load and prepare trajectory data for RiTINI.

    Args:
        trajectory_file: Path to trajectory .npy file with shape (n_timepoints, n_genes)
        granger_pval_file: Path to Granger causality p-values CSV
        granger_coef_file: Path to Granger causality coefficients CSV
        gene_names_file: Path to gene names file (hvg.txt)

    Returns:
        Dictionary containing:
            - trajectories: Trajectory data (n_timepoints, n_genes)
            - prior_adjacency: Prior graph as adjacency matrix (n_genes, n_genes)
            - prior_graph: Prior graph as NetworkX object
            - gene_names: List of gene names
            - n_genes: Total number of genes in input
            - n_timepoints: Number of timepoints
            - node_features: Node features (n_timepoints, n_genes) tensor
            - signed_score_df: Signed score dataframe from Granger causality
    """
    # Load trajectory data from file
    trajectories = np.load(trajectory_file)
    print(f"Trajectory shape: {trajectories.shape}")

    # Add trajectory dimension if needed: (n_timepoints, n_genes) -> (n_timepoints, 1, n_genes)
    if trajectories.ndim == 2:
        trajectories = trajectories[:, np.newaxis, :]
        
    # Load gene names
    gene_names = np.loadtxt(gene_names_file, dtype=str)
    
    # TODO: Organize Granger processing 
    #  Load Granger causality results
    pval_df = pd.read_csv(granger_pval_file, index_col=0)
    coef_df = pd.read_csv(granger_coef_file, index_col=0)
    
    # Create signed score dataframe
    log_pval_df = -np.log(pval_df + 2 ** -10)
    coef_sign_df = np.sign(coef_df)
    signed_score_df = log_pval_df * coef_sign_df
    signed_score_df.columns = signed_score_df.columns.str.strip('_y')
    signed_score_df.index = signed_score_df.index.str.strip('_x')
    
    n_timepoints, n_trajectories , n_genes = trajectories.shape

    # Convert trajectories to node features tensor
    # TODO: All trajectories get squeezed as a single one
    
    node_features = torch.tensor(
        trajectories.squeeze(1),
        dtype=torch.float32
    )  # Shape: (n_timepoints, n_genes)

    # Convert signed_score_df to adjacency matrix
    # Map gene names to indices
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}

    # Get indices of genes that appear in signed_score_df
    signed_genes = list(signed_score_df.index)
    signed_indices = [gene_to_idx[gene] for gene in signed_genes if gene in gene_to_idx]

    # Convert signed_score_df to numpy array for easier indexing
    signed_array = signed_score_df.values  # Shape: (n_signed_genes, n_genes)

    # Create the full n_genes x n_genes matrix initialized with zeros
    prior_adjacency = np.zeros((n_genes, n_genes))

    # Place the connections from signed genes to all genes
    for i, gene_idx in enumerate(signed_indices):
        for j, target_gene in enumerate(signed_score_df.columns):
            if target_gene in gene_to_idx:
                target_idx = gene_to_idx[target_gene]
                score = signed_array[i, j]
                if not np.isnan(score) and score != 0:
                    prior_adjacency[gene_idx, target_idx] = 1

    # Make it symmetric - for each connection from gene i to gene j, add connection from j to i
    for i, gene_idx in enumerate(signed_indices):
        for j, target_gene in enumerate(signed_score_df.columns):
            if target_gene in gene_to_idx:
                target_idx = gene_to_idx[target_gene]
                score = signed_array[i, j]
                if not np.isnan(score) and score != 0:
                    prior_adjacency[target_idx, gene_idx] = 1

    # Convert to torch tensor
    prior_adjacency = torch.tensor(prior_adjacency, dtype=torch.float32)

    return {
            'trajectories': trajectories,
            'prior_adjacency': prior_adjacency,
            'gene_names': gene_names,
            'n_genes': n_genes,
            'n_timepoints': n_timepoints,
            'node_features': node_features,
            'signed_score_df': signed_score_df
    }


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

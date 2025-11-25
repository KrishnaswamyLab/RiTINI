import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import anndata
import scanpy as sc
from ritini.utils.prior_graph import compute_prior_adjacency

def process_trajectory_data(raw_trajectory_file, 
                            raw_gene_names_file,
                            interest_genes_file,
                            output_trajectory_file='data/processed/trajectory.npy',
                            output_gene_names_file='data/processed/gene_names.txt',
                            output_prior_adjacency_file='data/processed/prior_adjacency.npy',
                            prior_graph_mode='granger_causality',
                            n_highly_variable_genes=200,
                            **kwargs):

    # Load trajectory data from file
    trajectory_path = Path(raw_trajectory_file)
    trajectories = np.load(trajectory_path)  # Shape: (n_timepoints, n_trajectories, n_genes)

    print(f"Trajectory shape: {trajectories.shape}")

    # Add trajectory dimension if needed: (n_timepoints, n_genes) -> (n_timepoints, 1, n_genes)
    if trajectories.ndim == 2:
        trajectories = trajectories[:, np.newaxis, :]

    # Average across trajectories dimension
    trajectory = trajectories.mean(axis=1, keepdims=True)  # Shape: (n_timepoints, 1, n_genes)
    # Squeeze trajectory dimension for prior computation
    trajectory = trajectory.squeeze(axis=1)  # Shape: (n_timepoints, n_genes)

    gene_names_path = Path(raw_gene_names_file)
    with open(gene_names_path, "r") as f:
        gene_names = [line.strip() for line in f.readlines()]

    # Load interest genes
    interest_genes_path = Path(interest_genes_file)

    with open(interest_genes_path, "r") as f:
        interest_genes = [line.strip() for line in f.readlines()]


    # Filter genes based on highly variable genes
    # Create AnnData object for scanpy
    adata = anndata.AnnData(X=trajectory)
    adata.var_names = gene_names

    print(f"Identifying {n_highly_variable_genes} highly variable genes.")
    # Identify highly variable genes
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_highly_variable_genes,
    )

    # Get indices of highly variable genes
    highly_variable_genes_idx = np.where(adata.var['highly_variable'])[0]
    highly_variable_genes = adata.var_names[highly_variable_genes_idx].tolist()

    # Get indices of interest genes
    selected_genes = highly_variable_genes + interest_genes

    # Filter trajectories to highly variable + interest genes
    # Convert gene names to indices using a dictionary lookup
    gene_name_to_idx = {name: idx for idx, name in enumerate(adata.var_names)}
    selected_gene_indices = np.array([gene_name_to_idx[gene] for gene in selected_genes])
    
    filtered_trajectory = trajectory[:, selected_gene_indices]
    filtered_gene_names = adata.var_names[selected_gene_indices]

    # Compute prior adjacency matrix and retrieve the corresponding gene names
    prior_adjacency, gene_names_in_prior = compute_prior_adjacency(filtered_trajectory,
                                                                     mode=prior_graph_mode, 
                                                                     gene_names=filtered_gene_names, 
                                                                     **kwargs)

    # Save processed files to processed path

    np.save(output_trajectory_file, filtered_trajectory)
    np.save(output_prior_adjacency_file, prior_adjacency)

    # Save gene names as text file (one per line)
    with open(output_gene_names_file, 'w') as f:
        for gene in gene_names_in_prior:
            f.write(f"{gene}\n")

    return output_trajectory_file, output_prior_adjacency_file, output_gene_names_file


if __name__ == "__main__":
    # Example usage
    raw_trajectory_file = 'data/raw/traj_data.npy' 
    raw_gene_names_file='data/raw/gene_names.txt'
    interest_genes_file = 'data/raw/interest_genes.txt'

    # Test the function with sample data
    trajectory_file, prior_adjacency_file, gene_names_file = process_trajectory_data(
        raw_trajectory_file=raw_trajectory_file,
        raw_gene_names_file=raw_gene_names_file,
        interest_genes_file=interest_genes_file,
        prior_graph_mode='fully_connected',
        n_highly_variable_genes=200,
        output_dir='data/processed/'
        # db_extract_file = 'data/DatabaseExtract_v_1.01.csv'
    )
    
    print(f"Trajectory file saved at: {trajectory_file}")
    print(f"Prior adjacency file saved at: {prior_adjacency_file}")
    print(f"Gene names file saved at: {gene_names_file}")
    
    # Verify files exist and can be loaded
    assert Path(trajectory_file).exists(), f"Trajectory file not found at {trajectory_file}"
    assert Path(prior_adjacency_file).exists(), f"Prior adjacency file not found at {prior_adjacency_file}"
    assert Path(gene_names_file).exists(), f"Gene names file not found at {gene_names_file}"
    
    print("\nAll files saved successfully!")
#!/usr/bin/env python3
"""
Example usage of the gene expression graph dataloader.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.gene_dataset import create_synthetic_data
from data.gene_graph_dataloader import (
    GeneGraphDataset,
    create_hierarchical_graphs,
    create_gene_dataloader,
    create_gene_dataloader_ddp
)


def main():
    """Demonstrate gene expression graph dataloader usage."""
    print("=== Gene Expression Graph Dataloader Example ===\n")

    # Parameters
    n_cells = 50
    n_genes = 100
    n_time_points = 20
    sequence_length = 5
    prediction_horizon = 1
    batch_size = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    print(f"Data shape: ({n_cells}, {n_genes}, {n_time_points})")
    print(f"Sequence length: {sequence_length}")
    print(f"Prediction horizon: {prediction_horizon}\n")

    # Create synthetic gene expression data
    print("Creating synthetic gene expression data...")
    gene_data = create_synthetic_data(
        n_cells=n_cells,
        n_genes=n_genes,
        n_time_points=n_time_points,
        noise_level=0.1,
        device=device
    )
    print(f"Created data with shape: {gene_data.shape}\n")

    # Example 1: Single-level graph dataset
    print("=== Example 1: Single-level Gene Graph Dataset ===")

    # Create different types of graphs
    graph_types = ['correlation', 'knn', 'fully_connected']

    for graph_type in graph_types:
        print(f"\nTesting {graph_type} graph:")

        # Create dataset
        dataset = GeneGraphDataset(
            gene_data=gene_data,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            graph_type=graph_type,
            correlation_threshold=0.3,
            k_neighbors=10,
            normalize=True,
            device=device
        )

        # Print statistics
        stats = dataset.get_statistics()
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Number of edges: {stats['num_edges']}")
        print(f"  Average degree: {stats['avg_degree']:.2f}")

        # Get a sample
        sample = dataset[0]
        print(f"  Sample node features shape: {sample.x.shape}")
        print(f"  Sample targets shape: {sample.y.shape}")
        print(f"  Sample edge index shape: {sample.edge_index.shape}")

    # Example 2: Hierarchical graphs
    print("\n\n=== Example 2: Hierarchical Gene Expression Graphs ===")

    # Create hierarchical graphs
    print("Creating hierarchical graphs...")
    hierarchical_graphs = create_hierarchical_graphs(
        gene_data=gene_data,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        high_level_graph_type='correlation',
        low_level_graph_type='knn',
        correlation_threshold=0.3,
        k_neighbors=10,
        normalize=True,
        device=device
    )

    high_level_graph = hierarchical_graphs[0]
    low_level_graphs = hierarchical_graphs[1:]

    print(f"High-level graph (cells): {high_level_graph.num_nodes} nodes, {high_level_graph.edge_index.size(1)} edges")
    print(f"Low-level graphs (genes): {len(low_level_graphs)} graphs")
    print(f"Each low-level graph: {low_level_graphs[0].num_nodes} nodes, {low_level_graphs[0].edge_index.size(1)} edges")

    # Example 3: Hierarchical dataloader
    print("\n\n=== Example 3: Hierarchical Dataloader ===")

    # Create dataloader
    print("Creating hierarchical dataloader...")
    dataloader = create_gene_dataloader(
        graphs=hierarchical_graphs,
        batch_size=batch_size,
        permute=True
    )

    print(f"Dataloader created with {len(dataloader)} batches")

    # Iterate through a few batches
    print("\nIterating through batches:")
    for batch_idx, (high_level_batch, low_level_batch, partition_indices) in enumerate(dataloader):
        if batch_idx >= 3:  # Only show first 3 batches
            break

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  High-level batch: {high_level_batch.num_nodes} cells")
        print(f"  Low-level batch: {low_level_batch.num_graphs} cell graphs")
        print(f"  Low-level node features shape: {low_level_batch.x.shape}")
        print(f"  Partition indices: {partition_indices.tolist()}")

    # Example 4: Time series prediction setup
    print("\n\n=== Example 4: Time Series Prediction Format ===")

    dataset = GeneGraphDataset(
        gene_data=gene_data,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        graph_type='correlation',
        correlation_threshold=0.3,
        normalize=True,
        device=device
    )

    # Show how data is structured for prediction
    sample = dataset[0]
    print(f"Input time series (per gene): {sample.x.shape} -> {sequence_length} time points")
    print(f"Target prediction (per gene): {sample.y.shape} -> {prediction_horizon} time points")
    print(f"Graph structure: {sample.edge_index.shape[1]} edges connecting {sample.num_nodes} genes")

    # Show actual values for first few genes
    print(f"\nExample values for first 3 genes:")
    print(f"Input sequence shape: {sample.x[:3].shape}")
    print(f"Input values:\n{sample.x[:3]}")
    print(f"Target values: {sample.y[:3]}")

    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from ritini.utils.prior_graph import compute_prior_adjacency


def _plot_prior_adjacency(prior_adjacency, gene_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_genes = prior_adjacency.shape[0]
    graph = nx.from_numpy_array(np.asarray(prior_adjacency), create_using=nx.DiGraph)

    # Remove zero-weight edges for a cleaner visualization
    edges_to_remove = [
        (u, v) for u, v, data in graph.edges(data=True)
        if data.get('weight', 0.0) <= 0
    ]
    graph.remove_edges_from(edges_to_remove)

    if graph.number_of_nodes() == 0:
        plt.figure(figsize=(8, 6))
        plt.title('Prior graph (empty)')
        plt.text(0.5, 0.5, 'No edges in prior graph', ha='center', va='center')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    node_labels = {i: gene_names[i] if i < len(gene_names) else str(i) for i in graph.nodes()}
    edge_weights = np.array([data['weight'] for _, _, data in graph.edges(data=True)], dtype=float)

    if edge_weights.size > 0:
        min_w = float(edge_weights.min())
        max_w = float(edge_weights.max())
        if max_w > min_w:
            widths = 0.5 + 2.5 * (edge_weights - min_w) / (max_w - min_w)
        else:
            widths = np.full_like(edge_weights, 1.5)
    else:
        widths = np.array([])

    figsize = (12, 10) if n_genes <= 80 else (14, 12)
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=300 if n_genes <= 80 else 160,
        node_color='lightblue',
        edgecolors='black',
        linewidths=0.5,
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        width=widths.tolist() if widths.size > 0 else 1.0,
        alpha=0.7,
        arrows=True,
        arrowsize=12,
        edge_color='gray',
        connectionstyle='arc3,rad=0.08',
    )

    if n_genes <= 40:
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=7)

    plt.title('Prior graph (NetworkX)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_trajectory_data(raw_trajectory_file, 
                            raw_gene_names_file,
                            interest_genes_file,
                            output_trajectory_file='data/processed/trajectory.npy',
                            output_gene_names_file='data/processed/gene_names.txt',
                            output_prior_adjacency_file='data/processed/prior_adjacency.npy',
                            output_prior_graph_plot_file='data/processed/prior_graph.png',
                            prior_graph_mode='granger_causality',
                            n_highly_variable_genes=200,
                            use_existing_prior_adjacency=False,
                            existing_prior_adjacency_file=None,
                            **kwargs):

    gene_names_path = Path(raw_gene_names_file)
    with open(gene_names_path, "r") as f:
        gene_names = [line.strip() for line in f.readlines()]

    # Load a single trajectory matrix from file.
    # Expected shape is genes x time, but time x genes is also accepted.
    trajectory_path = Path(raw_trajectory_file)
    trajectory = np.load(trajectory_path)
    if trajectory.ndim != 2:
        raise ValueError(
            f"Expected a single 2D trajectory matrix, got shape {trajectory.shape}."
        )

    print(f"Trajectory shape: {trajectory.shape}")

    n_genes = len(gene_names)
    if trajectory.shape[0] == n_genes and trajectory.shape[1] != n_genes:
        trajectory_gene_time = trajectory
    elif trajectory.shape[1] == n_genes and trajectory.shape[0] != n_genes:
        trajectory_gene_time = trajectory.T
    elif trajectory.shape[0] == n_genes and trajectory.shape[1] == n_genes:
        # Ambiguous square matrix; default to genes x time per expected convention.
        trajectory_gene_time = trajectory
    else:
        raise ValueError(
            f"Gene names count ({n_genes}) does not match either trajectory dimension {trajectory.shape}."
        )

    # Load interest genes
    interest_genes_path = Path(interest_genes_file)

    with open(interest_genes_path, "r") as f:
        interest_genes = [line.strip() for line in f.readlines()]


    print(f"Identifying {n_highly_variable_genes} highly variable genes.")
    # Highly variable genes are those with largest variance across time.
    gene_variances = np.var(trajectory_gene_time, axis=1)
    top_k = min(n_highly_variable_genes, trajectory_gene_time.shape[0])
    highly_variable_genes_idx = np.argsort(gene_variances)[-top_k:][::-1]
    highly_variable_genes = [gene_names[idx] for idx in highly_variable_genes_idx]

    # Combine HVGs + interest genes while preserving order and uniqueness
    selected_genes = list(dict.fromkeys(highly_variable_genes + interest_genes))
    # Filter trajectory to highly variable + interest genes
    # Convert gene names to indices using a dictionary lookup
    gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    selected_gene_indices = np.array(
        [gene_name_to_idx[gene] for gene in selected_genes if gene in gene_name_to_idx],
        dtype=int,
    )
    
    filtered_trajectory = trajectory_gene_time[selected_gene_indices, :].T
    filtered_gene_names = [gene_names[idx] for idx in selected_gene_indices]

    if use_existing_prior_adjacency:
        source_prior_file = existing_prior_adjacency_file or output_prior_adjacency_file
        source_prior_path = Path(source_prior_file)
        if not source_prior_path.exists():
            raise FileNotFoundError(
                f"Expected existing prior adjacency file at {source_prior_path}, but it does not exist."
            )

        prior_adjacency = np.load(source_prior_path)
        if prior_adjacency.ndim != 2 or prior_adjacency.shape[0] != prior_adjacency.shape[1]:
            raise ValueError(
                f"Existing prior adjacency must be a square 2D matrix. Got shape {prior_adjacency.shape}."
            )

        expected_genes = len(filtered_gene_names)
        if prior_adjacency.shape[0] != expected_genes:
            raise ValueError(
                "Existing prior adjacency shape does not match filtered gene count: "
                f"got {prior_adjacency.shape}, expected ({expected_genes}, {expected_genes})."
            )

        gene_names_in_prior = filtered_gene_names
    else:
        # Compute prior adjacency matrix and retrieve the corresponding gene names
        prior_adjacency, gene_names_in_prior = compute_prior_adjacency(filtered_trajectory,
                                                                        mode=prior_graph_mode, 
                                                                        gene_names=filtered_gene_names, 
                                                                        **kwargs)

    # Save processed files to processed path

    Path(output_trajectory_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_prior_adjacency_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_gene_names_file).parent.mkdir(parents=True, exist_ok=True)

    np.save(output_trajectory_file, filtered_trajectory)
    np.save(output_prior_adjacency_file, prior_adjacency)

    if output_prior_graph_plot_file is not None:
        _plot_prior_adjacency(
            prior_adjacency=prior_adjacency,
            gene_names=gene_names_in_prior,
            output_path=output_prior_graph_plot_file,
        )

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
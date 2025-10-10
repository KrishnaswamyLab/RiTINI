import os 
import numpy as np
import networkx as nx
import pickle
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

def create_temporal_graph_data(
    n_timepoints: int = 10,
    n_trajectories: int = 5,
    n_genes: int = 20,
    edge_density: float = 0.3,
    temporal_noise: float = 0.1,
    n_change: int = 1,
    regulation_strength: float = 0.5,
    save: bool = False,
    save_path: str = 'data'
) -> Tuple[List[np.ndarray], List[List[nx.DiGraph]]]:
    """
    Create n different temporal directed graphs representing Gene Regulatory Networks.
    Gene expression is influenced by regulatory edges from other genes.

    Parameters:
    -----------
    n_timepoints : int
        Number of time points
    n_trajectories : int
        Number of different trajectory types (separate graphs)
    n_genes : int
        Number of genes (nodes in the graph)
    edge_density : float
        Probability of edge existence at t=0 (0 to 1)
    temporal_noise : float
        Amount of temporal variation in features
    n_change : int
        Interval at which exactly two edges change (every n_change timesteps)
    regulation_strength : float
        Strength of regulatory influence (0 to 1)
    save : bool
        Whether to save the data
    save_path : str
        Path to save the data

    Returns:
    --------
    all_node_features : List[np.ndarray]
        List of n_trajectories arrays, each with shape (n_timepoints, n_genes)
    all_graphs : List[List[nx.DiGraph]]
        List of n_trajectories trajectory graphs, each containing n_timepoints directed graphs
    """

    all_node_features = []
    all_graphs = []

    # Create a separate graph for each trajectory
    for traj_idx in range(n_trajectories):
        # Initialize node features for this trajectory
        node_features = np.zeros((n_timepoints, n_genes))

        # Create initial basal expression levels for each gene
        basal_expression = np.random.randn(n_genes) * 0.5
        
        # Initialize base directed adjacency matrix for this trajectory
        base_adjacency = np.random.rand(n_genes, n_genes) < edge_density
        np.fill_diagonal(base_adjacency, False)  # No self-loops
        
        # Create regulation types: activation (+1) or repression (-1)
        regulation_type = np.where(base_adjacency, 
                                   np.random.choice([1, -1], size=(n_genes, n_genes)), 
                                   0)
        
        current_adjacency = base_adjacency.copy()
        current_regulation = regulation_type.copy()

        # Create temporal graphs for this trajectory
        graphs = []

        for t in range(n_timepoints):
            # Calculate gene expression influenced by regulators
            if t == 0:
                # Initial timepoint: use basal expression with temporal pattern
                for gene_idx in range(n_genes):
                    base_pattern = np.sin(gene_idx * 0.5)
                    trajectory_offset = np.random.randn() * 0.3
                    noise = np.random.randn() * temporal_noise
                    node_features[t, gene_idx] = base_pattern + trajectory_offset + basal_expression[gene_idx] + noise
            else:
                # Subsequent timepoints: expression influenced by regulators
                for gene_idx in range(n_genes):
                    # Start with previous expression (momentum)
                    prev_expression = node_features[t-1, gene_idx]
                    
                    # Calculate regulatory input from incoming edges
                    regulators = np.where(current_adjacency[:, gene_idx])[0]
                    regulatory_input = 0
                    
                    if len(regulators) > 0:
                        for reg_idx in regulators:
                            reg_expression = node_features[t-1, reg_idx]
                            reg_type = current_regulation[reg_idx, gene_idx]  # +1 or -1
                            regulatory_input += reg_type * reg_expression * regulation_strength
                    
                    # Temporal evolution with regulation
                    base_pattern = np.sin(np.pi * t / n_timepoints + gene_idx * 0.5)
                    trajectory_offset = basal_expression[gene_idx]
                    noise = np.random.randn() * temporal_noise
                    
                    # Combine: previous state + new pattern + regulatory influence
                    node_features[t, gene_idx] = (0.3 * prev_expression + 
                                                 0.3 * base_pattern + 
                                                 0.2 * trajectory_offset + 
                                                 regulatory_input + 
                                                 noise)

            # Only change exactly 2 edges at intervals of n_change timesteps
            if t > 0 and t % n_change == 0:
                # Get all possible directed edges
                possible_edges = []
                for i in range(n_genes):
                    for j in range(n_genes):
                        if i != j:  # No self-loops
                            possible_edges.append((i, j))
                
                # Randomly select 2 edges to flip
                edges_to_flip = np.random.choice(len(possible_edges), size=2, replace=False)
                
                for edge_idx in edges_to_flip:
                    i, j = possible_edges[edge_idx]
                    # Flip the edge
                    current_adjacency[i, j] = not current_adjacency[i, j]
                    
                    # If edge now exists, assign new regulation type
                    if current_adjacency[i, j]:
                        current_regulation[i, j] = np.random.choice([1, -1])
                    else:
                        current_regulation[i, j] = 0

            # Create directed graph for this timepoint
            G = nx.DiGraph()
            G.add_nodes_from(range(n_genes))

            # Add directed edges with regulation information
            for i in range(n_genes):
                for j in range(n_genes):
                    if current_adjacency[i, j]:
                        reg_type = 'activation' if current_regulation[i, j] == 1 else 'repression'
                        G.add_edge(i, j, 
                                  regulation=reg_type,
                                  weight=current_regulation[i, j])

            # Add node features and metadata as attributes
            for node_idx in range(n_genes):
                G.nodes[node_idx]['feature'] = node_features[t, node_idx]
                G.nodes[node_idx]['expression'] = node_features[t, node_idx]
                G.nodes[node_idx]['timepoint'] = t
                G.nodes[node_idx]['trajectory'] = traj_idx

            graphs.append(G)

        all_node_features.append(node_features)
        all_graphs.append(graphs)

    if save:
        toy_data_directory = os.path.join(save_path, 'toy_data')
        os.makedirs(toy_data_directory, exist_ok=True)
        
        # Save node features
        np.save(os.path.join(toy_data_directory, 'all_node_features.npy'), all_node_features)
        
        # Save NetworkX directed graphs
        with open(os.path.join(toy_data_directory, 'all_graphs.pkl'), 'wb') as f:
            pickle.dump(all_graphs, f)

    return all_node_features, all_graphs


def visualize_temporal_graph(
    node_features: np.ndarray,
    graphs: List[nx.Graph],
    timepoints_to_plot: List[int] = None,
    fixed_layout: bool = True,
    trajectory_idx: int = 0
):
    """
    Visualize the temporal graph at selected timepoints for a specific trajectory.

    Parameters:
    -----------
    node_features : np.ndarray
        Node features array for this trajectory (shape: n_timepoints x n_genes)
    graphs : List[nx.Graph]
        List of graphs for each timepoint
    timepoints_to_plot : List[int]
        Which timepoints to visualize (default: first, middle, last)
    fixed_layout : bool
        If True, use the same layout for all timepoints for easier comparison
    trajectory_idx : int
        Trajectory index for display purposes
    """
    n_timepoints = len(graphs)

    if timepoints_to_plot is None:
        timepoints_to_plot = [0, n_timepoints // 2, n_timepoints - 1]

    fig, axes = plt.subplots(1, len(timepoints_to_plot), figsize=(6*len(timepoints_to_plot), 5))

    if len(timepoints_to_plot) == 1:
        axes = [axes]

    # Use fixed layout based on first graph if requested
    if fixed_layout:
        pos = nx.spring_layout(graphs[0], seed=42)

    for idx, t in enumerate(timepoints_to_plot):
        G = graphs[t]
        ax = axes[idx]

        # Get node colors based on features
        node_colors = [G.nodes[n]['feature'] for n in G.nodes()]

        # Use per-graph layout if not fixed
        if not fixed_layout:
            pos = nx.spring_layout(G, seed=42)

        # Draw graph
        nx.draw(G, pos,
                node_color=node_colors,
                node_size=500,
                cmap='viridis',
                with_labels=True,
                ax=ax,
                edge_color='gray',
                alpha=0.7)

        ax.set_title(f'Trajectory {trajectory_idx}, Timepoint {t}\n({len(G.edges())} edges)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Creating temporal graph data...")
    # Parameters
    n_timepoints = 10
    n_trajectories = 2
    n_genes = 15

    # Create temporal graph data with slow edge changes
    all_node_features, all_graphs = create_temporal_graph_data(
        n_timepoints=n_timepoints,
        n_trajectories=n_trajectories,
        n_genes=n_genes,
        edge_density=0.3,
        temporal_noise=0.1,
        edge_change_rate=0.02  # Very low rate = slow changes (2% of edges change per timestep)
    )

    print(f"Number of trajectories: {len(all_graphs)}")
    print(f"Number of timepoints per trajectory: {len(all_graphs[0])}")

    # Print statistics for each trajectory
    for traj_idx in range(n_trajectories):
        print(f"\n=== Trajectory {traj_idx} ===")
        print(f"Node features shape: {all_node_features[traj_idx].shape}")
        print(f"Number of graphs: {len(all_graphs[traj_idx])}")

        # Print statistics for each timepoint
        print(f"\nGraph statistics over time:")
        for t, G in enumerate(all_graphs[traj_idx]):
            print(f"  t={t}: {len(G.nodes())} nodes, {len(G.edges())} edges, "
                  f"avg degree: {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}")

    # Visualize selected timepoints for first trajectory
    print("\nVisualizing graphs at selected timepoints for Trajectory 0...")
    visualize_temporal_graph(
        all_node_features[0],
        all_graphs[0],
        timepoints_to_plot=[0, 4, 9],
        fixed_layout=True,  # Same layout makes it easier to see edge changes
        trajectory_idx=0
    )

    # Example: Access data at specific timepoint and trajectory
    trajectory = 0
    timepoint = 5
    print(f"\nExample - Features at trajectory {trajectory}, timepoint {timepoint}:")
    print(all_node_features[trajectory][timepoint, :5])  # First 5 genes
    
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import os
import pickle

def visualize_node_features(all_node_features, trajectory_idx=0, save_dir="visualizations_toy_data"):
    """
    Visualize node features (gene expressions) over time.

    Args:
        all_node_features: List of node feature arrays, one per trajectory
        trajectory_idx: Which trajectory to visualize
        save_dir: Directory to save visualizations
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    node_features = all_node_features[trajectory_idx]
    n_timepoints, n_genes = node_features.shape

    # Plot 1: Heatmap of all gene expressions over time
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(node_features.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Gene Index', fontsize=12)
    ax.set_title(f'Gene Expression Over Time (Trajectory {trajectory_idx})', fontsize=14)
    plt.colorbar(im, ax=ax, label='Expression Level')
    plt.tight_layout()
    plt.savefig(save_path / f'gene_expression_heatmap_traj_{trajectory_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Individual gene trajectories
    fig, ax = plt.subplots(figsize=(14, 8))
    for gene_idx in range(n_genes):
        ax.plot(range(n_timepoints), node_features[:, gene_idx],
                alpha=0.7, linewidth=2, label=f'Gene {gene_idx}')

    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Expression Level', fontsize=12)
    ax.set_title(f'Gene Expression Trajectories (Trajectory {trajectory_idx})', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add legend if not too many genes
    if n_genes <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path / f'gene_trajectories_traj_{trajectory_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Statistics over time
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Mean expression
    mean_expr = node_features.mean(axis=1)
    axes[0].plot(range(n_timepoints), mean_expr, 'b-', linewidth=2, marker='o')
    axes[0].set_ylabel('Mean Expression', fontsize=11)
    axes[0].set_title('Mean Gene Expression Over Time', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Standard deviation
    std_expr = node_features.std(axis=1)
    axes[1].plot(range(n_timepoints), std_expr, 'r-', linewidth=2, marker='s')
    axes[1].set_ylabel('Std Expression', fontsize=11)
    axes[1].set_title('Standard Deviation of Gene Expression Over Time', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Range (max - min)
    range_expr = node_features.max(axis=1) - node_features.min(axis=1)
    axes[2].plot(range(n_timepoints), range_expr, 'g-', linewidth=2, marker='^')
    axes[2].set_xlabel('Timepoint', fontsize=11)
    axes[2].set_ylabel('Expression Range', fontsize=11)
    axes[2].set_title('Range of Gene Expression Over Time', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / f'expression_statistics_traj_{trajectory_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved node feature visualizations to {save_dir}/")


def visualize_graphs(all_graphs, trajectory_idx=0, save_dir="visualizations_toy_data",
                     timepoints_to_show=None, fixed_layout=True):
    """
    Visualize temporal graphs at selected timepoints.

    Args:
        all_graphs: List of graph lists, one per trajectory
        trajectory_idx: Which trajectory to visualize
        save_dir: Directory to save visualizations
        timepoints_to_show: List of timepoint indices to show (None = evenly spaced)
        fixed_layout: Use same layout for all timepoints for easier comparison
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    graphs = all_graphs[trajectory_idx]
    n_timepoints = len(graphs)

    if timepoints_to_show is None:
        # Show 6 evenly spaced timepoints
        n_show = min(6, n_timepoints)
        timepoints_to_show = np.linspace(0, n_timepoints - 1, n_show, dtype=int).tolist()

    # Use fixed layout based on first graph
    if fixed_layout:
        pos = nx.spring_layout(graphs[0], seed=42)

    # Plot individual timepoints
    n_cols = 3
    n_rows = (len(timepoints_to_show) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for idx, t in enumerate(timepoints_to_show):
        if idx >= len(axes):
            break

        G = graphs[t]
        ax = axes[idx]

        # Get node colors from features
        node_colors = [G.nodes[n]['feature'] for n in G.nodes()]

        # Use per-graph layout if not fixed
        if not fixed_layout:
            pos = nx.spring_layout(G, seed=42)

        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500,
                              cmap='viridis', ax=ax, vmin=min(node_colors), vmax=max(node_colors))
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.5, width=2)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

        ax.set_title(f'Timepoint {t} ({len(G.edges())} edges)', fontsize=12)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(len(timepoints_to_show), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path / f'graphs_over_time_traj_{trajectory_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved graph visualizations to {save_dir}/")


def visualize_graph_statistics(all_graphs, trajectory_idx=0, save_dir="visualizations_toy_data"):
    """
    Visualize graph structure statistics over time.

    Args:
        all_graphs: List of graph lists, one per trajectory
        trajectory_idx: Which trajectory to visualize
        save_dir: Directory to save visualizations
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    graphs = all_graphs[trajectory_idx]
    n_timepoints = len(graphs)

    # Compute statistics
    n_edges = [len(G.edges()) for G in graphs]
    avg_degree = [sum(dict(G.degree()).values()) / len(G.nodes()) for G in graphs]
    density = [nx.density(G) for G in graphs]

    # Compute graph changes between consecutive timepoints
    edge_changes = []
    for t in range(1, n_timepoints):
        edges_prev = set(graphs[t-1].edges())
        edges_curr = set(graphs[t].edges())
        # Count symmetric changes (undirected edges)
        added = len(edges_curr - edges_prev)
        removed = len(edges_prev - edges_curr)
        edge_changes.append(added + removed)

    # Plot statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Number of edges
    axes[0, 0].plot(range(n_timepoints), n_edges, 'b-', linewidth=2, marker='o')
    axes[0, 0].set_xlabel('Timepoint', fontsize=11)
    axes[0, 0].set_ylabel('Number of Edges', fontsize=11)
    axes[0, 0].set_title('Graph Size Over Time', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)

    # Average degree
    axes[0, 1].plot(range(n_timepoints), avg_degree, 'r-', linewidth=2, marker='s')
    axes[0, 1].set_xlabel('Timepoint', fontsize=11)
    axes[0, 1].set_ylabel('Average Degree', fontsize=11)
    axes[0, 1].set_title('Average Node Degree Over Time', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    # Density
    axes[1, 0].plot(range(n_timepoints), density, 'g-', linewidth=2, marker='^')
    axes[1, 0].set_xlabel('Timepoint', fontsize=11)
    axes[1, 0].set_ylabel('Density', fontsize=11)
    axes[1, 0].set_title('Graph Density Over Time', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

    # Edge changes
    axes[1, 1].bar(range(1, n_timepoints), edge_changes, color='orange', alpha=0.7)
    axes[1, 1].set_xlabel('Timepoint', fontsize=11)
    axes[1, 1].set_ylabel('Number of Edge Changes', fontsize=11)
    axes[1, 1].set_title('Graph Structural Changes Over Time', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path / f'graph_statistics_traj_{trajectory_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f"\nGraph Statistics for Trajectory {trajectory_idx}:")
    print(f"  Number of nodes: {len(graphs[0].nodes())}")
    print(f"  Average edges: {np.mean(n_edges):.1f} (±{np.std(n_edges):.1f})")
    print(f"  Average degree: {np.mean(avg_degree):.2f} (±{np.std(avg_degree):.2f})")
    print(f"  Average density: {np.mean(density):.3f} (±{np.std(density):.3f})")
    print(f"  Average edge changes per timestep: {np.mean(edge_changes):.1f}")

    print(f"\nSaved graph statistics to {save_dir}/")


def main():

    # Configuration
    node_features_path = "data/toy_data/all_node_features.npy"
    graphs_data_path =  "data/toy_data/all_graphs.pkl"
    save_dir = "visualizations_toy_data"

    # Load saved toy data
    print(f"\nLoading toy data from {node_features_path}...")
    all_node_features = np.load(node_features_path, allow_pickle=True)

    with open(os.path.join(graphs_data_path), 'rb') as f:
        all_graphs = pickle.load(f)
    
    n_trajectories = len(all_node_features)

    for traj_idx in range(n_trajectories):
        print(f"\n{'='*50}")
        print(f"Trajectory {traj_idx}:")
        print(f"  Node features shape: {all_node_features[traj_idx].shape}")
        print(f"  Number of graphs: {len(all_graphs[traj_idx])}")

        # Visualize node features
        print(f"\nVisualizing node features for trajectory {traj_idx}...")
        visualize_node_features(all_node_features, trajectory_idx=traj_idx, save_dir=save_dir)

        # Visualize graphs
        print(f"Visualizing graphs for trajectory {traj_idx}...")
        visualize_graphs(all_graphs, trajectory_idx=traj_idx, save_dir=save_dir,
                        timepoints_to_show=None, fixed_layout=True)

        # Visualize graph statistics
        print(f"Visualizing graph statistics for trajectory {traj_idx}...")
        visualize_graph_statistics(all_graphs, trajectory_idx=traj_idx, save_dir=save_dir)

    print(f"\n{'='*50}")
    print(f"\nAll visualizations saved to {save_dir}/")
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
import os

from ritini.utils.attention_graphs import adjacency_to_edge_index, attention_to_adjacency

def extract_attention_over_time(model, node_features, prior_adjacency, device, 
                                 history_length=5, dt=0.1):
    """
    Run inference on the trajectory and extract attention weights at each timestep.
    
    Args:
        model: Trained RiTINI model
        node_features: Tensor of shape (n_timepoints, n_genes)
        prior_adjacency: Prior adjacency matrix (n_genes, n_genes)
        device: torch device
        history_length: Length of history window (must match training)
        dt: Time step size (must match training)
        
    Returns:
        attention_history: List of attention weight tensors for each timestep
        predictions: List of predicted trajectories for each timestep
        edge_index_history: List of edge indices for each timestep
    """
    
    model.eval()
    n_timepoints, n_genes = node_features.shape
    
    # Convert prior adjacency to edge_index format
    edge_index = adjacency_to_edge_index(prior_adjacency).to(device)
    
    # Time evaluation points (matching training)
    t_eval = torch.arange(1, history_length, device=device) * dt
    
    attention_history = []
    predictions = []
    edge_index_history = []
    
    with torch.no_grad():
        # Slide through the trajectory with history windows
        for t in tqdm(range(n_timepoints - history_length), desc="Extracting attention"):
            # Get history window: shape (history_length, n_genes)
            history_window = node_features[t:t + history_length]
            
            # Reshape to match model input: (n_genes, history_length, 1)
            # Training does: x_history = history[b].T.unsqueeze(-1)
            # history[b] is (history_length, n_genes), so .T is (n_genes, history_length)
            x_history = history_window.T.unsqueeze(-1).to(device)  # (n_genes, history_length, 1)
            
            # Forward pass to get prediction and attention
            pred_traj, attention_output = model(x_history, edge_index, t_eval)
            
            # Store predictions
            predictions.append(pred_traj.cpu())
            
            # Store attention
            if attention_output is not None:
                edge_idx_attn, attn_weights = attention_output
                if attn_weights is not None:
                    attention_history.append(attn_weights.cpu())
                    edge_index_history.append(edge_idx_attn.cpu())
                else:
                    attention_history.append(None)
                    edge_index_history.append(edge_index.cpu())
            else:
                attention_history.append(None)
                edge_index_history.append(edge_index.cpu())
    
    return attention_history, predictions, edge_index_history

def visualize_single_graph(ax, attention_matrix, gene_names, title, 
                          pos=None, threshold=0.01, node_size=500,
                          active_nodes=None):
    """
    Visualize a single temporal graph with attention weights as edge colors.
    
    Args:
        ax: matplotlib axis
        attention_matrix: (n_genes, n_genes) attention weights
        gene_names: list of gene names
        title: plot title
        pos: node positions dict {node_idx: (x, y)} (if None, computed using spring layout)
        threshold: minimum attention weight to display edge
        node_size: size of nodes
        active_nodes: set of node indices to display (if None, shows only nodes with edges above threshold)
    """
    n_genes = attention_matrix.shape[0]
    
    # Find edges above threshold
    edges = []
    weights = []
    nodes_with_edges = set()
    
    for i in range(n_genes):
        for j in range(n_genes):
            if attention_matrix[i, j] > threshold:
                edges.append((i, j))
                weights.append(attention_matrix[i, j])
                nodes_with_edges.add(i)
                nodes_with_edges.add(j)
    # Determine which nodes to show
    if active_nodes is None:
        active_nodes = nodes_with_edges
    else:
        # Only keep nodes that are in both active_nodes and have edges
        active_nodes = active_nodes & nodes_with_edges
    
    # Filter edges to only include active nodes
    filtered_edges = []
    filtered_weights = []
    for (i, j), w in zip(edges, weights):
        if i in active_nodes and j in active_nodes:
            filtered_edges.append((i, j))
            filtered_weights.append(w)
    
    # Create directed graph with only active nodes
    G = nx.DiGraph()
    G.add_nodes_from(active_nodes)
    G.add_edges_from(filtered_edges)
    
    # Compute layout if not provided
    if pos is None:
        if len(active_nodes) > 0:
            pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(len(active_nodes)))
        else:
            pos = {}
    else:
        # Filter positions to only active nodes
        pos = {node: pos[node] for node in active_nodes if node in pos}
    
    # Handle empty graph
    if len(active_nodes) == 0:
        ax.text(0.5, 0.5, f'{title}\nNo edges above threshold', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        return pos
    
    # Normalize weights for color mapping
    if len(filtered_weights) > 0:
        weights_arr = np.array(filtered_weights)
        norm_weights = (weights_arr - weights_arr.min()) / (weights_arr.max() - weights_arr.min() + 1e-8)
    else:
        norm_weights = []
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, 
                          node_color='lightblue', edgecolors='black', linewidths=1.5)
    
    # Draw edges with colors based on attention
    if len(filtered_edges) > 0:
        edge_colors = [plt.cm.Reds(w) for w in norm_weights]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=filtered_edges, 
                              edge_color=edge_colors, width=2,
                              arrows=True, arrowsize=15, 
                              connectionstyle="arc3,rad=0.1",
                              alpha=0.8)
    
    # Draw labels
    labels = {i: gene_names[i] if i < len(gene_names) else str(i) for i in active_nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    return pos


def visualize_temporal_graphs(attention_history, edge_index_history, gene_names, 
                             n_genes, output_dir, threshold=0.01,
                             figsize=(8, 8), node_size=500):
    """
    Create individual visualizations of temporal graphs, showing only nodes with edges.
    
    Args:
        attention_history: List of attention weight tensors
        edge_index_history: List of edge indices
        gene_names: List of gene names
        n_genes: Number of genes
        output_dir: Directory to save figures
        threshold: Minimum attention weight to show edge
        figsize: Figure size for each plot
        node_size: Size of nodes in visualization
    """
    # Create output subdirectory for temporal graphs
    graphs_dir = os.path.join(output_dir, 'temporal_graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    
    n_timepoints = len(attention_history)
    
    # Convert all attention tensors to adjacency matrices
    adjacency_matrices = []
    for attn, edge_idx in zip(attention_history, edge_index_history):
        if attn is not None:
            adj = attention_to_adjacency(attn, edge_idx, n_genes,threshold=threshold)
            adj_np = adj.cpu().numpy() if hasattr(adj, 'cpu') else np.array(adj)
        else:
            adj_np = np.zeros((n_genes, n_genes))
        adjacency_matrices.append(adj_np)
    
    # Find all nodes that have edges above threshold in any timepoint
    all_active_nodes = set()
    for adj_np in adjacency_matrices:
        rows, cols = np.where(adj_np > threshold)
        all_active_nodes.update(rows)
        all_active_nodes.update(cols)
    
    if not all_active_nodes:
        print("No edges found above threshold across all timepoints.")
        return
    
    # Compute consistent layout using average attention
    avg_attention = np.mean(adjacency_matrices, axis=0)
    
    G_layout = nx.DiGraph()
    G_layout.add_nodes_from(all_active_nodes)
    for i in all_active_nodes:
        for j in all_active_nodes:
            if avg_attention[i, j] > 0:
                G_layout.add_edge(i, j)
    
    pos = nx.spring_layout(G_layout, seed=42, k=2/np.sqrt(len(all_active_nodes)))
    
    # Plot each timepoint
    for t, adj_np in enumerate(adjacency_matrices):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Count edges for title
        n_edges = np.sum(adj_np > threshold)
        rows, cols = np.where(adj_np > threshold)
        n_active = len(set(rows) | set(cols))
        
        visualize_single_graph(
            ax=ax,
            attention_matrix=adj_np,
            gene_names=gene_names,
            title=f'Time {t} ({n_active} nodes, {n_edges} edges)',
            pos=pos,
            threshold=threshold,
            node_size=node_size
        )
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f'graph_t{t:04d}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {n_timepoints} temporal graph plots to {graphs_dir}")

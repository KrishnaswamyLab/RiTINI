import torch

def adjacency_to_edge_index(adjacency):
    """
    Convert adjacency matrix to PyTorch Geometric edge_index format.

    Args:
        adjacency: (n_nodes, n_nodes) adjacency matrix

    Returns:
        edge_index: (2, num_edges) tensor in COO format
    """

    edge_index = adjacency.nonzero().t().contiguous()
    return edge_index


def attention_to_adjacency(attention_weights, edge_index, n_nodes, threshold=0.5):
    """
    Convert attention weights back to adjacency matrix.

    Args:
        attention_weights: (num_edges, 1) or (num_edges,) attention weights
        edge_index: (2, num_edges) edge indices
        n_nodes: Number of nodes
        threshold: Threshold for creating edges

    Returns:
        adjacency: (n_nodes, n_nodes) adjacency matrix
    """
    # Ensure attention_weights is 1D
    if attention_weights.dim() > 1:
        attention_weights = attention_weights.squeeze(-1)  # (num_edges, 1) -> (num_edges,)

    # Create adjacency matrix
    adjacency = torch.zeros(n_nodes, n_nodes, device=attention_weights.device)

    # Assign attention weights to adjacency matrix
    src = edge_index[0].long()
    dst = edge_index[1].long()
    adjacency[src, dst] = attention_weights

    # # Apply threshold to make it binary
    # TODO: verify if this is necessary: In the case we are regularizing against the previous graph.
    # adjacency = (adjacency > threshold).float()

    return adjacency
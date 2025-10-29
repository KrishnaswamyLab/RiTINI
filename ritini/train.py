import torch

from ritini.utils.attention_graphs import adjacency_to_edge_index, attention_to_adjacency

def train_epoch(model, dataloader, optimizer, criterion, device, n_genes, prior_adjacency,graph_reg_weight=0.1):
    """
    Train for one epoch.

    Args:
        model: RiTINI model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        n_genes: Number of genes/nodes
        prior_adjacency: Initial adjacency matrix at t=0

    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    total_feature_loss = 0
    total_graph_loss = 0
    n_samples = 0

    for batch in dataloader:

        node_features = batch['node_features'].to(device)  # (batch, time_window, n_genes)

        batch_size, time_window, n_genes = node_features.shape

        # Process each sequence in the batch
        batch_loss = 0
        batch_feature_loss = 0
        batch_graph_loss = 0

        # We need to start the first edge_index with the prior graph adjacency for t=0
        current_adj = prior_adjacency.to(device)

        for b in range(batch_size):

            # Iterate through time sequence
            for t in range(time_window - 1):

                # Current timepoint features
                x_t = node_features[b, t]  # (n_genes,)

                # Reshape to (n_nodes, n_features) where n_nodes=n_genes, n_features=1
                x_t = x_t.unsqueeze(-1)  # (n_genes, 1)

                # Convert adjacency to edge_index
                edge_index = adjacency_to_edge_index(current_adj)

                # Forward pass
                pred_features, (edge_index_attention, attn_weights) = model(x_t, edge_index)
                
                # Reshape prediction back to (n_genes,)
                pred_features = pred_features.squeeze(-1)

                # Target is next timepoint
                target_features = node_features[b, t + 1]

                # Compute loss (feature prediction)
                feature_loss = criterion(pred_features, target_features)
                batch_feature_loss += feature_loss

                current_adj = attention_to_adjacency(attn_weights, edge_index_attention, n_genes)

                # Compute graph regularization loss (penalize deviation from prior)
                # Using L1 norm 
                graph_loss = torch.sum(torch.abs(current_adj - prior_adjacency))
                batch_graph_loss += graph_loss

        # Average losses over batch and time
        batch_feature_loss = batch_feature_loss / (batch_size * (time_window - 1))
        batch_graph_loss = batch_graph_loss / (batch_size * (time_window - 1))

        # Combined loss with regularization
        batch_loss = batch_feature_loss + graph_reg_weight * batch_graph_loss

        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        total_feature_loss += batch_feature_loss.item()
        total_graph_loss += batch_graph_loss.item()
        n_samples += 1

    avg_loss = total_loss / n_samples
    avg_feature_loss = total_feature_loss / n_samples
    avg_graph_loss = total_graph_loss / n_samples
    
    return avg_loss,avg_feature_loss,avg_graph_loss
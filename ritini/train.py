import torch
from ritini.utils.attention_graphs import adjacency_to_edge_index, attention_to_adjacency

def train_epoch(model, dataloader, optimizer, criterion, device, n_genes, prior_adjacency, graph_reg_weight=0.0, sparsity_weight=0.0):
    """
    Train for one epoch with multi-step prediction.
    Returns: avg_loss, avg_feature_loss, avg_graph_loss, avg_sparsity_loss
    """
    model.train()
    total_loss = 0
    total_feature_loss = 0
    total_graph_loss = 0
    total_sparsity_loss = 0
    n_samples = 0
    
    # Ensure prior_adjacency is binary (0s and 1s) for BCE loss
    assert torch.all((prior_adjacency == 0) | (prior_adjacency == 1)), "prior_adjacency must be binary"
    
    # Remove self-loops from prior once (set diagonal to 0)
    prior_adjacency = prior_adjacency.clone()
    prior_adjacency.fill_diagonal_(0)
    
    edge_index = adjacency_to_edge_index(prior_adjacency).to(device)
    
    dt = 0.1
    time_window = 5
    t_eval = torch.arange(1, time_window, device=device) * dt
    for batch in dataloader:
        optimizer.zero_grad()
        
        history = batch['history'].to(device)
        node_features = batch['node_features'].to(device)
        batch_size = history.shape[0]
        
        batch_feature_loss = 0
        batch_graph_loss = 0
        batch_sparsity_loss = 0
        
        for b in range(batch_size):
            x_history = history[b].T.unsqueeze(-1)
            
            pred_traj, attention_output = model(x_history, edge_index, t_eval)
            pred_traj = pred_traj.squeeze(-1)
            targets = node_features[b, 1:]
            
            # Feature prediction loss
            feature_loss = criterion(pred_traj, targets)
            batch_feature_loss += feature_loss
            
            # Graph regularization loss
            edge_index_attn, attn_weights = attention_output
            
            # GAT normalizes over incoming edges (targets), producing valid probabilities
            # Clamp to [0, 1] to handle rare numerical precision issues (typically <0.1% of values)
            clamped_attn = torch.clamp(attn_weights, 0.0, 1.0)
            
            current_adj = attention_to_adjacency(clamped_attn, edge_index_attn, n_genes)
            
            # Verify no self-loops exist (diagonal should be 0)
            # Note: We don't zero it here as that would break softmax normalization if self-loops existed
            # Instead, we prevent self-loops at the source (prior adjacency + add_self_loops=False)
            assert torch.allclose(torch.diag(current_adj), torch.zeros(n_genes, device=current_adj.device)), \
                "Self-loops detected in attention! This should not happen."
            
            # Binary cross-entropy: current_adj has probabilities (post-softmax), prior is binary
            # Normalize by number of possible edges (n_genes^2) for scale-invariance
            bce_loss = torch.nn.functional.binary_cross_entropy(current_adj, prior_adjacency, reduction='mean')
            batch_graph_loss += bce_loss
            
            # Entropy Sparsity loss (minimizing entropy makes attention spiky/sparse)
            # Since attention weights are probabilities via softmax, L1 is constant (1.0 per node).
            # Entropy provides the correct gradient for sparsity on distributions.
            eps = 1e-8
            entropy_loss = -torch.mean(clamped_attn * torch.log(clamped_attn + eps))
            batch_sparsity_loss += entropy_loss
        
        # Average over batch
        batch_feature_loss = batch_feature_loss / batch_size
        batch_graph_loss = batch_graph_loss / batch_size
        batch_sparsity_loss = batch_sparsity_loss / batch_size
        
        # Combined loss
        batch_loss = batch_feature_loss + graph_reg_weight * batch_graph_loss + sparsity_weight * batch_sparsity_loss
        
        batch_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += batch_loss.item()
        total_feature_loss += batch_feature_loss.item()
        total_graph_loss += batch_graph_loss.item()
        total_sparsity_loss += batch_sparsity_loss.item()
        n_samples += 1
    
    avg_loss = total_loss / n_samples
    avg_feature_loss = total_feature_loss / n_samples
    avg_graph_loss = total_graph_loss / n_samples
    avg_sparsity_loss = total_sparsity_loss / n_samples
    
    return avg_loss, avg_feature_loss, avg_graph_loss, avg_sparsity_loss
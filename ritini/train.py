import torch

from ritini.utils.attention_graphs import adjacency_to_edge_index

def train_epoch(model, dataloader, optimizer, criterion, device, n_genes, prior_adjacency, graph_reg_weight=0.0):
    """
    Train for one epoch with multi-step prediction.
    Returns: avg_loss, avg_feature_loss, avg_graph_loss
    """
    model.train()
    total_loss = 0
    total_feature_loss = 0
    total_graph_loss = 0
    n_samples = 0
    
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
        
        for b in range(batch_size):
            x_history = history[b].T.unsqueeze(-1)
            
            pred_traj, attention_output = model(x_history, edge_index, t_eval)
            pred_traj = pred_traj.squeeze(-1)
            targets = node_features[b, 1:]
            
            # Feature prediction loss
            feature_loss = criterion(pred_traj, targets)
            batch_feature_loss += feature_loss
            
            # Graph regularization loss (optional)
            if graph_reg_weight > 0 and attention_output is not None:
                edge_index_attn, attn_weights = attention_output
                from ritini.utils.attention_graphs import attention_to_adjacency
                current_adj = attention_to_adjacency(attn_weights, edge_index_attn, n_genes)
                graph_loss = torch.sum(torch.abs(current_adj - prior_adjacency))
                batch_graph_loss += graph_loss
        
        # Average over batch
        batch_feature_loss = batch_feature_loss / batch_size
        batch_graph_loss = batch_graph_loss / batch_size if graph_reg_weight > 0 else torch.tensor(0.0)
        
        # Combined loss
        batch_loss = batch_feature_loss + graph_reg_weight * batch_graph_loss
        
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += batch_loss.item()
        total_feature_loss += batch_feature_loss.item()
        total_graph_loss += batch_graph_loss.item() if isinstance(batch_graph_loss, torch.Tensor) else 0
        n_samples += 1
    
    avg_loss = total_loss / n_samples
    avg_feature_loss = total_feature_loss / n_samples
    avg_graph_loss = total_graph_loss / n_samples
    
    return avg_loss, avg_feature_loss, avg_graph_loss
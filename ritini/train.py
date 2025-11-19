import torch

from ritini.utils.attention_graphs import adjacency_to_edge_index

def train_epoch(model, dataloader, optimizer, criterion, device, n_genes, prior_adjacency, graph_reg_weight=0.1):
    """
    Train for one epoch with multi-step prediction.
    """
    model.train()
    total_loss = 0
    n_samples = 0
    
    edge_index = adjacency_to_edge_index(prior_adjacency).to(device)
    
    dt = 0.1
    time_window = 5
    t_eval = torch.arange(1, time_window, device=device) * dt
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        history = batch['history'].to(device)  # (batch, history_length, n_genes)
        node_features = batch['node_features'].to(device)  # (batch, time_window, n_genes)
        batch_size = history.shape[0]
        
        batch_loss = 0
        
        for b in range(batch_size):
            # History for all nodes
            x_history = history[b].T.unsqueeze(-1)  # (n_genes, history_length, 1)
            
            # Forward pass
            pred_traj, attention_output = model(x_history, edge_index, t_eval)
            pred_traj = pred_traj.squeeze(-1)  # (time_window-1, n_genes)
            
            # Ground truth for future timesteps
            targets = node_features[b, 1:]  # (time_window-1, n_genes)
            
            # Compute loss
            loss = criterion(pred_traj, targets)
            batch_loss += loss
        
        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += batch_loss.item()
        n_samples += 1
    
    avg_loss = total_loss / n_samples
    return avg_loss
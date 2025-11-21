import torch
from torch.utils.data import Dataset

class TemporalGraphDataset(Dataset):
    """
    Dataset for temporal graph sequences with history.
    Returns historical node features + future prediction window.
    """
    def __init__(self, node_features, time_window=5, history_length=5):
        """
        Args:
            node_features: np.ndarray or tensor of shape (n_timepoints, n_genes)
            time_window: int, length of prediction window
            history_length: int, number of past timesteps to use as context
        """
        self.node_features = torch.FloatTensor(node_features)  # (T, N)
        self.n_timepoints = node_features.shape[0]
        self.n_nodes = node_features.shape[1]
        self.time_window = time_window
        self.history_length = history_length
        
    def __len__(self):
        # Need enough data for history + prediction window
        # Start from history_length, predict until end
        return self.n_timepoints - self.history_length - self.time_window + 1
    
    def __getitem__(self, idx):
        """
        Returns historical sequence + prediction window.
        """
        # Actual start index (offset by history_length)
        start_idx = idx + self.history_length
        
        # History: [start_idx - history_length : start_idx]
        history = self.node_features[start_idx - self.history_length : start_idx]
        
        # Current + future window: [start_idx : start_idx + time_window]
        window = self.node_features[start_idx : start_idx + self.time_window]
        
        return {
            'history': history,  # (history_length, n_nodes)
            'node_features': window,  # (time_window, n_nodes)
            'start_idx': start_idx,
            'timepoints': torch.arange(start_idx, start_idx + self.time_window)
        }
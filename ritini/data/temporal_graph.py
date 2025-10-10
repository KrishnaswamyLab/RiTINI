import torch
from torch.utils.data import Dataset

class TemporalGraphDataset(Dataset):
    """
    Dataset for temporal graph sequences.
    Returns node features for each timepoint.
    """
    def __init__(self, node_features, time_window=None):
        """
        Args:
            node_features: np.ndarray of shape (n_timepoints, n_genes)
            time_window: int, if None uses all timepoints as one sequence
        """
        self.node_features = torch.FloatTensor(node_features)  # (T, N)
        self.n_timepoints = node_features.shape[0]
        self.n_nodes = node_features.shape[1]

        self.time_window = time_window if time_window else self.n_timepoints

    def __len__(self):
        # Number of sequences we can extract
        if self.time_window >= self.n_timepoints:
            return 1
        return self.n_timepoints - self.time_window + 1

    def __getitem__(self, idx):
        """
        Returns a temporal sequence of node features.
        """
        if self.time_window >= self.n_timepoints:
            return {
                'node_features': self.node_features,  # (T, N)
                'timepoints': torch.arange(self.n_timepoints)
            }

        end_idx = idx + self.time_window
        return {
            'node_features': self.node_features[idx:end_idx],  # (seq_len, N)
            'timepoints': torch.arange(idx, end_idx)
        }
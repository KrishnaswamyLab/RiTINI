import torch
import torch.nn as nn


class TimeAttention(nn.Module):
    """
    Time-attention module that computes attention weights across a time window
    and produces a weighted sum of the time-series node features.

    The attention scores are computed by a small feed-forward network operating on
    per-time summaries (mean across nodes) and then softmaxed across the time axis
    to produce l_delta weights as in the paper diagram.
    TODO: I'm not sure if the shape of x is handled correctly here?
    """

    def __init__(self, in_features, hidden=32):
        super(TimeAttention, self).__init__()
        self.in_features = in_features

        self.scorer = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (time_window T, number of nodes N)
        Returns:
            aggregated: Tensor (number of nodes N, F)
            weights: Tensor (T,) attention weights across time (softmaxed)
        """
        T, N, F = x.shape

        if F != self.in_features:
            proj = nn.Linear(F, self.in_features).to(x.device)
            x = proj(x)
            _, _, F = x.shape

        # Per-time summary: mean over nodes (T, F)
        time_summary = x.mean(dim=1)  # (T, F)

        # Compute scores per timepoint (T, 1)
        scores = self.scorer(time_summary).squeeze(-1)

        # Softmax across time axis to obtain l_delta
        weights = torch.softmax(scores, dim=0)  # (T,)

        # Weighted sum across time
        w = weights.view(T, 1, 1)
        aggregated = (w * x).sum(dim=0)

        return aggregated, weights

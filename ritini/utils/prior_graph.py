from typing import Optional, Union, Any
import pandas as pd

import numpy as np
import torch
from statsmodels.tsa.stattools import grangercausalitytests
def compute_prior_adjacency(
    time_series,
    mode = 'granger_causality',
    lag_order= 1,
    test= 'ssr_chi2test',
    neg_log_threshold= 5.0,
    p_value_threshold= None,
    edge_weighted= False,
    directed= True,
    fully_connected_self_loops= False,
    eps= 1e-10,
):
    """Compute a prior adjacency matrix.

    Args:
        time_series: (T, N) array
        mode:{'granger_causality', 'fully_connected', 'identity', 'zeros'}.
        lag_order: lag order for Granger causality (only for granger_causality).
        test: which statsmodels test key to use (default 'ssr_chi2test').
        neg_log_threshold: threshold on -log(p).
        p_value_threshold: threshold p-values directly (overrides neg_log).
        edge_weighted: if True and mode=='granger_causality', return weighted adjacency using -log(p). Otherwise return binary adjacency.
        directed: whether returned adjacency is directed. If False, symmetrize.
        fully_connected_self_loops: if True, keep diagonal ones for fully_connected.
        eps: small value added before taking log to avoid -inf.

    Returns:
        adj: torch.FloatTensor of shape (N, N) where adj[i, j] indicates edge i -> j.
    """

    T, N = time_series.shape

    if mode == 'fully_connected':
        adj = np.ones((N, N), dtype=float)
        if not fully_connected_self_loops:
            np.fill_diagonal(adj, 0.0)
        return torch.from_numpy(adj.astype(np.float32))

    if mode == 'identity':
        adj = np.eye(N, dtype=float)
        return torch.from_numpy(adj.astype(np.float32))

    if mode == 'zeros':
        adj = np.zeros((N, N), dtype=float)
        return torch.from_numpy(adj.astype(np.float32))

    if mode == 'granger_causality':

        # Prepare adjacency / weight matrix
        pvals = np.ones((N, N), dtype=float) * 1.0
        neglog = np.zeros((N, N), dtype=float)

        # For each target i and predictor j, test whether j -> i (j causes i)
        for i in range(N):
            xi = time_series[:, i]
            for j in range(N):
                if i == j:
                    pvals[j, i] = 1.0
                    neglog[j, i] = 0.0
                    continue

                xj = time_series[:, j]
                data_2col = np.vstack([xi, xj]).T
                try:
                    res = grangercausalitytests(data_2col, maxlag=lag_order, verbose=False)
                    # collect p-values across lags and take min
                    p_list = []
                    for lag in range(1, lag_order + 1):
                        try:
                            pv = res[lag][0][test][1]
                            p_list.append(float(pv))
                        except Exception:
                            # fallback: try other keys or skip
                            p_list.append(1.0)
                    p_min = min(p_list) if len(p_list) > 0 else 1.0
                except Exception:
                    p_min = 1.0

                pvals[j, i] = p_min
                neglog[j, i] = -np.log(p_min + eps)

        if p_value_threshold is not None:
            adj = (pvals <= float(p_value_threshold)).astype(float)
        elif neg_log_threshold is not None:
            adj = (neglog >= float(neg_log_threshold)).astype(float)
        else:
            # default: return weighted neg-log values or binary using typical 0.05
            if edge_weighted:
                adj = neglog
            else:
                adj = (pvals <= 0.05).astype(float)

        if edge_weighted and not (p_value_threshold is not None or neg_log_threshold is not None):
            # return weights (neglog)
            adj = neglog

        # Symmetrize if undirected requested
        if not directed:
            adj = np.maximum(adj, adj.T)

        return torch.from_numpy(adj.astype(np.float32))


from typing import Optional, Union, Any
import pandas as pd

import numpy as np
import torch
from statsmodels.tsa.stattools import grangercausalitytests
def compute_prior_adjacency(
    time_series,
    mode: str = 'granger_causality',
    lag_order: int = 1,
    test: str = 'ssr_chi2test',
    neg_log_threshold: float = 5.0,
    p_value_threshold: Optional[float] = None,
    edge_weighted: bool = False,
    directed: bool = True,
    fully_connected_self_loops: bool = False,
    eps: float = 1e-10,
    gene_names: Optional[Union[list, np.ndarray]] = None,
    db_extract_file: Optional[str] = None,
    db_column: str = 'HGNC symbol',
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

        # If a DB extract is provided, build a set of allowed identifiers to filter tests.
        # Use a helper to extract identifiers inside parentheses (e.g. 'GENE (ENSG...)' -> 'ENSG...')
        def _extract_id(name: str) -> str:
            s = str(name)
            if '(' in s and s.endswith(')'):
                return s.split('(')[-1].replace(')', '').strip()
            return s.strip()

        db_allowed = None
        if db_extract_file is not None:
            try:
                db_df = pd.read_csv(db_extract_file)
                if db_column in db_df.columns:
                    db_allowed = set(db_df[db_column].astype(str).str.strip())
                else:
                    db_allowed = set(db_df.iloc[:, 0].astype(str).str.strip())
            except Exception:
                print("Warning: could not read DB extract file; proceeding without DB filtering.")
                db_allowed = None

        # Normalize gene names list and precompute mask of which genes appear in the DB
        if gene_names is None:
            gene_names = [str(i) for i in range(N)]
        else:
            gene_names = [str(g) for g in gene_names]

        if db_allowed is not None:
            gene_ids = set(_extract_id(g) for g in gene_names)
            in_db_ids = gene_ids & db_allowed
            in_db_mask = [_extract_id(g) in in_db_ids for g in gene_names]
        else:
            in_db_mask = [True] * len(gene_names)

        # For each target i and predictor j, test whether j causes i
        for i in range(N):
            xi = time_series[:, i]
            name_i = gene_names[i] if i < len(gene_names) else str(i)
            i_in_db = in_db_mask[i] if i < len(in_db_mask) else True

            for j in range(N):
                if i == j:
                    pvals[j, i] = 1.0
                    neglog[j, i] = 0.0
                    continue

                name_j = gene_names[j] if j < len(gene_names) else str(j)
                j_in_db = in_db_mask[j] if j < len(in_db_mask) else True

                # If either gene is not present in the DB, skip testing (leave p=1)
                if not (i_in_db and j_in_db):
                    print(f"Skipping test for genes {name_i} and {name_j} as one or both are not in the DB.")
                    p_min = 1.0
                else:
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


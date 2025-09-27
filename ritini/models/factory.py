"""
Model factory for creating and loading neural network models.
"""

import logging
from typing import Dict, Any, Union
from pathlib import Path
import torch
import torch.nn as nn
from ..utils.data_processing import *
from .gatConvWithAttention import GATConvWithAttention
from .gde import *
from .odeblock import *
from .pyg import *

logger = logging.getLogger(__name__)


def get_model(model_type: str, config: Dict[str, Any]):
    """
    Factory function to create model instances.

    Args:
        model_type: Type of model ('gcn', 'gat', 'gde')
        config: Configuration dictionary

    Returns:
        Model instance
    """
    logger.info(f"Creating model of type: {model_type}")

    # TODO: MAJOR TODO HERE. NEED TO RETRIEVE THE MODELS PROPERLY
    if model_type == 'default':
        gnn = PygSequential(
            GATConvWithAttention(
                in_feats=config['in_feats'],
                out_feats=config['out_feats'],
                num_heads=1,
                residual=False,
                activation=nn.Tanh(),
                feat_drop=0.0, 
                attn_drop=0.0,
            ),
            MeanAttentionLayer(),
        )
        gdefunc = GDEFunc(gnn)
        gde = ODEBlock(func=gdefunc, method='rk4', atol=1e-3, rtol=1e-4, adjoint=False).to(config['device'])
        return gde

    if model_type == 'gcn':
        from .gcn import GCN
        # TODO: Extract GCN class from gcn.py and instantiate
        logger.info("GCN model created")
        return None  # Placeholder

    elif model_type == 'gat':
        from .gat import GraphAttentionLayer
        # TODO: Extract GAT class from gat.py and instantiate
        logger.info("GAT model created")
        return None  # Placeholder

    elif model_type == 'gde':
        from .gde import GDE
        # TODO: Extract GDE class from gde.py and instantiate
        logger.info("GDE model created")
        return None  # Placeholder

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_data(data_path: Union[str, Path],
                top_genes: list,
                train_g: Any,
                n_cells_at_t: int,
                time_bins: list,
                num_cell_types: int,
                cell_types: list) -> Dict[str, Any]:
    """
    Load and prepare data for training.

    Args:
        data_path: Path to training data
        top_genes: List of top genes
        train_g: Training graph
        n_cells_at_t: Number of cells at each time point
        time_bins: Time bins for temporal data
        num_cell_types: Number of cell types
        cell_types: List of cell types

    Returns:
        Dictionary containing prepared data structures
    """
    logger.info("Loading and preparing training data...")

    try:
        # Load the main dataframe
        df_train = load_data(str(data_path))

        # Create node names and mapping
        nodes_names = [top_genes[i] for i in train_g.nodes().numpy()]
        node_map_full = {n: i for i, n in enumerate(nodes_names)}

        # Identify transcription factors (every 5th gene)
        tfs = top_genes[::5]

        # Prepare time-series data
        data_tis = []
        for _t, time_i in enumerate(time_bins[:-1]):
            t0 = time_bins[_t]
            t1 = time_bins[_t + 1]

            data_t0 = get_n_cells_of_all_types_at_time_t(df_train, n_cells_at_t, t0, genes=top_genes)
            data_t1 = get_n_cells_of_all_types_at_time_t(df_train, n_cells_at_t, t1, genes=top_genes)

            if _t == 0:
                data_tis.append(torch.Tensor(data_t0))
            data_tis.append(torch.Tensor(data_t1))

        # Store prepared data
        data = {
            'df_train': df_train,
            'top_genes': top_genes,
            'train_g': train_g,
            'nodes_names': nodes_names,
            'node_map_full': node_map_full,
            'tfs': tfs,
            'n_cells_at_t': n_cells_at_t,
            'time_bins': time_bins,
            'num_cell_types': num_cell_types,
            'cell_types': cell_types,
            'data_tis': data_tis
        }

        logger.info("Data loading completed successfully!")
        return data

    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise
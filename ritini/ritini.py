"""
Main RiTINI operator for regulatory temporal interaction network inference.
"""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn

from .models.pyg import *
from .models.gde import *
from .models.odeblock import *

from .models.factory import get_model, load_data

from .models.gatConvWithAttention import GATConvWithAttention
from .utils.plots import *
from .utils.data_processing import *
from .data_generation.sergio import *
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import StepLR


logger = logging.getLogger(__name__)


class RiTINI():
    """
    Main RiTINI operator for regulatory temporal interaction network inference.

    This class provides a simple interface for training models, inferring networks,
    generating synthetic data, and creating visualizations.

    Example:
        >>> import ritini
        >>> ri = ritini.RiTINI()
        >>> ri.train(data_path="data.csv", model_type="gcn", epochs=100)
        >>> networks = ri.infer(data_path="test_data.csv")
    """

    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize RiTINI operator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.trained = False
        self.data = None
        self.model = get_model(self.config['model_type'], self.config)

        # Setup logging
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def train(self,
              data_path: Union[str, Path] = None,
              data: Optional[Dict[str, Any]] = None,
              df_train: Optional[pd.DataFrame] = None,
              model_type: str = "gcn",
              epochs: int = 100,
              learning_rate: float = 0.001,
              weight_decay: float = 5e-4,
              step_size: int = 350,
              gamma: float = 0.1,
              output_dir: Optional[Union[str, Path]] = None,
              input_features: Optional[int] = None,
              output_features: Optional[int] = None,
              # Training loop specific parameters
              top_genes: Optional[list] = None,
              train_g: Optional[Any] = None,
              n_cells_at_t: Optional[int] = None,
              time_bins: Optional[list] = None,
              num_cell_types: Optional[int] = None,
              cell_types: Optional[list] = None,
              steps: int = 100,
              verbose_step: int = 1,
              link_step: int = 2,
              add_n: int = 5,
              del_n: int = 5,
              lambda_l1: float = 10.0,
              device: str = 'cpu',
              **kwargs) -> None:    
        """
        Train a neural ODE model for network inference.

        Args:
            data_path: Path to training data
            model_type: Type of model ('gcn', 'gat', 'gde')
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            output_dir: Directory to save trained model
            **kwargs: Additional model configuration parameters
        """
        logger.info(f"Starting training with model: {model_type}")

        # Update configuration
        self.config.update({
            'model_type': model_type,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'step_size': step_size,
            'gamma': gamma,
            'device': device,
            **kwargs
        })

        if data_path:
            self.config['data_path'] = str(data_path)

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self.config['output_path'] = str(output_path)

        try:
            # Initialize model
            logger.info(f"Initializing {model_type} model...")
            self.model = get_model(model_type, self.config)
            model = self.model.to(device)

            # Setup optimizer and scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            criterion = torch.nn.MSELoss()

            # Prepare data structures
            # PyTorch Geometric graph - use num_nodes to get node indices
            node_indices = list(range(train_g.num_nodes))
            edge_index = train_g.edge_index.to(device)

            nodes_names = [top_genes[i] for i in node_indices]
            node_map_full = {n: i for i, n in enumerate(nodes_names)}
            tfs = top_genes[::5]

            attentions = {}
            all_losses = []
            all_main_losses = []
            all_l1_losses = []

            # Training loop
            logger.info("Starting training...")
            for step_i in range(steps):
                data_tps = []
                data_tis = []

                for _t, time_i in enumerate(time_bins[:-1]):
                    t0 = time_bins[_t]
                    t1 = time_bins[_t + 1]

                    ## Here we are retrieving the X for the genes.
                    data_t0 = get_n_cells_of_all_types_at_time_t(df_train, n_cells_at_t, t0, genes=top_genes)
                    data_t1 = get_n_cells_of_all_types_at_time_t(df_train, n_cells_at_t, t1, genes=top_genes)

                    data_t0 = torch.Tensor(data_t0)
                    data_t1 = torch.Tensor(data_t1)

                    model.train()

                    # We need to change here to receive data_t0 and the edge_index
                    data_tp = model(
                        data_t0,
                        edge_index,
                        torch.Tensor([t0, t1]),
                        return_whole_sequence=False
                    )

                    _, attn = model.func.gnn.layers[0](data_t0,edge_index, get_attention=True)
                    main_loss = criterion(data_tp, data_t1)
                    l1_loss = torch.norm(attn, 1)

                    loss = main_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if _t == 0:
                        data_tis.append(data_t0.clone().detach())
                    data_tis.append(data_t1.clone().detach())
                    data_tps.append(data_tp.clone().detach())
                    all_losses.append(loss)
                    all_main_losses.append(main_loss)
                    all_l1_losses.append(l1_loss)

                avg_main_losses = torch.mean(torch.tensor(all_main_losses)).item()
                print("Dynamics Prediction loss:", avg_main_losses)
                scheduler.step()

                if step_i % verbose_step == 0:
                    model.eval()
                    # TODO: Implement test_loop integration
                    pass

            self.trained = True
            logger.info("Training completed successfully!")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def infer(self,
              data_path: Union[str, Path],
              model_path: Optional[Union[str, Path]] = None,
              threshold: float = 0.5,
              output_path: Optional[Union[str, Path]] = None) -> Any:
        """
        Infer regulatory networks from trained model.

        Args:
            data_path: Path to input data
            model_path: Path to trained model (if loading from file)
            threshold: Edge probability threshold
            output_path: Path to save inferred networks

        Returns:
            Inferred regulatory networks
        """
        logger.info("Starting network inference...")

        try:
            if model_path:
                # TODO: Load model from file
                logger.info(f"Loading model from: {model_path}")
            elif not self.trained:
                raise ValueError("No trained model available. Train a model first or provide model_path.")

            # Load data
            inference_data = load_data(str(data_path))

            # TODO: Implement inference logic
            networks = None  # Placeholder

            if output_path:
                # TODO: Save networks to file
                logger.info(f"Saving networks to: {output_path}")

            logger.info("Inference completed successfully!")
            return networks

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def simulate(self,
                 genes: int = 100,
                 cells: int = 1000,
                 bins: int = 10,
                 config_path: Optional[Union[str, Path]] = None,
                 output_path: Optional[Union[str, Path]] = None,
                 **kwargs) -> Any:
        """
        Generate synthetic data using SERGIO simulator.

        Args:
            genes: Number of genes
            cells: Number of cells
            bins: Number of cell types/bins
            config_path: Path to SERGIO configuration file
            output_path: Path to save simulated data
            **kwargs: Additional SERGIO parameters

        Returns:
            Simulated data
        """
        logger.info("Starting data simulation with SERGIO...")

        try:
            # TODO: Implement SERGIO simulation
            simulated_data = None  # Placeholder

            if output_path:
                # TODO: Save simulated data
                logger.info(f"Saving simulated data to: {output_path}")

            logger.info("Simulation completed successfully!")
            return simulated_data

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise

    def plot(self,
             data_path: Union[str, Path],
             plot_type: str = "network",
             output_path: Optional[Union[str, Path]] = None,
             **kwargs) -> None:
        """
        Generate visualizations.

        Args:
            data_path: Path to data file
            plot_type: Type of plot ('network', 'heatmap', 'trajectory')
            output_path: Path to save plot
            **kwargs: Additional plotting parameters
        """
        logger.info(f"Generating {plot_type} plot...")

        try:
            # TODO: Implement plotting logic

            if output_path:
                logger.info(f"Saving plot to: {output_path}")

            logger.info("Plot generated successfully!")

        except Exception as e:
            logger.error(f"Plotting failed: {e}")
            raise

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()

    def set_config(self, config: Dict[str, Any]) -> None:
        """Set configuration."""
        self.config.update(config)
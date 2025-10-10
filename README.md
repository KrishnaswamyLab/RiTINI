# RiTINI: Regulatory Temporal Interaction Network Inference

A Graph Neural Ordinary Differential Equation (ODE) framework for modeling and predicting gene expression trajectories over time. RiTINI combines Graph Attention Networks (GATs) with Neural ODEs to capture the continuous temporal dynamics of gene regulatory networks.

## Overview

RiTINI (Regulatory Temporal Interaction Network Inference) is designed to:
- Model temporal gene expression data using graph neural networks
- Learn attention-based gene regulatory networks from trajectory data
- Predict future gene expression states through continuous-time ODEs
- Visualize learned attention patterns and regulatory interactions

## Installation

### Requirements
- Python >= 3.10
- PyTorch >= 2.8.0
- PyTorch Geometric >= 2.6.1

### Setup

1. Clone the repository:
```bash
git clone git@github.com:KrishnaswamyLab/RiTINI.git
cd RiTINI
```

2. Install dependencies using uv (recommended):
```bash
pip install uv
uv sync
```

Or using pip:
```bash
pip install -e .
```

## Quick Start

### Training on Real Data

Run the main training script:

```bash
python main.py
```

This will:
1. Load trajectory data from `data/trajectories/traj_data.pkl`
2. Incorporate prior network information from `data/trajectories/cancer_granger_prior_graph_nx_20.pkl`
3. Train the RiTINI model for 200 epochs
4. Save the best model to `best_model.pt`
5. Save training history to `training_history.json`

### Training on Synthetic Data

```bash
python test_toy_data_ritini.py
```

### Using the Model

```python
import torch
from ritini.models.RiTINI import RiTINI

# Initialize model
model = RiTINI(
    in_features=1,          # Gene expression values
    out_features=1,         # Predicted expression
    n_heads=1,              # Number of attention heads
    feat_dropout=0.1,
    attn_dropout=0.1,
    negative_slope=0.2,
    residual=False,
    activation=torch.nn.Tanh(),
    ode_method='rk4',       # ODE solver method
    atol=1e-3,
    rtol=1e-4,
    device='cuda'
)

# Forward pass
node_features = torch.randn(n_genes, 1)  # (n_genes, features)
edge_index = torch.tensor([[0, 1], [1, 0]])  # Graph connectivity

predictions, attention_output = model(node_features, edge_index)
```

## Model Architecture

RiTINI consists of three main components:

1. **GAT Convolutional Layer** ([gatConvwithAttention.py](ritini/models/gatConvwithAttention.py))
   - Multi-head attention mechanism
   - Learns edge weights between genes
   - Aggregates neighbor information

2. **Graph Differential Equation** ([graphDifferentialEquation.py](ritini/models/graphDifferentialEquation.py))
   - Wraps GAT layer as ODE function
   - Computes derivatives for continuous-time evolution

3. **ODE Block** ([ode.py](ritini/models/ode.py))
   - Integrates dynamics using Neural ODE solvers
   - Supports multiple integration methods (RK4, Dopri5, etc.)
   - Optional adjoint method for memory-efficient backprop

## Configuration

Key hyperparameters in [main.py](main.py):

```python
# Data parameters
n_top_genes = 20              # Number of genes to model
time_window = 5               # Length of temporal sequences
batch_size = 4

# Model parameters
n_heads = 1                   # GAT attention heads
feat_dropout = 0.1            # Feature dropout rate
attn_dropout = 0.1            # Attention dropout rate
activation_func = nn.Tanh()
residual = False              # Residual connections

# Training parameters
n_epochs = 200
learning_rate = 0.001
lr_factor = 0.5               # Scheduler reduction factor
lr_patience = 10              # Scheduler patience
```

## Data Format

### Trajectory Data
Expected format: `(n_timepoints, n_trajectories, n_genes)`

```python
from ritini.data.trajectory_loader import prepare_trajectories_data

data = prepare_trajectories_data(
    trajectory_file='data/trajectories/traj_data.pkl',
    prior_graph_file='data/trajectories/prior_graph.pkl',
    gene_names_file='data/trajectories/gene_names.txt',
    n_top_genes=20,
    use_mean_trajectory=True
)
```

### Prior Network
Can be provided as:
- NetworkX graph (`.pkl` file)
- Adjacency matrix (NumPy array)

## Visualization

Generate attention heatmaps and trajectory predictions:

```bash
# Visualize RiTINI attention patterns
python ritini/visualizations/visualize_graphs_attention_ritini.py

```

Visualizations are saved to the `visualizations/` directory.

## Testing

Run the test suite:

```bash
# Test on toy data
pytest tests/test_toy_data_ritini.py

# Test on real data
pytest tests/test_real_data_gat.py

# Run all tests
pytest tests/
```

## Dependencies

Core dependencies:
- `torch >= 2.8.0` - Deep learning framework
- `torch-geometric >= 2.6.1` - Graph neural network library
- `torchdiffeq >= 0.2.3` - Neural ODE solvers
- `networkx >= 3.0` - Graph manipulation
- `numpy >= 1.24.0` - Numerical computing
- `matplotlib >= 3.10.6` - Plotting
- `seaborn >= 0.13.2` - Statistical visualization
- `scanpy >= 1.11.4` - Single-cell analysis
- `scikit-misc >= 0.5.1` - Scientific computing utilities

See [pyproject.toml](pyproject.toml) for full dependency list.

## Citation

If you use RiTINI in your research, please cite:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2306.07803,
  doi = {10.48550/ARXIV.2306.07803},
  url = {https://arxiv.org/abs/2306.07803},
  author = {Bhaskar,  Dhananjay and Magruder,  Sumner and De Brouwer,  Edward and Venkat,  Aarthi and Wenkel,  Frederik and Wolf,  Guy and Krishnaswamy,  Smita},
  keywords = {Machine Learning (cs.LG),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Inferring dynamic regulatory interaction graphs from time series data with perturbations},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## License

Yale License


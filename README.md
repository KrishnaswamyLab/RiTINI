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
uv venv
source .venv/bin/activate
uv sync
```

Or using pip:
```bash
pip install -e .
```

## Usage

### Running the Full Pipeline

To run preprocessing and training sequentially:
```bash
python main.py
```

### Running Steps Independently

Each step can be run as a standalone script, which is useful for iterating on specific stages.

#### Preprocessing Only
```bash
python preprocess.py
```
Preprocessing performs:
- Load raw trajectory data from `.npy` file
- Filter genes based on interest genes list
- Compute prior adjacency matrix (uses Granger Causality by default)
- Average all trajectories into a single representative trajectory
- Normalize features using z-score normalization
- Save preprocessed data to `data/preprocessed/` directory

#### Training Only
```bash
python train.py
```
Training performs:
- Load preprocessed data
- Create temporal graph dataset with sliding time windows
- Initialize and train the RiTINI model
- Apply graph regularization based on prior network
- Save best model based on total loss
- Store training history with all loss components

### Trajectory Inference and Visualization
```bash
python gene_inference_viz.py
```

### Obtaining the Time varying GRNs
```bash
python gene_trajectory_viz.py
```

## Input Data

The preprocessing script requires three input files:

1. **Trajectory Data** (`raw_trajectory_file`): `.npy` file containing gene expression trajectories
   - Shape: `(n_timepoints, n_trajectories, n_genes)`
   
2. **Gene Names** (`raw_gene_names_file`): `.txt` file with names of all genes
   
3. **Interest Genes** (`interest_genes_file`): `.txt` file with subset of genes to analyze

### Default Paths
```python
raw_trajectory_file = 'data/raw/traj_data.npy'
raw_gene_names_file = 'data/raw/gene_names.txt'
interest_genes_file = 'data/raw/interest_genes.txt'
```

## Training on Synthetic Data

```bash
python test_toy_data_ritini.py
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

### RiTINI Model Parameters
- **Input features**: 1 (gene expression value per node)
- **Output features**: 1 (predicted expression value)
- **Architecture**: Temporal Graph Attention Network
- **Attention mechanism**: Multi-head attention with configurable heads

## Hyperparameters

### Architecture
```python
n_heads = 1                    # Number of attention heads
feat_dropout = 0.1             # Feature dropout rate
attn_dropout = 0.1             # Attention dropout rate
activation_func = nn.Tanh()    # Activation function
residual = False               # Use residual connections
negative_slope = 0.2           # LeakyReLU negative slope
```

### ODE Integration (Model Defaults)
The RiTINI model uses Neural ODEs for continuous-time modeling:
```python
ode_method = 'rk4'             # ODE solver (rk4, dopri5, etc.)
atol = 1e-3                    # Absolute tolerance
rtol = 1e-4                    # Relative tolerance
use_adjoint = False            # Use adjoint method for memory efficiency
```

### Training
```python
n_epochs = 200                 # Number of training epochs
learning_rate = 0.001          # Initial learning rate
batch_size = 4                 # Batch size
time_window = 5                # Temporal window length (None = all timepoints)
```

### Loss Function
```python
graph_reg_weight = 0.1         # Weight for graph regularization loss
```

Total loss = Feature reconstruction loss + (graph_reg_weight × Graph regularization loss)

### Learning Rate Scheduler
```python
lr_factor = 0.5                # LR reduction factor
lr_patience = 10               # Epochs to wait before reducing LR
```

## Configuration

Key hyperparameters in training:

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


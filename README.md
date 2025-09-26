# RiTINI: Regulatory Temporal Interaction Network Inference

A Python package for inferring regulatory temporal interaction networks using Neural ODEs and Graph Neural Networks.

## Features

- **Neural ODE Models**: Implement continuous-time dynamics for gene regulatory networks
- **Graph Neural Networks**: Support for GCN, GAT, and custom GDE architectures
- **SERGIO Simulator**: Generate synthetic single-cell gene expression data with known ground truth
- **Command-Line Interface**: Easy-to-use CLI for training, inference, and simulation
- **Modular Design**: Clean, extensible codebase with proper logging and error handling

## Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/joaofelipe/RiTINI.git
cd RiTINI

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Dependencies

RiTINI requires Python 3.8+ and the following packages:
- PyTorch >= 1.9.0
- DGL >= 0.9.0
- NumPy, Pandas, SciPy
- NetworkX, Matplotlib, Seaborn
- torchdiffeq >= 0.2.0
- Click >= 8.0.0

## Quick Start

### Command Line Usage

After installation, RiTINI provides several CLI commands:

```bash
# Train a model
ritini train --data /path/to/data.csv --output ./results --model gcn --epochs 100

# Infer networks from trained model
ritini infer --model-path ./results/model.pt --data /path/to/test_data.csv --output networks.csv

# Generate synthetic data with SERGIO
ritini simulate --genes 100 --cells 1000 --bins 5 --output synthetic_data.csv

# Create visualizations
ritini plot --data /path/to/data.csv --output plot.png --plot-type network
```

### Python API Usage

```python
import ritini
from ritini.core.models import get_model
from ritini.core.data import load_data
from ritini.data_generation import sergio

# Load and prepare data
data = load_data('path/to/data.csv')

# Create and train a model
model = get_model('gcn', config={'epochs': 100, 'lr': 0.001})
# model.train(data)  # Training implementation needed

# Generate synthetic data
sim = sergio(number_genes=50, number_bins=3, number_sc=500,
             noise_params=0.2, noise_type='sp', decays=0.8)
# synthetic_data = sim.simulate()  # Implementation needed
```

## Project Structure

```
ritini/
├── src/ritini/
│   ├── __init__.py                 # Main package
│   ├── cli/                        # Command-line interface
│   │   ├── __init__.py
│   │   └── main.py
│   ├── core/                       # Core functionality
│   │   ├── __init__.py
│   │   ├── models/                 # Neural network models
│   │   │   ├── gcn.py             # Graph Convolutional Networks
│   │   │   ├── gat.py             # Graph Attention Networks
│   │   │   ├── gde.py             # Graph Differential Equations
│   │   │   ├── odeblock.py        # Neural ODE blocks
│   │   │   └── factory.py         # Model factory
│   │   ├── data/                   # Data handling
│   │   │   └── data.py            # Data loading and processing
│   │   └── utils/                  # Utilities
│   │       ├── utils.py           # General utilities
│   │       └── plots.py           # Visualization functions
│   └── data_generation/            # Synthetic data generation
│       ├── sergio.py              # SERGIO simulator
│       └── gene.py                # Gene class
├── tests/                          # Unit tests
├── docs/                           # Documentation
├── examples/                       # Example notebooks and scripts
├── pyproject.toml                  # Package configuration
└── README.md                       # This file
```

## Development

### Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/joaofelipe/RiTINI.git
cd RiTINI

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_basic.py

# Run with coverage
pytest --cov=ritini tests/
```

## CLI Reference

### Global Options
- `--verbose, -v`: Increase verbosity (can be used multiple times)
- `--quiet, -q`: Suppress all output except errors

### Commands

#### `train`
Train a neural ODE model for network inference.

```bash
ritini train [OPTIONS]

Options:
  --config, -c PATH     Configuration file path
  --data, -d PATH       Training data path [required]
  --output, -o PATH     Output directory [required]
  --model, -m CHOICE    Model type (gcn|gat|gde) [default: gcn]
  --epochs INTEGER      Number of training epochs [default: 100]
  --lr FLOAT           Learning rate [default: 0.001]
```

#### `infer`
Infer regulatory networks from trained model.

```bash
ritini infer [OPTIONS]

Options:
  --model-path, -m PATH  Path to trained model [required]
  --data, -d PATH        Input data path [required]
  --output, -o PATH      Output file path [required]
  --threshold FLOAT      Edge probability threshold [default: 0.5]
```

#### `simulate`
Generate synthetic data using SERGIO simulator.

```bash
ritini simulate [OPTIONS]

Options:
  --config, -c PATH      SERGIO configuration file
  --genes, -g INTEGER    Number of genes [default: 100]
  --cells, -C INTEGER    Number of cells [default: 1000]
  --bins, -b INTEGER     Number of cell types/bins [default: 10]
  --output, -o PATH      Output file path [required]
```

#### `plot`
Generate visualizations.

```bash
ritini plot [OPTIONS]

Options:
  --data, -d PATH              Data file path [required]
  --output, -o PATH            Output plot path [required]
  --plot-type CHOICE           Plot type (network|heatmap|trajectory) [default: network]
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use RiTINI in your research, please cite:

```bibtex
@software{ritini,
  title={RiTINI: Regulatory Temporal Interaction Network Inference},
  author={Santos, João Felipe},
  year={2024},
  url={https://github.com/joaofelipe/RiTINI}
}
```

## Acknowledgments

- SERGIO simulator for synthetic data generation
- DGL library for graph neural network implementations
- torchdiffeq for neural ODE implementations

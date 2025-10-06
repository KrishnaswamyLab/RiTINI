# RiTINI: Regulatory Temporal Interaction Network Inference

A Python package for inferring regulatory temporal interaction networks using Neural ODEs and Graph Neural Networks.

## Features

- **Neural ODE Models**: Implement continuous-time dynamics for gene regulatory networks
- **Graph Neural Networks**: Support for GCN, GAT, and custom GDE architectures
- **SERGIO Simulator**: Generate synthetic single-cell gene expression data with known ground truth
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

RiTINI requires Python 3.9+ and the following packages:
- PyTorch >= 1.9.0
- PyTorch Geometric >= 2.3.0
- NumPy, Pandas, SciPy
- NetworkX, Matplotlib, Seaborn
- torchdiffeq >= 0.2.0
- scikit-learn >= 1.6.1

## Quick Start

### Python API Usage

```python
import ritini
from ritini.models.factory import get_model
from ritini.data_generation import sergio

# Create RiTINI instance
ri = ritini.RiTINI()

# Train a model
ri.train(data_path='path/to/data.csv', model_type='gcn', epochs=100)

# Infer networks
networks = ri.infer(data_path='path/to/test_data.csv')

# Generate synthetic data with SERGIO
synthetic_data = ri.simulate(genes=100, cells=1000, bins=5)

# Create visualizations
ri.plot(data_path='path/to/data.csv', plot_type='network')
```

## Project Structure

```
ritini/
├── ritini/
│   ├── __init__.py                 # Main package
│   ├── ritini.py                   # Main RiTINI class
│   ├── models/                     # Neural network models
│   │   ├── __init__.py
│   │   ├── gcn.py                 # Graph Convolutional Networks
│   │   ├── gat.py                 # Graph Attention Networks
│   │   ├── gde.py                 # Graph Differential Equations
│   │   ├── pyg.py                 # PyTorch Geometric integration
│   │   ├── odeblock.py            # Neural ODE blocks
│   │   └── factory.py             # Model factory
│   ├── utils/                      # Utilities
│   │   ├── __init__.py
│   │   ├── utils.py               # General utilities
│   │   ├── plots.py               # Visualization functions
│   │   └── data_processing.py     # Data processing utilities
│   └── data_generation/            # Synthetic data generation
│       ├── __init__.py
│       ├── sergio.py              # SERGIO simulator
│       └── gene.py                # Gene class
├── tests/                          # Unit tests
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
black ritini/ tests/
isort ritini/ tests/

# Type checking
mypy ritini/
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

## Acknowledgments

- SERGIO simulator for synthetic data generation
- DGL library for graph neural network implementations
- torchdiffeq for neural ODE implementations

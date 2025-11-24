import yaml
import torch
import torch.nn as nn

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name.lower()]

def get_device(device_config: str) -> torch.device:
    """Get torch device from config."""
    if device_config == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_config)
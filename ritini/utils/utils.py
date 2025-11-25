import yaml
import torch
import torch.nn as nn
from ritini.models.RiTINI import RiTINI

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

def load_trained_model(checkpoint_path, model_config, device):
    """Load a trained RiTINI model from checkpoint using config parameters."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract normalization parameters from checkpoint
    n_genes = checkpoint.get('n_genes')
    mean = checkpoint.get('mean', 0.0)
    std = checkpoint.get('std', 1.0)
    
    # Initialize model with architecture from config
    model = RiTINI(
        in_features=1,
        out_features=1,
        n_heads=model_config['n_heads'],
        feat_dropout=model_config['feat_dropout'],
        attn_dropout=model_config['attn_dropout'],
        negative_slope=model_config['negative_slope'],
        residual=model_config['residual'],
        activation=get_activation(model_config['activation']),
        bias=model_config['bias'],
        device=device
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, n_genes, mean, std
#!/usr/bin/env python3
"""
Example script for training gene expression temporal prediction models.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.gene_dataset import GeneExpressionDataset, create_synthetic_data
from models.temporal_predictor import create_model, count_parameters
from models.trainer import GeneExpressionTrainer
from utils.visualization import (
    plot_training_history,
    plot_predictions_vs_actual,
    plot_correlation_heatmap,
    plot_temporal_predictions
)


def main():
    """Main training function."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    config = {
        # Data parameters
        'n_cells': 200,
        'n_genes': 100,
        'n_time_points': 15,
        'sequence_length': 8,
        'noise_level': 0.15,

        # Model parameters
        'model_type': 'lstm',  # 'lstm', 'transformer', or 'cnn_lstm'
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.2,

        # Training parameters
        'num_epochs': 50,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'validation_split': 0.2,
        'early_stopping_patience': 15,

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=== Gene Expression Temporal Prediction ===")
    print(f"Device: {config['device']}")
    print(f"Model: {config['model_type']}")

    # Generate synthetic data
    print(f"\nGenerating synthetic data...")
    print(f"Shape: ({config['n_cells']}, {config['n_genes']}, {config['n_time_points']})")

    data = create_synthetic_data(
        n_cells=config['n_cells'],
        n_genes=config['n_genes'],
        n_time_points=config['n_time_points'],
        noise_level=config['noise_level'],
        device=config['device']
    )

    # Create dataset
    dataset = GeneExpressionDataset(
        data=data,
        sequence_length=config['sequence_length'],
        normalize=True,
        device=config['device']
    )

    print(f"\nDataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Create model
    print(f"\nCreating {config['model_type']} model...")

    model_kwargs = {
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers'],
        'dropout': config['dropout']
    }

    if config['model_type'] == 'transformer':
        model_kwargs = {
            'd_model': config['hidden_size'],
            'nhead': 8,
            'num_layers': config['num_layers'],
            'dropout': config['dropout']
        }
    elif config['model_type'] == 'cnn_lstm':
        model_kwargs = {
            'cnn_channels': 64,
            'lstm_hidden_size': config['hidden_size'],
            'lstm_layers': config['num_layers'],
            'dropout': config['dropout']
        }

    model = create_model(
        model_type=config['model_type'],
        n_genes=config['n_genes'],
        **model_kwargs
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Create trainer
    trainer = GeneExpressionTrainer(
        model=model,
        device=config['device'],
        learning_rate=config['learning_rate']
    )

    # Train model
    print(f"\nStarting training...")
    print(f"Epochs: {config['num_epochs']}, Batch size: {config['batch_size']}")

    training_results = trainer.train(
        dataset=dataset,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        validation_split=config['validation_split'],
        early_stopping_patience=config['early_stopping_patience'],
        save_path="models/best_model.pt",
        verbose=True
    )

    print(f"\nTraining completed!")
    print(f"Best validation loss: {training_results['best_val_loss']:.6f}")
    print(f"Total training time: {training_results['total_time']:.2f} seconds")

    # Evaluate model
    print(f"\nEvaluating model...")

    # Create test data for evaluation
    with torch.no_grad():
        # Get a batch of test data
        test_inputs = []
        test_targets = []

        for i in range(min(100, len(dataset))):
            inp, targ = dataset[i]
            test_inputs.append(inp.unsqueeze(0))
            test_targets.append(targ.unsqueeze(0))

        test_inputs = torch.cat(test_inputs, dim=0)
        test_targets = torch.cat(test_targets, dim=0)

        # Make predictions
        predictions = trainer.predict(test_inputs)

        # Compute final metrics
        mse = torch.nn.functional.mse_loss(predictions, test_targets)
        mae = torch.nn.functional.l1_loss(predictions, test_targets)

        # Compute correlation
        pred_flat = predictions.flatten().cpu()
        target_flat = test_targets.flatten().cpu()
        correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]

        print(f"Final Test Metrics:")
        print(f"  MSE: {mse.item():.6f}")
        print(f"  MAE: {mae.item():.6f}")
        print(f"  Correlation: {correlation.item():.4f}")

    # Create visualizations
    print(f"\nGenerating visualizations...")

    # Plot training history
    plot_training_history(
        training_results['train_history'],
        training_results['val_history'],
        save_path="plots/training_history.png"
    )

    # Plot predictions vs actual
    plot_predictions_vs_actual(
        predictions[:20],  # First 20 samples
        test_targets[:20],
        n_samples=5,
        n_genes=8,
        save_path="plots/predictions_vs_actual.png"
    )

    # Plot correlation heatmap
    correlations = plot_correlation_heatmap(
        predictions,
        test_targets,
        save_path="plots/correlation_heatmap.png"
    )

    print(f"Mean gene correlation: {np.mean(correlations):.4f}")
    print(f"Std gene correlation: {np.std(correlations):.4f}")

    # Plot temporal predictions for one cell
    plot_temporal_predictions(
        data,
        predictions,
        cell_idx=0,
        gene_indices=[0, 1, 2, 3, 4],
        sequence_length=config['sequence_length'],
        save_path="plots/temporal_predictions.png"
    )

    print(f"\nVisualization saved to plots/ directory")
    print(f"Model saved to models/best_model.pt")


def test_different_models():
    """Test different model architectures."""
    print("=== Testing Different Model Architectures ===")

    # Create small synthetic data for quick testing
    data = create_synthetic_data(n_cells=50, n_genes=30, n_time_points=10, device='cpu')
    dataset = GeneExpressionDataset(data, sequence_length=5, normalize=True)

    models_to_test = ['lstm', 'transformer', 'cnn_lstm']

    for model_type in models_to_test:
        print(f"\nTesting {model_type} model...")

        # Create model
        model = create_model(model_type, n_genes=30, hidden_size=64, num_layers=2)
        print(f"  Parameters: {count_parameters(model):,}")

        # Quick training
        trainer = GeneExpressionTrainer(model, device='cpu', learning_rate=1e-3)
        results = trainer.train(
            dataset,
            num_epochs=5,
            batch_size=16,
            validation_split=0.2,
            verbose=False
        )

        print(f"  Final val loss: {results['val_history']['loss'][-1]:.6f}")
        print(f"  Final val correlation: {results['val_history']['correlation'][-1]:.4f}")


if __name__ == "__main__":
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)

    # Run main training
    main()

    # Test different models
    print("\n" + "="*50)
    test_different_models()
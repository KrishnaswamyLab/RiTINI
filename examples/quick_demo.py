#!/usr/bin/env python3
"""
Quick demo script showing basic usage of the gene expression prediction models.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.gene_dataset import GeneExpressionDataset, create_synthetic_data
from models.temporal_predictor import create_model
from models.trainer import GeneExpressionTrainer


def quick_demo():
    """Quick demonstration of the gene expression prediction pipeline."""
    print("🧬 Gene Expression Temporal Prediction - Quick Demo")
    print("=" * 60)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create synthetic data
    print("\n📊 Creating synthetic data...")
    data = create_synthetic_data(
        n_cells=50,
        n_genes=25,
        n_time_points=12,
        noise_level=0.1,
        device=device
    )
    print(f"Data shape: {data.shape}")

    # Create dataset
    print("\n🔧 Creating dataset...")
    dataset = GeneExpressionDataset(
        data=data,
        sequence_length=6,
        normalize=True,
        device=device
    )

    stats = dataset.get_statistics()
    print(f"Dataset size: {stats['total_sequences']} sequences")
    print(f"Sequence length: {stats['sequence_length']}")

    # Create model
    print("\n🤖 Creating LSTM model...")
    model = create_model(
        model_type='lstm',
        n_genes=25,
        hidden_size=64,
        num_layers=2,
        dropout=0.1
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create trainer
    print("\n🎯 Training model...")
    trainer = GeneExpressionTrainer(
        model=model,
        device=device,
        learning_rate=1e-3
    )

    # Quick training
    results = trainer.train(
        dataset=dataset,
        num_epochs=10,
        batch_size=32,
        validation_split=0.2,
        early_stopping_patience=5,
        verbose=True
    )

    # Test prediction
    print("\n🔮 Making predictions...")

    # Get a test sample
    test_input, test_target = dataset[0]
    test_input = test_input.unsqueeze(0)  # Add batch dimension
    test_target = test_target.unsqueeze(0)

    # Predict
    with torch.no_grad():
        prediction = trainer.predict(test_input)

    # Calculate metrics
    mse = torch.nn.functional.mse_loss(prediction, test_target)
    mae = torch.nn.functional.l1_loss(prediction, test_target)

    print(f"Test MSE: {mse.item():.6f}")
    print(f"Test MAE: {mae.item():.6f}")

    # Show a few predictions vs actual
    print(f"\nSample predictions (first 5 genes):")
    pred_values = prediction[0, :5].cpu().numpy()
    actual_values = test_target[0, :5].cpu().numpy()

    for i in range(5):
        print(f"  Gene {i}: Predicted={pred_values[i]:.4f}, Actual={actual_values[i]:.4f}")

    print(f"\n✅ Demo completed successfully!")
    print(f"Final validation loss: {results['val_history']['loss'][-1]:.6f}")
    print(f"Final validation correlation: {results['val_history']['correlation'][-1]:.4f}")


if __name__ == "__main__":
    quick_demo()
import torch
import os
import yaml
from tqdm import tqdm
import numpy as np
import argparse

from ritini.data.trajectory_loader import prepare_trajectories_data
from ritini.utils.utils import load_config, get_device, load_trained_model
from ritini.utils.attention_graphs import adjacency_to_edge_index, attention_to_adjacency
from ritini.visualizations.graph_visualizations import extract_attention_over_time, visualize_temporal_graphs

def main(checkpoint_path: str):
    """Main visualization pipeline.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint (.pt file).
                        The config.yaml is expected to be in the same directory.
    """
    
    # Derive paths from checkpoint location
    output_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(output_dir, 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}. "
                               f"Expected config.yaml in the same directory as checkpoint.")
    
    # Load configuration
    config = load_config(config_path)
    
    # Device configuration
    device = get_device(config['device'])
    print(f"Using device: {device}")
    
    # Model config
    model_config = config['model']
    
    # Output configuration
    vis_output_dir = os.path.join(output_dir, 'visualizations')
    model_checkpoint = checkpoint_path
    
    # Batching parameters (needed for inference)
    time_window = config['batching']['time_window']
    
    # Visualization-specific config (with defaults)
    # TODO Crete a new visualization config file?
    # TODO: understand dt config.
    # Data parameters from new visualizations config
    # trajectory_file: data/processed/trajectory.npy
    # gene_names_file: data/processed/gene_names.txt
    trajectory_file = 'data/processed/trajectory.npy'
    prior_graph_adjacency_file = 'data/processed/prior_adjacency.npy'
    gene_names_file = 'data/processed/gene_names.txt'

    vis_config = config.get('visualization', {})
    n_cols = vis_config.get('n_cols', 4)
    threshold = vis_config.get('attention_threshold', 0.0053)
    top_k_edges = vis_config.get('top_k_edges', 10)
    animation_fps = vis_config.get('animation_fps', 2)
    figsize_per_graph = tuple(vis_config.get('figsize_per_graph', [4, 4]))
    dt = vis_config.get('dt', 0.1)  # Time step size
    
    # Create output directory
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # Preprocess and load data
    print("\nLoading data...")
    
    data = prepare_trajectories_data(
        trajectory_file=trajectory_file,
        prior_graph_adjacency_file=prior_graph_adjacency_file,
        gene_names_file=gene_names_file
    )
    
    trajectories = data['trajectories']
    n_genes = data['n_genes']
    n_timepoints = data['n_timepoints']
    prior_adjacency = data['prior_adjacency']
    gene_names = data.get('gene_names', [f'Gene_{i}' for i in range(n_genes)])
    
    # Convert prior adjacency to tensor
    prior_adjacency = torch.tensor(prior_adjacency, dtype=torch.float32).to(device)
    
    # Prepare node features (same as training)
    trajectory_idx = 0
    node_features = torch.tensor(trajectories[:, trajectory_idx, :], dtype=torch.float32)
    
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Number of genes: {n_genes}")
    print(f"  Number of timepoints: {n_timepoints}")
    print(f"  Gene names: {gene_names[:5]}..." if len(gene_names) > 5 else f"  Gene names: {gene_names}")
    
    # Load trained model
    print("\nLoading trained model...")
    model, _, mean, std = load_trained_model(model_checkpoint, model_config, device)
    
    # Normalize features using training statistics
    node_features = (node_features - mean) / std
    
    print(f"  Model loaded from {model_checkpoint}")
    print(f"  Normalization: mean={mean:.3f}, std={std:.3f}")
    
    # Extract attention over time
    print("\nExtracting attention weights...")
    attention_history, predictions, edge_index_history = extract_attention_over_time(
        model, node_features, prior_adjacency, device,
        history_length=time_window, dt=dt
    )
    
    print(f"Extracted attention for {len(attention_history)} timepoints")
    
    # Defining the attention threshold as the median + 1 std deviation of all attention weights
    all_weights = []
    for attn in attention_history:
        if attn is not None:
            all_weights.extend(attn.flatten().tolist())

    if all_weights:
        weights_array = np.array(all_weights)
        median_val = np.median(weights_array)
        std_val = np.std(weights_array)
        # threshold = median_val + std_val
        threshold = median_val + 2*std_val
        
        print(f"Attention weights - MIN: {weights_array.min()}, MAX: {weights_array.max()}")
        print(f"Median: {median_val}, Std: {std_val}, Threshold: {threshold}")
    else:
        print("No attention weights found")
        threshold = 0.005  # fallback default
        print(f"Threshold: {threshold}")


    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Grid of temporal graphs
    print("  Creating temporal graph plots...")
    visualize_temporal_graphs(attention_history, 
                              edge_index_history, 
                              gene_names, 
                              n_genes, 
                              vis_output_dir, 
                              threshold=threshold,
                              figsize=(8, 8), node_size=500
    )
    
    # # 2. Attention heatmaps
    # print("  Creating attention heatmaps...")
    # visualize_attention_heatmap_over_time(
    #     attention_history, edge_index_history, gene_names, 
    #     n_genes, vis_output_dir
    # )
    

    # # 3. Top edges over time
    # print("  Creating top edges plot...")
    top_edges = []
    # top_edges = visualize_top_edges_over_time(
    #     attention_history, edge_index_history, gene_names, 
    #     n_genes, vis_output_dir, top_k=top_k_edges
    # )
    
    # # 4. Animation
    # print(" Creating animation...")
    # create_animated_graph(
    #     attention_history, edge_index_history, gene_names, 
    #     n_genes, vis_output_dir, fps=animation_fps
    # )
    
    # # 5. Save raw attention data
    # print("  Saving attention data...")
    # save_attention_data(
    #     attention_history, edge_index_history, gene_names, 
    #     n_genes, vis_output_dir
    # )
    
    # Print summary of top edges
    # print("\n" + "="*60)
    # print("TOP 10 MOST ATTENDED EDGES (by average attention)")
    # print("="*60)
    
    # # Compute average attention matrix
    # attention_matrices = []
    # for attn, edge_idx in zip(attention_history, edge_index_history):
    #     if attn is not None:
    #         attention_matrices.append(attention_to_adjacency(attn, edge_idx, n_genes))
    # avg_attention = np.mean(attention_matrices, axis=0)
    
    # for i, (src, dst) in enumerate(top_edges):
    #     src_name = gene_names[src] if gene_names else str(src)
    #     dst_name = gene_names[dst] if gene_names else str(dst)
    #     print(f"  {i+1}. {src_name} → {dst_name}: {avg_attention[src, dst]:.4f}")
    
    # print("\n" + "="*60)
    # print(f"All visualizations saved to: {vis_output_dir}")
    # print("="*60)
    
    return attention_history, edge_index_history, top_edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize RiTINI temporal graphs')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    args = parser.parse_args()
    
    main(checkpoint_path=args.checkpoint)

# uv run graph_inference.py --checkpoint /home/jcr222/workspace/RiTINI/output/best_model.pt
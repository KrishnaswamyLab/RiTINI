import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from ritini.data.temporal_graph import TemporalGraphDataset
from ritini.data.trajectory_loader import prepare_trajectories_data
from ritini.models.RiTINI import RiTINI
from ritini.train import train_epoch
from ritini.utils.attention_graphs import attention_to_adjacency
from ritini.utils.preprocess import process_trajectory_data
from ritini.utils.utils import get_activation, get_device, load_config, load_trained_model
from ritini.visualizations.graph_visualizations import (
    extract_attention_over_time,
    visualize_focus_gene_temporal_graphs,
    visualize_prior_vs_inferred,
    visualize_temporal_graphs,
)


def _parse_focus_gene_value(value: Any):
    if value is None:
        return None

    normalized = str(value).strip()
    if normalized.lower() in {"none", "null", ""}:
        return None

    if normalized.lstrip("-").isdigit():
        return int(normalized)

    return normalized


def _resolve_data_paths(config: dict[str, Any]) -> tuple[str, str, str]:
    data_cfg = config.get("data", {})
    processed = data_cfg.get("processed", {})

    trajectory_file = processed.get(
        "trajectory_file",
        data_cfg.get("trajectory_file", "data/processed/trajectory.npy"),
    )
    prior_graph_adjacency_file = processed.get(
        "prior_graph_adjacency_file",
        data_cfg.get("prior_adjacency_file", "data/processed/prior_adjacency.npy"),
    )
    gene_names_file = processed.get(
        "gene_names_file",
        data_cfg.get("gene_names_file", "data/processed/gene_names.txt"),
    )

    return trajectory_file, prior_graph_adjacency_file, gene_names_file


def _resolve_focus_idx(focus_gene, gene_names, n_genes):
    if focus_gene is None:
        raise ValueError("No focus gene provided.")

    if isinstance(focus_gene, str):
        if focus_gene not in gene_names:
            raise ValueError(f"Focus gene '{focus_gene}' not found in gene names.")
        focus_idx = gene_names.index(focus_gene)
    else:
        focus_idx = int(focus_gene)

    if focus_idx < 0 or focus_idx >= n_genes:
        raise ValueError(f"Focus gene index {focus_idx} out of range [0, {n_genes - 1}].")

    return focus_idx


def _build_focus_adjacencies(adjacency_matrices, focus_idx, threshold, mode="both", top_k=None):
    valid_modes = {"incoming", "outgoing", "both"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {sorted(valid_modes)}")

    n_timepoints = len(adjacency_matrices)
    n_genes = adjacency_matrices[0].shape[0]

    in_strength = np.zeros(n_timepoints)
    out_strength = np.zeros(n_timepoints)
    filtered_matrices = []
    all_focus_nodes = {focus_idx}

    for adj_np in adjacency_matrices:
        focus_adj = np.zeros_like(adj_np)

        incoming_candidates = [
            (src, float(adj_np[src, focus_idx]))
            for src in range(n_genes)
            if adj_np[src, focus_idx] > threshold
        ]
        outgoing_candidates = [
            (dst, float(adj_np[focus_idx, dst]))
            for dst in range(n_genes)
            if adj_np[focus_idx, dst] > threshold
        ]

        if top_k is not None and top_k > 0:
            incoming_candidates = sorted(incoming_candidates, key=lambda item: item[1], reverse=True)[:top_k]
            outgoing_candidates = sorted(outgoing_candidates, key=lambda item: item[1], reverse=True)[:top_k]

        if mode in {"incoming", "both"}:
            for src, weight in incoming_candidates:
                focus_adj[src, focus_idx] = weight
                all_focus_nodes.add(src)

        if mode in {"outgoing", "both"}:
            for dst, weight in outgoing_candidates:
                focus_adj[focus_idx, dst] = weight
                all_focus_nodes.add(dst)

        in_strength[len(filtered_matrices)] = focus_adj[:, focus_idx].sum()
        out_strength[len(filtered_matrices)] = focus_adj[focus_idx, :].sum()
        filtered_matrices.append(focus_adj)

    return filtered_matrices, in_strength, out_strength, all_focus_nodes


def _snapshot_indices(n_timepoints, n_snapshots=None, snapshot_step=None):
    if snapshot_step is not None and snapshot_step > 0:
        indices = list(range(0, n_timepoints, snapshot_step))
        return indices if len(indices) > 0 else [0]

    n_snapshots = 6 if n_snapshots is None else max(1, int(n_snapshots))
    if n_snapshots >= n_timepoints:
        return list(range(n_timepoints))

    return sorted(set(np.linspace(0, n_timepoints - 1, n_snapshots, dtype=int).tolist()))


def _draw_focus_graph(ax, focus_adj, pos, gene_names, focus_idx, threshold, focus_label):
    rows, cols = np.where(focus_adj > threshold)
    graph = nx.DiGraph()

    active_nodes = set(rows) | set(cols)
    if focus_idx in pos:
        active_nodes.add(focus_idx)

    graph.add_nodes_from(active_nodes)
    for src, dst in zip(rows, cols):
        graph.add_edge(int(src), int(dst), weight=float(focus_adj[src, dst]))

    if len(active_nodes) == 0:
        ax.text(0.5, 0.5, "No edges above threshold", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    draw_pos = {node: pos[node] for node in active_nodes if node in pos}

    node_colors = ["orange" if node == focus_idx else "lightblue" for node in graph.nodes()]
    nx.draw_networkx_nodes(
        graph,
        draw_pos,
        ax=ax,
        node_size=650,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.2,
    )

    weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    if len(weights) > 0:
        min_w, max_w = min(weights), max(weights)
        if max_w > min_w:
            norm = [(w - min_w) / (max_w - min_w) for w in weights]
        else:
            norm = [0.7 for _ in weights]

        edge_colors = [plt.cm.Reds(0.35 + 0.65 * w) for w in norm]
        edge_widths = [1.5 + 2.5 * w for w in norm]

        nx.draw_networkx_edges(
            graph,
            draw_pos,
            ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowsize=14,
            connectionstyle="arc3,rad=0.1",
            alpha=0.95,
        )

    labels = {i: gene_names[i] if i < len(gene_names) else str(i) for i in graph.nodes()}
    nx.draw_networkx_labels(graph, draw_pos, labels, ax=ax, font_size=8)

    n_edges = int((focus_adj > threshold).sum())
    ax.set_title(f"t = {focus_label} ({n_edges} edges)", fontsize=10)
    ax.axis("off")


def _create_focus_gene_storyboard(
    attention_history,
    edge_index_history,
    gene_names,
    n_genes,
    output_dir,
    focus_gene,
    threshold,
    mode="both",
    top_k=None,
    n_snapshots=6,
    snapshot_step=None,
    raw_signals=None,
    history_length=5,
    dt=1.0,
):
    focus_idx = _resolve_focus_idx(focus_gene, gene_names, n_genes)
    focus_label = gene_names[focus_idx] if focus_idx < len(gene_names) else f"G{focus_idx}"

    adjacency_matrices = []
    for attn, edge_idx in zip(attention_history, edge_index_history):
        if attn is not None:
            adj = attention_to_adjacency(attn, edge_idx, n_genes, threshold=threshold)
            adj_np = adj.cpu().numpy() if hasattr(adj, "cpu") else np.array(adj)
        else:
            adj_np = np.zeros((n_genes, n_genes))
        adjacency_matrices.append(adj_np)

    filtered_matrices, _, _, all_focus_nodes = _build_focus_adjacencies(
        adjacency_matrices=adjacency_matrices,
        focus_idx=focus_idx,
        threshold=threshold,
        mode=mode,
        top_k=top_k,
    )

    if len(all_focus_nodes) == 1:
        raise ValueError(f"No focus-gene edges found above threshold for {focus_label}.")

    n_timepoints = len(filtered_matrices)
    snapshot_times = _snapshot_indices(n_timepoints, n_snapshots=n_snapshots, snapshot_step=snapshot_step)

    if raw_signals is None:
        raise ValueError("raw_signals is required to plot actual gene-expression trajectory.")

    raw_np = np.asarray(raw_signals)
    if raw_np.ndim != 2:
        raise ValueError("raw_signals must have shape (n_timepoints, n_genes).")

    if raw_np.shape[0] < history_length + n_timepoints:
        raise ValueError(
            "raw_signals length is too short to align with extracted attention timeline. "
            f"Need at least {history_length + n_timepoints} rows, got {raw_np.shape[0]}."
        )

    expression_values = raw_np[history_length:history_length + n_timepoints, focus_idx]
    expression_time_indices = np.arange(history_length, history_length + n_timepoints)
    expression_time_axis = expression_time_indices * float(dt)

    union_graph = nx.DiGraph()
    union_graph.add_nodes_from(all_focus_nodes)
    for t in snapshot_times:
        focus_adj = filtered_matrices[t]
        rows, cols = np.where(focus_adj > threshold)
        for src, dst in zip(rows, cols):
            union_graph.add_edge(int(src), int(dst))

    pos = nx.spring_layout(union_graph, seed=42, k=2 / np.sqrt(max(1, len(all_focus_nodes))))

    n_cols = max(1, len(snapshot_times))
    fig = plt.figure(figsize=(4 * n_cols, 10))
    gs = fig.add_gridspec(2, n_cols, height_ratios=[1.2, 2.2])

    ax_traj = fig.add_subplot(gs[0, :])
    ax_traj.plot(
        expression_time_axis,
        expression_values,
        color="black",
        linewidth=2,
        label=f"{focus_label} expression",
    )

    for t in snapshot_times:
        x_t = expression_time_axis[t]
        y_t = expression_values[t]
        absolute_t = int(expression_time_indices[t])
        ax_traj.scatter(x_t, y_t, color="tab:orange", s=55, zorder=3)
        ax_traj.text(x_t, y_t, f" t={absolute_t}", fontsize=8, va="bottom")

    y_min = float(np.min(expression_values))
    y_max = float(np.max(expression_values))
    y_span = max(y_max - y_min, 1e-8)
    y_arrow = y_max + 0.12 * y_span

    for left, right in zip(snapshot_times[:-1], snapshot_times[1:]):
        ax_traj.annotate(
            "",
            xy=(expression_time_axis[right], y_arrow),
            xytext=(expression_time_axis[left], y_arrow),
            arrowprops=dict(arrowstyle="->", color="gray", linewidth=1.5),
        )

    ax_traj.set_title(f"Actual trajectory with graph snapshots: {focus_label} ({mode})", fontsize=13)
    ax_traj.set_xlabel("Time")
    ax_traj.set_ylabel("Gene expression")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc="upper right")
    ax_traj.set_ylim(y_min - 0.05 * y_span, y_max + 0.25 * y_span)

    for col, t in enumerate(snapshot_times):
        ax_graph = fig.add_subplot(gs[1, col])
        absolute_t = int(expression_time_indices[t])
        _draw_focus_graph(
            ax=ax_graph,
            focus_adj=filtered_matrices[t],
            pos=pos,
            gene_names=gene_names,
            focus_idx=focus_idx,
            threshold=threshold,
            focus_label=str(absolute_t),
        )

    storyboard_dir = os.path.join(output_dir, "focus_gene_storyboards")
    os.makedirs(storyboard_dir, exist_ok=True)
    save_path = os.path.join(storyboard_dir, f"storyboard_{focus_label}_{mode}.png")

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved focus-gene storyboard to {save_path}")
    return save_path


def preprocess(config_path: str = "configs/config.yaml") -> tuple[str, str, str]:
    """Run RiTINI preprocessing from config and save processed files.

    Returns:
        Tuple of (trajectory_file, prior_adjacency_file, gene_names_file).
    """
    config = load_config(config_path)
    device = get_device(config["device"])
    print(f"Using device: {device}")

    raw_cfg = config["data"]["raw"]
    processed_cfg = config["data"]["processed"]

    outputs = process_trajectory_data(
        raw_cfg["trajectory_file"],
        raw_cfg["gene_names_file"],
        raw_cfg["interest_genes_file"],
        output_trajectory_file=processed_cfg["trajectory_file"],
        output_gene_names_file=processed_cfg["gene_names_file"],
        output_prior_adjacency_file=processed_cfg["prior_graph_adjacency_file"],
        output_prior_graph_plot_file=processed_cfg.get("prior_graph_plot_file"),
        prior_graph_mode=config["data"]["prior_graph_mode"],
        n_highly_variable_genes=config["data"]["n_highly_variable_genes"],
        use_existing_prior_adjacency=config["data"].get("use_existing_prior_adjacency", False),
        existing_prior_adjacency_file=config["data"].get("existing_prior_adjacency_file"),
    )

    print("Data preprocessed successfully.")
    return outputs


def fit(
    config_path: str = "configs/config.yaml",
    preprocess_first: bool = False,
    n_epochs: int | None = None,
    learning_rate: float | None = None,
) -> str:
    """Train RiTINI from a config file.

    Args:
        config_path: Path to YAML config.
        preprocess_first: If True, run preprocessing before training.
        n_epochs: Optional override for training.epochs.
        learning_rate: Optional override for training.learning_rate.

    Returns:
        Path to the best model checkpoint.
    """
    config = load_config(config_path)

    if preprocess_first:
        preprocess(config_path=config_path)

    if n_epochs is not None:
        config["training"]["n_epochs"] = int(n_epochs)
    if learning_rate is not None:
        config["training"]["learning_rate"] = float(learning_rate)

    device = get_device(config["device"])
    print(f"Using device: {device}")

    trajectory_file, prior_graph_adjacency_file, gene_names_file = _resolve_data_paths(config)

    batch_size = config["batching"]["batch_size"]
    time_window = config["batching"]["time_window"]

    model_config = config["model"]
    activation_func = get_activation(model_config["activation"])

    n_epochs_cfg = config["training"]["n_epochs"]
    learning_rate_cfg = config["training"]["learning_rate"]

    graph_reg_weight = config["loss"]["graph_reg_weight"]
    sparsity_weight = config["loss"].get("sparsity_weight", 0.0)

    scheduler_config = config["scheduler"]
    lr_factor = scheduler_config["factor"]
    lr_patience = scheduler_config["patience"]

    output_dir = config["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    data = prepare_trajectories_data(
        trajectory_file=trajectory_file,
        prior_graph_adjacency_file=prior_graph_adjacency_file,
        gene_names_file=gene_names_file,
    )

    trajectories = data["trajectories"]
    n_genes = data["n_genes"]
    n_timepoints = data["n_timepoints"]
    prior_adjacency = torch.tensor(data["prior_adjacency"], dtype=torch.float32).to(device)

    trajectory_idx = 0
    train_node_features = torch.tensor(trajectories[:, trajectory_idx, :], dtype=torch.float32)

    mean = train_node_features.mean()
    std = train_node_features.std()
    train_node_features = (train_node_features - mean) / std

    print("\nData loaded successfully:")
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Number of genes: {n_genes}")
    print(f"  Number of timepoints: {n_timepoints}")
    print(f"  Prior adjacency shape: {prior_adjacency.shape}")

    dataset = TemporalGraphDataset(node_features=train_node_features, time_window=time_window)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = RiTINI(
        in_features=1,
        out_features=1,
        input_latent_dim=model_config["input_latent_dim"],
        history_length=model_config["history_length"],
        n_heads=model_config["n_heads"],
        feat_dropout=model_config["feat_dropout"],
        attn_dropout=model_config["attn_dropout"],
        negative_slope=model_config["negative_slope"],
        residual=model_config["residual"],
        activation=activation_func,
        bias=model_config["bias"],
        ode_method=model_config["ode_method"],
        device=device,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_cfg)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
    )

    print("\nStarting training...")
    best_loss = float("inf")
    training_history = []

    for epoch in tqdm(range(n_epochs_cfg)):
        epoch_loss, epoch_feature_loss, epoch_graph_loss, epoch_sparsity_loss = train_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            device,
            n_genes,
            prior_adjacency,
            graph_reg_weight,
            sparsity_weight,
        )

        training_history.append(
            {
                "total_loss": epoch_loss,
                "feature_loss": epoch_feature_loss,
                "graph_loss": epoch_graph_loss,
                "sparsity_loss": epoch_sparsity_loss,
            }
        )

        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "feature_loss": epoch_feature_loss,
                    "graph_loss": epoch_graph_loss,
                    "sparsity_loss": epoch_sparsity_loss,
                    "n_genes": n_genes,
                    "mean": mean.item(),
                    "std": std.item(),
                },
                os.path.join(output_dir, "best_model.pt"),
            )

    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump({"history": training_history}, f, indent=2)

    best_model_path = os.path.join(output_dir, "best_model.pt")
    print(f"Training completed. Best model: {best_model_path}")
    return best_model_path


def train(*args, **kwargs) -> str:
    """Alias for fit() for a familiar sklearn-style API."""
    return fit(*args, **kwargs)


def focus_storyboard(
    checkpoint_path: str,
    focus_gene=None,
    focus_mode: str | None = None,
    focus_top_k: int | None = None,
    n_snapshots: int = 6,
    snapshot_step: int | None = None,
) -> str:
    """Create focus-gene storyboard visualization and return output image path."""
    output_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(output_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}.")

    config = load_config(config_path)
    device = get_device(config["device"])

    model_config = config["model"]
    time_window = config["batching"]["time_window"]
    vis_config = config.get("visualization", {})

    focus_gene_final = _parse_focus_gene_value(focus_gene) if focus_gene is not None else vis_config.get("focus_gene")
    focus_mode_final = focus_mode if focus_mode is not None else vis_config.get("focus_mode", "both")
    focus_top_k_final = focus_top_k if focus_top_k is not None else vis_config.get("focus_top_k")
    dt = vis_config.get("dt", 0.1)

    if focus_gene_final is None:
        raise ValueError("No focus gene set. Pass focus_gene or set visualization.focus_gene in config.")

    trajectory_file, prior_graph_adjacency_file, gene_names_file = _resolve_data_paths(config)
    data = prepare_trajectories_data(
        trajectory_file=trajectory_file,
        prior_graph_adjacency_file=prior_graph_adjacency_file,
        gene_names_file=gene_names_file,
    )

    trajectories = data["trajectories"]
    n_genes = data["n_genes"]
    prior_adjacency = torch.tensor(data["prior_adjacency"], dtype=torch.float32).to(device)
    gene_names = data.get("gene_names", [f"Gene_{i}" for i in range(n_genes)])

    trajectory_idx = 0
    node_features = torch.tensor(trajectories[:, trajectory_idx, :], dtype=torch.float32)
    model, _, mean, std = load_trained_model(checkpoint_path, model_config, device)
    node_features = (node_features - mean) / std

    attention_history, _, edge_index_history = extract_attention_over_time(
        model,
        node_features,
        prior_adjacency,
        device,
        history_length=time_window,
        dt=dt,
    )

    all_weights = []
    for attn in attention_history:
        if attn is not None:
            all_weights.extend(attn.flatten().tolist())

    threshold = 0.005 if len(all_weights) == 0 else float(np.median(np.array(all_weights)) + np.std(np.array(all_weights)))

    save_path = _create_focus_gene_storyboard(
        attention_history=attention_history,
        edge_index_history=edge_index_history,
        gene_names=gene_names,
        n_genes=n_genes,
        output_dir=os.path.join(output_dir, "visualizations"),
        focus_gene=focus_gene_final,
        threshold=threshold,
        mode=focus_mode_final,
        top_k=focus_top_k_final,
        n_snapshots=n_snapshots,
        snapshot_step=snapshot_step,
        raw_signals=trajectories[:, trajectory_idx, :],
        history_length=time_window,
        dt=dt,
    )
    return save_path


def trajectory_viz(
    checkpoint_path: str,
    visualization_config_path: str = "configs/visualization.yaml",
):
    """Generate trajectory prediction plots.

    Returns:
        rollout_predictions, rollout_times
    """
    output_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(output_dir, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}.")

    config = load_config(config_path)
    device = get_device(config["device"])

    model_config = config["model"]
    time_window = config["batching"]["time_window"]
    history_length = config["batching"].get("history_length", time_window)
    batch_size = config["batching"].get("batch_size", 4)

    trajectory_file, prior_graph_adjacency_file, gene_names_file = _resolve_data_paths(config)

    viz_config = load_config(visualization_config_path)
    dt = viz_config.get("dt", 0.1)
    n_genes_to_plot = viz_config.get("n_genes_to_plot", None)
    plots_dir = Path(output_dir) / viz_config["plots_dir"]
    plots_dir.mkdir(exist_ok=True)

    data = prepare_trajectories_data(
        trajectory_file=trajectory_file,
        prior_graph_adjacency_file=prior_graph_adjacency_file,
        gene_names_file=gene_names_file,
    )
    trajectories = data["trajectories"]
    gene_names = data["gene_names"]
    prior_adjacency = torch.tensor(data["prior_adjacency"], dtype=torch.float32).to(device)
    n_genes = data["n_genes"]
    n_timepoints = data["n_timepoints"]

    signals = trajectories[:, 0, :].astype(np.float32)
    model, _, mean, std = load_trained_model(checkpoint_path, model_config, device)

    signals_normalized = (signals - mean) / std
    train_node_features = torch.tensor(signals_normalized, dtype=torch.float32)

    from ritini.utils.attention_graphs import adjacency_to_edge_index
    edge_index = adjacency_to_edge_index(prior_adjacency).to(device)
    t_eval = torch.arange(1, time_window, device=device) * dt

    gene_variances = np.var(signals, axis=0)
    if n_genes_to_plot is None:
        gene_indices = np.arange(n_genes)
    else:
        gene_indices = np.argsort(gene_variances)[-n_genes_to_plot:][::-1]

    n_genes_plot = len(gene_indices)
    time = np.arange(n_timepoints) * dt

    n_cols = min(3, n_genes_plot)
    n_rows = (n_genes_plot + n_cols - 1) // n_cols
    fig_height = 5 * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, fig_height))
    if n_genes_plot == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    current_t = history_length
    rollout_predictions = []
    rollout_times = []
    temp_features = train_node_features.clone()

    with torch.no_grad():
        while current_t < n_timepoints - time_window:
            x_history = temp_features[current_t - history_length:current_t].T.unsqueeze(-1).to(device)
            pred_traj, _ = model(x_history, edge_index, t_eval)
            pred_traj = pred_traj.squeeze(-1).cpu()

            for i in range(len(pred_traj)):
                rollout_predictions.append(pred_traj[i].numpy())
                rollout_times.append(current_t + 1 + i)

            temp_features[current_t + time_window - 1] = pred_traj[-1]
            current_t += time_window - 1

    rollout_predictions = np.array(rollout_predictions)
    rollout_predictions = rollout_predictions * std + mean
    rollout_times = np.array(rollout_times) * dt

    for plot_idx, gene_idx in enumerate(gene_indices):
        ax = axes_flat[plot_idx]
        ax.plot(time, signals[:, gene_idx], "k-", linewidth=2, label="Ground Truth", alpha=0.8)
        ax.plot(rollout_times, rollout_predictions[:, gene_idx], "r-", linewidth=2, label="Autoregressive", alpha=0.8)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Expression", fontsize=11)
        ax.set_title(f"{gene_names[gene_idx]}", fontsize=12)
        ax.grid(alpha=0.3)
        if plot_idx == 0:
            ax.legend(fontsize=10)

    for idx in range(n_genes_plot, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle("Autoregressive Rollout: Long-Term Prediction", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig(plots_dir / "prediction_autoregressive.png", dpi=200, bbox_inches="tight")
    plt.close()

    dataset = TemporalGraphDataset(
        node_features=train_node_features,
        time_window=time_window,
        history_length=history_length,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_start_indices = []

    with torch.no_grad():
        window_idx = 0
        for batch in dataloader:
            history = batch["history"].to(device)
            batch_size_actual = history.shape[0]
            for b in range(batch_size_actual):
                x_history = history[b].T.unsqueeze(-1)
                pred_traj, _ = model(x_history, edge_index, t_eval)
                pred_traj = pred_traj.squeeze(-1).cpu().numpy()
                all_predictions.append(pred_traj)
                all_start_indices.append(window_idx + history_length)
                window_idx += 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, fig_height))
    if n_genes_plot == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for plot_idx, gene_idx in enumerate(gene_indices):
        ax = axes_flat[plot_idx]
        ax.plot(time, signals[:, gene_idx], "k-", linewidth=2, label="Ground Truth", alpha=0.8)
        pred_by_time = {t: [] for t in range(1, n_timepoints)}

        for pred, start_idx in zip(all_predictions, all_start_indices):
            for i, t in enumerate(range(start_idx + 1, start_idx + 1 + len(pred))):
                if t < n_timepoints:
                    pred_by_time[t].append(pred[i, gene_idx] * std + mean)

        pred_times = []
        pred_means = []
        pred_stds = []
        for t in sorted(pred_by_time.keys()):
            if len(pred_by_time[t]) > 0:
                pred_times.append(time[t])
                pred_means.append(np.mean(pred_by_time[t]))
                pred_stds.append(np.std(pred_by_time[t]))

        pred_times = np.array(pred_times)
        pred_means = np.array(pred_means)
        pred_stds = np.array(pred_stds)

        ax.plot(pred_times, pred_means, "r-", linewidth=2, label="Mean Prediction")
        ax.fill_between(pred_times, pred_means - pred_stds, pred_means + pred_stds, color="r", alpha=0.3, label="+-1 Std")
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Expression", fontsize=11)
        ax.set_title(f"{gene_names[gene_idx]}", fontsize=12)
        ax.grid(alpha=0.3)
        if plot_idx == 0:
            ax.legend(fontsize=9)

    for idx in range(n_genes_plot, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle("Mean +- Std Across All Sliding Windows", fontsize=16)
    plt.tight_layout()
    plt.savefig(plots_dir / "prediction_mean_std.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Trajectory visualizations saved to: {plots_dir}")
    return rollout_predictions, rollout_times


def graph_inference(
    checkpoint_path: str,
    focus_gene=None,
    focus_mode: str | None = None,
    focus_top_k: int | None = None,
):
    """Generate temporal graph inference visualizations from a trained model."""
    output_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(output_dir, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}.")

    config = load_config(config_path)
    device = get_device(config["device"])
    model_config = config["model"]
    time_window = config["batching"]["time_window"]

    vis_config = config.get("visualization", {})
    threshold = vis_config.get("attention_threshold", 0.0053)
    dt = vis_config.get("dt", 0.1)

    focus_gene_final = _parse_focus_gene_value(focus_gene) if focus_gene is not None else vis_config.get("focus_gene")
    focus_mode_final = focus_mode if focus_mode is not None else vis_config.get("focus_mode", "both")
    focus_top_k_final = focus_top_k if focus_top_k is not None else vis_config.get("focus_top_k")

    trajectory_file, prior_graph_adjacency_file, gene_names_file = _resolve_data_paths(config)

    data = prepare_trajectories_data(
        trajectory_file=trajectory_file,
        prior_graph_adjacency_file=prior_graph_adjacency_file,
        gene_names_file=gene_names_file,
    )

    trajectories = data["trajectories"]
    n_genes = data["n_genes"]
    prior_adjacency = torch.tensor(data["prior_adjacency"], dtype=torch.float32).to(device)
    gene_names = data.get("gene_names", [f"Gene_{i}" for i in range(n_genes)])

    node_features = torch.tensor(trajectories[:, 0, :], dtype=torch.float32)

    model, _, mean, std = load_trained_model(checkpoint_path, model_config, device)
    node_features = (node_features - mean) / std

    attention_history, _, edge_index_history = extract_attention_over_time(
        model,
        node_features,
        prior_adjacency,
        device,
        history_length=time_window,
        dt=dt,
    )

    all_weights = []
    for attn in attention_history:
        if attn is not None:
            all_weights.extend(attn.flatten().tolist())

    if all_weights:
        weights_array = np.array(all_weights)
        threshold = float(np.median(weights_array) + np.std(weights_array))
    else:
        threshold = 0.005

    vis_output_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_output_dir, exist_ok=True)

    visualize_temporal_graphs(
        attention_history,
        edge_index_history,
        gene_names,
        n_genes,
        vis_output_dir,
        threshold=threshold,
        figsize=(8, 8),
        node_size=500,
    )

    attention_matrices = []
    for attn, edge_idx in zip(attention_history, edge_index_history):
        if attn is not None:
            attention_matrices.append(attention_to_adjacency(attn, edge_idx, n_genes))

    if len(attention_matrices) > 0:
        avg_attention = torch.stack(attention_matrices, dim=0).mean(dim=0)
        visualize_prior_vs_inferred(
            prior_adjacency=prior_adjacency,
            inferred_adjacency=avg_attention,
            gene_names=gene_names,
            output_dir=vis_output_dir,
            threshold=threshold,
            node_size=500,
        )

    if focus_gene_final is not None:
        visualize_focus_gene_temporal_graphs(
            attention_history=attention_history,
            edge_index_history=edge_index_history,
            gene_names=gene_names,
            n_genes=n_genes,
            output_dir=vis_output_dir,
            focus_gene=focus_gene_final,
            threshold=threshold,
            mode=focus_mode_final,
            top_k=focus_top_k_final,
            figsize=(8, 8),
            node_size=600,
        )

    print(f"Graph inference visualizations saved to {vis_output_dir}")
    return attention_history, edge_index_history


def storyboard(*args, **kwargs):
    """Alias for focus_storyboard()."""
    return focus_storyboard(*args, **kwargs)


def trajectory(*args, **kwargs):
    """Alias for trajectory_viz()."""
    return trajectory_viz(*args, **kwargs)


def infer_graph(*args, **kwargs):
    """Alias for graph_inference()."""
    return graph_inference(*args, **kwargs)

import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from ritini.data.trajectory_loader import prepare_trajectories_data
from ritini.utils.attention_graphs import attention_to_adjacency
from ritini.utils.utils import get_device, load_config, load_trained_model
from ritini.visualizations.graph_visualizations import extract_attention_over_time


def _parse_focus_gene_value(value):
    if value is None:
        return None

    normalized = str(value).strip()
    if normalized.lower() in {"none", "null", ""}:
        return None

    if normalized.lstrip("-").isdigit():
        return int(normalized)

    return normalized


def _resolve_focus_idx(focus_gene, gene_names, n_genes):
    if focus_gene is None:
        raise ValueError("No focus gene provided. Use --focus-gene with gene name or index.")

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


def create_focus_gene_storyboard(
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
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved focus-gene storyboard to {save_path}")
    return save_path


def main(
    checkpoint_path,
    focus_gene=None,
    focus_mode=None,
    focus_top_k=None,
    n_snapshots=6,
    snapshot_step=None,
):
    output_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(output_dir, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at {config_path}. Expected config.yaml in the same directory as checkpoint."
        )

    config = load_config(config_path)
    device = get_device(config["device"])
    model_config = config["model"]
    time_window = config["batching"]["time_window"]
    vis_config = config.get("visualization", {})

    focus_gene_cfg = vis_config.get("focus_gene")
    focus_mode_cfg = vis_config.get("focus_mode", "both")
    focus_top_k_cfg = vis_config.get("focus_top_k")
    dt = vis_config.get("dt", 0.1)

    focus_gene_final = _parse_focus_gene_value(focus_gene) if focus_gene is not None else focus_gene_cfg
    focus_mode_final = focus_mode if focus_mode is not None else focus_mode_cfg
    focus_top_k_final = focus_top_k if focus_top_k is not None else focus_top_k_cfg

    if focus_gene_final is None:
        raise ValueError("No focus gene set. Pass --focus-gene or set visualization.focus_gene in config.")

    trajectory_file = "data/processed/trajectory.npy"
    prior_graph_adjacency_file = "data/processed/prior_adjacency.npy"
    gene_names_file = "data/processed/gene_names.txt"

    print(f"Using device: {device}")
    print("Loading data...")
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

    print("Loading trained model...")
    model, _, mean, std = load_trained_model(checkpoint_path, model_config, device)
    node_features = (node_features - mean) / std

    print("Extracting attention weights...")
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

    if len(all_weights) == 0:
        threshold = 0.005
    else:
        weights_array = np.array(all_weights)
        threshold = float(np.median(weights_array) + np.std(weights_array))

    create_focus_gene_storyboard(
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create focus-gene dynamics + graph storyboard visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument(
        "--focus-gene",
        type=str,
        default=None,
        help="Gene name (e.g., G51) or integer index (e.g., 12)",
    )
    parser.add_argument(
        "--focus-mode",
        type=str,
        choices=["incoming", "outgoing", "both"],
        default=None,
        help="Focus-edge direction mode",
    )
    parser.add_argument(
        "--focus-top-k",
        type=int,
        default=None,
        help="Keep only top-k neighbors per snapshot",
    )
    parser.add_argument(
        "--n-snapshots",
        type=int,
        default=10,
        help="Number of snapshots to show (ignored if --snapshot-step is set)",
    )
    parser.add_argument(
        "--snapshot-step",
        type=int,
        default=None,
        help="Snapshot every N timepoints",
    )

    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        focus_gene=args.focus_gene,
        focus_mode=args.focus_mode,
        focus_top_k=args.focus_top_k,
        n_snapshots=args.n_snapshots,
        snapshot_step=args.snapshot_step,
    )

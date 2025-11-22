
from ritini.utils.prior_graph import compute_prior_adjacency
from ritini.data.trajectory_loader import prepare_trajectories_data, process_single_trajectory_data


if __name__ == "__main__":

    # Data parameters
    trajectory_file = 'data/natalia/traj_data.npy' 
    gene_names_file='data/natalia/gene_names.txt'
    interest_genes_file = 'data/natalia/interest_genes.txt'


    # Load trajectory data
    data = prepare_trajectories_data(
        trajectory_file=trajectory_file,
        prior_graph_adjacency_file=prior_graph_adjacency_file,
        gene_names_file=gene_names_file,
        use_mean_trajectory=True,
    )


    prior_adj = compute_prior_adjacency(
        time_series,
        mode='granger_causality',
        lag_order=2,
        neg_log_threshold=3.0,
        edge_weighted=True,
        directed=True
    )

    print("Computed Prior Adjacency Matrix:")
    print(prior_adj)
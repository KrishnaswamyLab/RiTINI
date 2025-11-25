from ritini.utils.utils import load_config, get_device
from ritini.utils.preprocess import process_trajectory_data

def preprocess(config_path: str = 'config/config.yaml'):
    # Load configuration
    config = load_config(config_path)
    
    # Device configuration
    device = get_device(config['device'])
    print(f"Using device: {device}")

    # Data parameters from config
    raw_trajectory_file = config['data']['raw']['trajectory_file']
    raw_gene_names_file = config['data']['raw']['gene_names_file']
    interest_genes_file = config['data']['raw']['interest_genes_file']

    prior_graph_mode = config['data']['prior_graph_mode']
    n_highly_variable_genes = config['data']['n_highly_variable_genes']

    # Processed data parameters from config
    trajectory_file = config['data']['processed']['trajectory_file']
    gene_names_file = config['data']['processed']['gene_names_file']
    prior_graph_adjacency_file = config['data']['processed']['prior_graph_adjacency_file']

    # Preprocess input data
    process_trajectory_data(
        raw_trajectory_file,
        raw_gene_names_file,
        interest_genes_file,
        output_trajectory_file=trajectory_file,
        output_gene_names_file=gene_names_file,
        output_prior_adjacency_file=prior_graph_adjacency_file,
        prior_graph_mode=prior_graph_mode,
        n_highly_variable_genes=n_highly_variable_genes)
    
    print(f"\nData preprocessed successfully:")

if __name__ == "__main__":
    preprocess()

import numpy as np
import pandas as pd
#TODO: CLEAN THIS MESS
def get_n_cells_of_type_k_at_time_t(df, n, k, t, genes):
    n_genes = len(genes)
    groups = df.groupby(['cell_types', 'pseudotime'])
    if (k, t) not in groups.groups:
        values = np.array([[0 for cell in range(n)] for gene in range(n_genes)])
    else:
        unique_genes = list(pd.unique(genes))
        values = groups.get_group((k, t))\
                    .filter(unique_genes).sample(n, replace=True)\
                    .values.T
        # values = groups.get_group((k, t))\
        #        .loc[:, unique_genes].sample(n, replace=True)\
        #        .values.T
        # values_list.append(values)
    # e.g. shape = (100 genes, 10 cells)
    genes_x_cells = values
    return genes_x_cells

def get_n_cells_of_all_types_at_time_t(df, n, t, types=None, genes=None):
    if types is None:
        types = np.unique(df['cell_types'])
    return np.hstack(tuple([
        get_n_cells_of_type_k_at_time_t(df, n, k, t, genes=genes)
        for k in types
    ]))
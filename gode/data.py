# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/06_data.ipynb.

# %% auto 0
__all__ = ['augment_with_time', 'group_extract', 'sample_group', 'sample_group_index', 'clean_features_and_drop_columns',
           'aggregate_df_group', 'sample_aggregate_group_at_t', 'representative_cell_types_at_t',
           'make_train_test_dataframe', 'get_mean_features_per_group_and_time', 'make_mean_data_ti', 'get_data_ti',
           'stack_timepoints_into_dataframe', 'make_results_dataframe', 'get_spearmanr_at_t_k', 'get_spearmanr']

# %% ../nbs/06_data.ipynb 3
import os, numpy as np, pandas as pd, torch, itertools
from typing import Callable, Union
from .utils import get_device, torch_t

def augment_with_time(
    x:torch.Tensor, 
    t:int, size:int=1, 
    augment:bool=True
) -> torch.Tensor:  
    '''
    Augment feature matrix x with zeros and time.

    Parameters
    ----------
    x
        The input features to augment.
    
    t
        Time to append to x.

    size
        Number of columns of zeros to add to x.

    augment
        Whether or not to augment x with zeroes and time. If `False` returns x unchanged
    
    Returns
    -------
    augmented
        The augmented tensor (x, t, zeros...).
    '''
    # Internally handle if / else statement
    if not augment:
        return x
    
    # Ensure t is wrapped as torch Tensor
    t = torch_t(t, device=x.device)
    
    # Augment with size number of 0s
    zeros = torch.zeros(x.size(dim=0), size).to(x.device)
    
    # Time is only concatenated once
    times = t.repeat(x.size(dim=0), 1)
        
    augmented = torch.cat((x, times, zeros), dim=1)
    return augmented

# %% ../nbs/06_data.ipynb 4
def group_extract(
    df:pd.DataFrame,
    group:str, groupby:str='binned_sim_time', 
    index:str=None, set_index:bool=True, 
    as_df:bool=False
):
    '''
    Gets group from a DataFrame

    Parameters
    ----------
    df
        Pandas DataFrame

    group
        group to extract
    
    groupby
        key in df to group by

    index
        key in df to set to index

    set_index
        whether or not to set index

    as_df
        whether or not to return results in a DataFrame or NumPy array
    
    Returns
    -------
    group
        records of df that have df.groupby == group 
    '''
    # Get just the records in the corresponding group
    df_g = df.groupby(groupby).get_group(group)
    
    # Set the requested index column
    if index is not None and set_index:
        df_g = df_g.set_index(index)
    
    # Just get the numpy array
    if not as_df:
        df_g = df_g.values
    
    return df_g


def sample_group(
    df:pd.DataFrame,
    group:str, groupby:str='binned_sim_time', 
    index:str=None, set_index:bool=True, 
    as_df:bool=False,
    size:int=100, replace:bool=False,
    to_torch:bool=False, device=None
):
    '''
    Samples a group from a DataFrame

    Parameters
    ----------
    df
        Pandas DataFrame

    group
        group to extract
    
    groupby
        key in df to group by

    index
        key in df to set to index

    set_index
        whether or not to set index

    as_df
        whether or not to return results in a DataFrame or NumPy array

    size
        number of records to retrieve

    replace
        whether or not to sample with replacement

    to_torch
        whether or not to cast to `torch.Tensor`

    device
        which device to put results on if `to_tensor=True`
    
    Returns
    -------
    sampled
        `size` records of df that have df.groupby == group 
    '''
    sub = group_extract(df, group, groupby, index, set_index, as_df)
    sampled = sub.sample(size, replace=replace)
    if to_torch:
        sampled = torch.tensor(sub)
        if device is not None:
            sampled = sub.to(device)
    return sampled


def sample_group_index(
    df, group, groupby:str='binned_sim_time', size:int=100, replace:bool=False
) -> np.array:
    '''
    Samples a group from a DataFrame

    Parameters
    ----------
    df
        Pandas DataFrame

    group
        group to extract
    
    groupby
        key in df to group by

    size
        number of records to retrieve

    replace
        whether or not to sample with replacement

    Returns
    -------
    indicies
        indicies of the sampled DataFrame
    '''
    df_sampled = sample_group(df, group, groupby, size=size, as_df=True, replace=replace)
    return df_sampled.index.values

# %% ../nbs/06_data.ipynb 5
def clean_features_and_drop_columns(df, features=None, drop_columns=[]):
    '''
    Given a DataFrame returns all columns in features not in drop_columns

    Parameters
    ----------
    df
        Pandas DataFrame

    features
        columns explicitly to keep. If None, then uses all of them.
    
    drop_columns
       columns explicitly to drop. If [], then drops none of them.

    Returns
    -------
    features
        columns from features that are also in df

    drop_columns
        columns from drop_columns that are also in df
    '''
    # Get all known features in DataFrame
    if features is None:
        features = df.columns
        
    # Can only drop columns if they are in the DataFrame to begin with
    drop_columns = [col for col in drop_columns if col in df.columns]
    
    # Can subset DataFrame columns by keeping columns we want, or dropping unwanted ones
    # here we use a combined approach
    features = [feature for feature in features if feature not in drop_columns]
    return features, drop_columns


def aggregate_df_group(
    df_grouped, group, aggregation='mean', missing_value=0,
    features=None, drop_columns=[]
):
    '''
    Aggregates rows for a given group of a DataFrame

    Parameters
    ----------
    df_grouped
        Pandas DataFrame grouped 

    group
        group to extract
    
    aggregation
        aggregation function to use e.g. mean, sum, etc

    missing_value
        What to fill in for each column if group does not exist.

    features
        columns of df explicitly to keep. If None, then uses all of them.
    
    drop_columns
       columns of df explicitly to drop. If [], then drops none of them.

    Returns
    -------
    group_agg
        values of the aggregated group from the df
    '''
    if features is None:
        features, drop_columns = clean_features_and_drop_columns(df_group, features, drop_columns)

    # TODO: maybe fix sampling?
    # Poor sample, missing a cluster group so just setting its value to all zeros
    if group not in df_grouped.groups:
        return [missing_value for feature in features]

    # Ids of rows in current group
    df_group = df_grouped.get_group(group)
    group_idx = df_group.index

    
    # Filter df
    # NOTE: .drop(columns=drop_columns) would be redundant, we already filtered
    # those out for features
    df_group = df_group.loc[group_idx, features]

    # Aggregate across groups e.g. this is akin to df.mean()
    group_agg = getattr(df_group, aggregation)()
    return group_agg.values


def sample_aggregate_group_at_t(
    df, t, time_key:str='binned_sim_time', 
    size=100, replace:bool=False,
    groupby:str='cell_type',  groups = None,
    aggregation='mean', missing_value = 0, 
    features = None, drop_columns=[]
):
    '''
    Aggregates rows for a given group of a DataFrame

    Parameters
    ----------
    df
        Pandas DataFrame 

    t
        current time to extract
    
    time_key
        column name of df corresponding to time

    size
        number of samples to extract

    replace
        whether or not to sample with replacement

    groupby
        key of df to group by.

    groups
        list of all known groups in groupby. If None will calculate
            via df[groupby].unique()

    aggregation
        aggregation function to use e.g. mean, sum, etc

    missing_value
        What to fill in for each column if group does not exist.

    features
        columns of df explicitly to keep. If None, then uses all of them.
    
    drop_columns
       columns of df explicitly to drop. If [], then drops none of them.

    Returns
    -------
    df_groups_t
        values of the aggregated groups of the df at t
    '''
    # Get all known clusters regardless of time
    if groups is None:
        groups = sorted(df[groupby].unique())
   
    features, drop_columns = clean_features_and_drop_columns(df, features, drop_columns)

    # Get ids of the recprds in our sample
    idx = sample_group_index(df, t, time_key, size, replace)
    
    # Group records from our sample to their cluster
    df_grouped = df.loc[idx].groupby(groupby)
    
    # NOTE: could do this in list comprehension, but this is easier to read
    # Bookkeeping variable
    results = []
    for group in groups:
        group_agg = aggregate_df_group(
            df_grouped, group, aggregation, missing_value,
            features, drop_columns
        )        
        results.append(group_agg)
    df_groups_t = pd.DataFrame(results, index=groups, columns=features)
    return df_groups_t

def representative_cell_types_at_t(
    df_cells, t, time_key:str='binned_sim_time', 
    size=100, replace:bool=False,
    groupby:str='cell_type',  groups = None,
    aggregation='mean', missing_value = 0, 
    features = None, drop_columns=[]
):
    '''
    Wrapper for sample_aggregate_group_at_t
    '''
    return sample_aggregate_group_at_t(
        df_cells, t, time_key, size, replace, groupby, groups, 
        aggregation, missing_value, features, drop_columns
    )

# %% ../nbs/06_data.ipynb 6
def make_train_test_dataframe(df:pd.DataFrame, fraction:float=85/100):
    df_train = df.sample(frac=85/100)
    df_test = df.loc[~df.index.isin(df_train.index)]
    return df_train, df_test

def get_mean_features_per_group_and_time(
    df:pd.DataFrame, 
    features:Union[list, np.ndarray],
    time_key:str='binned_sim_time',
    groupby:str='cell_type',
):
    df_mu = df.groupby([time_key, groupby])\
    .mean().filter(features)\
    .reset_index()
    return df_mu

def make_mean_data_ti(
    df:pd.DataFrame, 
    features:Union[list, np.ndarray, pd.Series],
    time_key:str='binned_sim_time',
    groupby:str='cell_type',
    device:torch.device=None,
    time_bins:Union[list, np.ndarray, pd.Series]=None
):
    if device is None:
        device = get_device()
        
    if time_bins is None:
        time_bins = np.sort(df[time_key].unique())

    groups = np.sort(df[groupby].unique())

    # NOTE: sometimes throws ValueError: setting an array element with a sequence.
    # for no reason
    # df_mu = get_mean_features_per_group_and_time(df, features, time_key, groupby)
    # data_ti = np.array([
    #     df_mu[df_mu[time_key] == t].filter(features).values.astype(float)
    #     for t in time_bins
    # ]).astype(float)
    # data_ti = torch.transpose(torch.Tensor(data_ti), 1, 2).to(device)

    # NOTE: Replacement option 1. for loops
    # res = []
    # for t in time_bins:
    #     df_t = df_mu[df_mu[time_key] == t]

    #     for group in groups:
    #         df_tc = df_t[df_t[groupby] == group]
    #         df_tc = df_tc.filter(features)
    #         if df_tc.empty:
    #             values = [0 for feature in features]
    #         else:
    #             values = df_tc.values.flatten().tolist()
    #             values
    #         res.append(values)
    # data_ti = np.array(res).astype(float)

    # NOTE: Replacement option 2. itertools
    df_g = df.groupby([time_key, groupby])
    keys = itertools.product(time_bins, groups)    
    res = np.empty(0)
    for (t, group) in keys:
        try:
            values = df_g.get_group((t, group)).filter(features).mean().values
        except KeyError:
            values = np.array([0 for feature in features])
        res = np.vstack((res, values)) if res.size else values
    data_ti = np.array(res).astype(float).reshape(len(time_bins), len(groups), -1)
            
    data_ti = torch.transpose(torch.Tensor(data_ti), 1, 2).to(device)
    return data_ti

def get_data_ti(
    df:pd.DataFrame, 
    t, 
    size:int,
    features:Union[list, np.ndarray, pd.Series],
    replace:bool=False,
    time_key:str='binned_sim_time',
    groupby:str='cell_type',
    device:torch.device=None
):
    if device is None:
        device = get_device()
        
    return torch.Tensor(
        sample_aggregate_group_at_t(
            df, t, time_key=time_key, 
            size=size, replace=replace,
            groupby=groupby, features=features
        ).values
    ).to(device).T


def stack_timepoints_into_dataframe(
    data_ti:Union[list, np.ndarray, torch.Tensor],
    genes:Union[list, np.ndarray, torch.Tensor],
    cell_types:Union[list, np.ndarray, torch.Tensor],
    transcription_factors:Union[list, np.ndarray, torch.Tensor],
):
    inner_cols = np.concatenate((genes, ['time']))
    outer_cols = np.concatenate((['cell_type', 'time'], transcription_factors))
    df = pd.DataFrame(
            np.array([
                pd.DataFrame(
                    np.hstack((
                        dt.T.detach().cpu().numpy(), 
                        np.repeat(i, len(cell_types)).reshape(-1, 1)
                    )), columns=inner_cols, index=cell_types
                )
                    .reset_index()
                    .rename(columns={'index':'cell_type'})
                    .loc[:, outer_cols]
                    .values
                for i, dt in enumerate(data_ti)
            ], dtype=object).reshape(-1, 2 + len(transcription_factors)),
            columns=outer_cols
    )
    return df

def make_results_dataframe(
    data_ti:Union[list, np.ndarray, torch.Tensor],
    data_pi:Union[list, np.ndarray, torch.Tensor],
    genes:Union[list, pd.Series, np.ndarray],
    cell_types:Union[list, np.ndarray],
    transcription_factors:Union[list, np.ndarray],
):
    df_true = stack_timepoints_into_dataframe(data_ti, genes, cell_types, transcription_factors)
    df_pred = stack_timepoints_into_dataframe(data_pi, genes, cell_types, transcription_factors)
    res = []
    for kind, df_cur in zip(('ground_truth', 'prediction'), (df_true, df_pred)):
        for row, record in df_cur.iterrows():
            for tf in transcription_factors:
                res.append({
                    'cell_type': record.cell_type,
                    'time': record.time,
                    'tf': tf,
                    'expression': record[tf],
                    'type': kind
                })
    df_res = pd.DataFrame(res)
    return df_res

 

def make_results_dataframe(
    data_ti:Union[list, np.ndarray, torch.Tensor],
    data_pi:Union[list, np.ndarray, torch.Tensor],
    genes:Union[list, pd.Series, np.ndarray],
    cell_types:Union[list, np.ndarray],
    transcription_factors:Union[list, np.ndarray],
):
    df_true = stack_timepoints_into_dataframe(data_ti, genes, cell_types, transcription_factors)
    df_pred = stack_timepoints_into_dataframe(data_pi, genes, cell_types, transcription_factors)
    df_pred.loc[:, 'time'] += 1
    res = []
    for kind, df_cur in zip(('ground_truth', 'prediction'), (df_true, df_pred)):
        for row, record in df_cur.iterrows():
            for tf in transcription_factors:
                res.append({
                    'cell_type': record.cell_type,
                    'time': record.time,
                    'tf': tf,
                    'expression': record[tf],
                    'type': kind
                })
    df_res = pd.DataFrame(res)
    return df_res



# %% ../nbs/06_data.ipynb 7
from scipy.stats import spearmanr
from .utils import dearray
def get_spearmanr_at_t_k(data_ti, data_tp, t=0, k=0, drop_one=True):
    # NOTE: t = gene (TF factor), k = kind, 
    
    # NOTE: start at one, because data_tp is for t=1-->t=N
    #       whereas data_ti starts at t=0
    n = 1 if drop_one else 0
    arr_x = data_ti[n:, t, k]    
    arr_y = data_tp[:, t, k]
    s_cor = spearmanr(arr_x, arr_y)
    return s_cor

def get_spearmanr(
    data_ti, data_tp, drop_one=True,
    columns=None, index=None
):
    # shape for dyngen was like (50, 116, 2) i.e. (t, genes, cell types)
    x = np.array(dearray(data_ti))
    y = np.array(dearray(data_tp))
    corrs = np.array([ 
        [
           get_spearmanr_at_t_k(x, y, t, k, drop_one).correlation
            for k in range(data_ti.shape[-1])    
        ]
        for t in range(data_ti.shape[1])
    ])
    return pd.DataFrame(corrs, columns=columns, index=index).replace(np.nan, 0)  

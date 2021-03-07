import warnings

import arviz as az
import numpy as np
import pandas as pd
import scipy
import xarray as xr
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform


def level_to_data_column(level_name, kwargs):
    from collections import Iterable
    # import itertools
    # flatten = itertools.chain.from_iterable
    x = kwargs[level_name]
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        if len(x) > 1:
            raise ValueError(f'Multiple conditions are not supported:{x}')
        # list(flatten(level_name))
        return kwargs[level_name][0]
    else:
        return kwargs[level_name]


def parse_levels(treatment, condition, group):
    levels = []
    if treatment:
        levels += [treatment]
    if condition[0]:
        levels += condition
    if group:
        levels += [group]
    return levels


def add_data_to_posterior(df_data,
                          posterior,
                          y=None,  # Only for fold change
                          fold_change_index_cols=None,
                          treatment_name='Event',
                          treatments=None,  # eg ('stim_on', 'stim_stop')
                          b_name='b_stim_per_condition',  # for posterior
                          posterior_index_name='',  # for posterior
                          do_make_change='subtract',
                          do_mean_over_trials=True,
                          add_data=False
                          ):
    # group_name should be conditions
    if type(fold_change_index_cols) == str:
        fold_change_index_cols = [fold_change_index_cols]
    fold_change_index_cols = list(fold_change_index_cols)
    treatments = treatments or df_data[treatment_name].drop_duplicates().sort_values().values

    assert len(treatments) == 2, f'{treatment_name}={treatments}. Should be only two instead!'
    assert do_make_change in [False, 'subtract', 'divide']
    if not (treatment_name in fold_change_index_cols):
        fold_change_index_cols.append(treatment_name)
    # if not (group_name in index_cols):
    #     index_cols.append(group_name)
    if do_mean_over_trials:
        df_data = df_data.groupby(fold_change_index_cols).mean().reset_index()
    if do_make_change:
        # Make (fold) change
        df_data, _ = make_fold_change(df_data,
                                      y=y,
                                      index_cols=fold_change_index_cols,
                                      treatment_name=treatment_name,
                                      treatments=treatments,
                                      fold_change_method=do_make_change,
                                      do_take_mean=False)
    # Convert to dataframe and fill in data:
    df_bayes, posterior = trace2df(posterior, df_data, b_name=b_name, posterior_index_name=posterior_index_name,
                                   add_data=add_data)
    return df_bayes, posterior


def fill_row(condition_val, rows, df_bayes, condition_name):
    this_hdi = df_bayes.loc[df_bayes[condition_name] == condition_val]
    assert this_hdi.shape[0] > 0, \
        f'No such value {condition_val} in {condition_name}: it"s {df_bayes[condition_name].unique()}'
    for col in ['lower interval', 'higher interval', 'center interval']:
        rows.insert(rows.shape[1] - 1, col, this_hdi[col].values.squeeze())
    return rows


def hdi2df_many_conditions(df_bayes, posterior_index_name, df_data):
    # Check
    if len(df_data[posterior_index_name].unique()) != len(df_bayes[posterior_index_name].unique()):
        raise ValueError('Groups were constructed differently for estimation and data. Cant add data for plots')
    rows = [fill_row(group_val, rows, df_bayes, posterior_index_name)
            for group_val, rows in df_data.groupby([posterior_index_name])]
    return pd.concat(rows)


def hdi2df_one_condition(df_bayes, group_name, df_data):
    df_bayes[group_name] = df_data[group_name].iloc[0]
    for col in ['lower interval', 'higher interval', 'center interval']:
        df_data.insert(df_data.shape[1], col, df_bayes[col].values.squeeze())
    return df_data


def _mode(*args, **kwargs):
    vals = scipy.stats.mode(*args, **kwargs)
    # only return the mode (discard the count)
    return vals[0].squeeze()


def xar_mode(obj, dims_to_reduce: list = None, dim=None):
    # xar_mode(trace[b_name], dims_to_reduce=['chain', 'draw']) # OR:
    # xar_mode(trace[b_name], dim='neuron')

    # note: apply always moves core dimensions to the end
    # usually axis is simply -1 but scipy's mode function doesn't seem to like that
    # this means that this version will only work for DataArrays (not Datasets)
    assert isinstance(obj, xr.DataArray)
    assert (dims_to_reduce is not None) or (dim is not None)
    axis = obj.ndim - 1
    if dims_to_reduce is None:
        dims_to_reduce = list(set(obj.dims) - {dim})
    return xr.apply_ufunc(_mode, obj,
                          input_core_dims=[dims_to_reduce],
                          # exclude_dims=set(['chain','draw']),
                          kwargs={'axis': axis}
                          )


def trace2df(trace, df_data, b_name='b_stim_per_condition', posterior_index_name='neuron', add_data=False):
    """
    # Convert to dataframe and fill in original conditions
    group name is whatever was used to index posterior
    """
    # TODO this is lazy. There may be more than one condition, need to include them all instead of combined_condition
    if f'{b_name}_dim_0' in trace:
        trace = trace.rename({f'{b_name}_dim_0': posterior_index_name})
    if f'a_subject_dim_0' in trace:
        trace = trace.rename({f'a_subject_dim_0': 'subject'})
    if add_data and (df_data[posterior_index_name].dtype != 'int'):
        warnings.warn(
            f"Was {posterior_index_name} a string? It's safer to recast it as integer. I'll try to do that...")
        df_data[posterior_index_name] = df_data[posterior_index_name].astype(int)

    hdi = az.hdi(trace)[b_name]

    # from scipy.stats import mode
    max_a_p = xar_mode(trace[b_name], dims_to_reduce=['chain', 'draw'])
    # Or mean trace[b_name].mean(['chain', 'draw']).values,
    if hdi.ndim == 1:
        mean = xr.DataArray([max_a_p],
                            coords={'hdi': ["center"], },
                            dims='hdi')
        df_bayes = xr.concat([hdi, mean], 'hdi').to_dataframe().reset_index()
        df_bayes = df_bayes.pivot_table(columns='hdi').reset_index(drop=True)
        if not df_bayes.columns.str.contains('interval').any():
            # This may be always?
            df_bayes.columns += ' interval'
        if not add_data:  # Done
            return df_bayes, trace
        return hdi2df_one_condition(df_bayes, posterior_index_name, df_data), trace
    else:
        est = xr.DataArray([max_a_p],
                           coords={'hdi': ["center"], posterior_index_name: hdi[posterior_index_name]},
                           dims=['hdi', posterior_index_name])

        df_bayes = xr.concat([hdi, est], 'hdi').rename('interval').to_dataframe()
        df_bayes = df_bayes.pivot_table(index=posterior_index_name, columns=['hdi', ]).reset_index()
        # Reset 2-level column from pivot_table:
        df_bayes.columns = [" ".join(np.flip(pair)) for pair in df_bayes.columns]
        df_bayes.rename({f' {posterior_index_name}': posterior_index_name}, axis=1, inplace=True)
        if not add_data:  # Done
            return df_bayes, trace
        return hdi2df_many_conditions(df_bayes, posterior_index_name, df_data), trace


def make_fold_change(df, y='log_firing_rate', index_cols=('Brain region', 'Stim phase'),
                     treatment_name='stim', treatments=(0, 1), do_take_mean=False, fold_change_method='divide'):
    for treatment in treatments:
        assert treatment in df[treatment_name].unique(), f'{treatment} not in {df[treatment_name].unique()}'
    if y not in df.columns:
        raise ValueError(f'{y} is not a column in this dataset: {df.columns}')
    index_cols = list(index_cols)
    # Take mean of trials:
    if do_take_mean:
        df = df.groupby(index_cols).mean().reset_index()

    # Make multiindex
    mdf = df.set_index(index_cols).copy()
    if (mdf.xs(treatments[1], level=treatment_name).size !=
        mdf.xs(treatments[0], level=treatment_name).size):
        raise IndexError(f'Uneven number of entries in conditions! Try setting do_take_mean=True'
                         f'{mdf.xs(treatments[0], level=treatment_name).size, mdf.xs(treatments[1], level=treatment_name).size}')

    # Subtract/divide
    try:
        if fold_change_method == 'subtract':
            data = (
                mdf.xs(treatments[1], level=treatment_name)[y] -
                mdf.xs(treatments[0], level=treatment_name)[y]
            ).reset_index()
        else:
            data = (mdf.xs(treatments[1], level=treatment_name)[y] /
                    mdf.xs(treatments[0], level=treatment_name)[y]
                    ).reset_index()
    except Exception as e:
        print(f'Try recasting {treatment_name} as integer and try again. Alternatively, use bayes_window.workflow.'
              f' We do that automatically there ')
        raise e
    y1 = f'{y} diff'
    data.rename({y: y1}, axis=1, inplace=True)
    if np.isnan(data[y1]).all():
        print(f'For {treatments}, data has all-nan {y1}: {data.head()}')
        print(f'Condition 1: {mdf.xs(treatments[1], level=treatment_name)[y].head()}')
        print(f'Condition 2: {mdf.xs(treatments[0], level=treatment_name)[y].head()}')
        raise ValueError(f'For {treatments}, data has all-nan {y1}. Ensure there a similar treatment to {y} does not'
                         f'shadow it!')

    return data, y1

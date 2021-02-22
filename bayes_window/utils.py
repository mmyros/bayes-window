import warnings

import arviz as az
import inflection
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform


def add_data_to_posterior(df_data,
                          trace,
                          y=None,  # Only for fold change
                          index_cols=None,
                          condition_name='Event',
                          conditions=None,  # eg ('stim_on', 'stim_stop')
                          b_name='b_stim_per_condition',  # for posterior
                          group_name='Condition code',  # for posterior
                          do_make_change='subtract',
                          do_mean_over_trials=True,
                          ):
    if type(index_cols) == str:
        index_cols = [index_cols]
    index_cols = list(index_cols)

    conditions = conditions or df_data[condition_name].drop_duplicates().sort_values().values
    assert len(conditions) == 2, f'{condition_name}={conditions}. Should be only two instead!'
    assert do_make_change in [False, 'subtract', 'divide']
    if not (condition_name in index_cols):
        index_cols.append(condition_name)
    if do_mean_over_trials:
        df_data = df_data.groupby(index_cols).mean().reset_index()
    if do_make_change:
        # Make (fold) change
        df_data, y = make_fold_change(df_data,
                                      y=y,
                                      index_cols=index_cols,
                                      condition_name=condition_name,
                                      conditions=conditions,
                                      fold_change_method=do_make_change,
                                      do_take_mean=False)
        # Condition is removed from both index columns and dfbayes
        index_cols.remove(condition_name)
        # df_bayes = df_bayes.drop(condition_name, axis=1, errors='ignore')
    # Convert to dataframe and fill in data:
    df_bayes = trace2df(trace, df_data, b_name=b_name, group_name=group_name)
    return df_bayes


def fill_row(group_val, rows, df_bayes, group_name):
    this_hdi = df_bayes.loc[df_bayes[group_name] == group_val]
    for col in ['lower HDI', 'higher HDI', 'mean HDI']:
        rows.insert(rows.shape[1] - 1, col, this_hdi[col].values.squeeze())
    return rows


def hdi2df_many_conditions(trace, hdi, b_name, group_name, df_data):
    hdi = hdi.rename({f'{b_name}_dim_0': group_name})
    mean = xr.DataArray([trace[b_name].mean(['chain', 'draw']).values],
                        coords={'hdi': ["mean"], group_name: hdi[group_name]},
                        dims=['hdi', group_name])
    df_bayes = xr.concat([hdi, mean], 'hdi').rename('HDI').to_dataframe()
    df_bayes = df_bayes.pivot_table(index=group_name, columns=['hdi', ]).reset_index()
    # Reset 2-level column from pivot_table:
    df_bayes.columns = [" ".join(np.flip(pair)) for pair in df_bayes.columns]
    df_bayes.rename({f' {group_name}': group_name}, axis=1, inplace=True)
    rows = [fill_row(group_val, rows, df_bayes, group_name) for group_val, rows in df_data.groupby([group_name])]
    return pd.concat(rows)


def hdi2df_one_condition(trace, hdi, b_name, group_name, df_data):
    mean = xr.DataArray([trace[b_name].mean(['chain', 'draw']).values],
                        coords={'hdi': ["mean"], },
                        dims='hdi')
    df_bayes = xr.concat([hdi, mean], 'hdi').to_dataframe().reset_index()
    df_bayes = df_bayes.pivot_table(columns='hdi').reset_index(drop=True)
    df_bayes[group_name] = df_data[group_name].iloc[0]
    df_bayes.columns += ' HDI'
    for col in ['lower HDI', 'higher HDI', 'mean HDI']:
        df_data.insert(df_data.shape[1] - 1, col, df_bayes[col].values.squeeze())
    return df_data


def trace2df(trace, df_data, b_name='b_stim_per_condition', group_name='condition_code'):
    """
    # Convert to dataframe and fill in original conditions
    :param trace:
    :param df_data:
    :param b_name:
    :return:
    """

    # TODO this can be done by simply using Dataset.replace({})
    if df_data[group_name].dtype != 'int':
        warnings.warn(f"Was {group_name} a string? It's safer to recast it as integer. I'll try to do that...")
        df_data[group_name] = df_data[group_name].astype(int)

    hdi = az.hdi(trace)[b_name]
    if hdi.ndim == 1:
        return hdi2df_one_condition(trace, hdi, b_name, group_name, df_data)
    else:
        return hdi2df_many_conditions(trace, hdi, b_name, group_name, df_data)


def make_fold_change(df, y='log_firing_rate', index_cols=('Brain region', 'Stim phase'),
                     condition_name='stim', conditions=(0, 1), do_take_mean=False, fold_change_method='divide'):
    for condition in conditions:
        assert condition in df[condition_name].unique(), f'{condition} not in {df[condition_name].unique()}'
    if y not in df.columns:
        raise ValueError(f'{y} is not a column in this dataset: {df.columns}')

    # Take mean of trials:
    if do_take_mean:
        df = df.groupby(list(index_cols)).mean().reset_index()

    # Make multiindex
    mdf = df.set_index(list(set(index_cols) - {'i_spike'})).copy()
    if (mdf.xs(conditions[1], level=condition_name).size !=
        mdf.xs(conditions[0], level=condition_name).size):
        raise IndexError(f'Uneven number of entries in conditions! Try setting do_take_mean=True'
                         f'{mdf.xs(conditions[0], level=condition_name).size, mdf.xs(conditions[1], level=condition_name).size}')

    # Subtract/divide
    try:
        if fold_change_method == 'subtract':
            data = (mdf.xs(conditions[1], level=condition_name) -
                    mdf.xs(conditions[0], level=condition_name)
                    ).reset_index()
        else:
            data = (mdf.xs(conditions[1], level=condition_name) /
                    mdf.xs(conditions[0], level=condition_name)
                    ).reset_index()
    except Exception as e:
        print(f'Try recasting {condition_name} as integer and try again. Alternatively, use bayes_window.workflow.'
              f' We do that automatically there ')
        raise e
    y1 = f'{y} diff'
    data.rename({y: y1}, axis=1, inplace=True)
    if np.isnan(data[y1]).all():
        print(f'For {conditions}, data has all-nan {y1}: {data.head()}')
        print(f'Condition 1: {mdf.xs(conditions[1], level=condition_name)[y].head()}')
        print(f'Condition 2: {mdf.xs(conditions[0], level=condition_name)[y].head()}')
        raise ValueError(f'For {conditions}, data has all-nan {y1}. Ensure there a similar condition to {y} does not'
                         f'shadow it!')

    y = y1
    return data, y

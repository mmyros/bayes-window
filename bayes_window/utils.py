import warnings

import arviz as az
import inflection
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform


def add_data_to_posterior(df,
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
        index_cols=[index_cols]
    index_cols = list(index_cols)

    conditions = conditions or df[condition_name].drop_duplicates().sort_values().values
    assert len(conditions) == 2, f'{condition_name}={conditions}. Should be only two instead!'
    assert do_make_change in [False, 'subtract', 'divide']
    if not (condition_name in index_cols):
        index_cols.append(condition_name)
    if do_mean_over_trials:
        df = df.groupby(index_cols).mean().reset_index()
    # Convert to dataframe and fill in original conditions:
    df_bayes = trace2df(trace, df, b_name=b_name, group_name=group_name)
    if do_make_change:
        # Make fold change #TODO add option of divide or subtract
        df, y = make_fold_change(df,
                                 y=y,
                                 index_cols=index_cols,
                                 condition_name=condition_name,
                                 conditions=conditions,
                                 fold_change_method=do_make_change,
                                 do_take_mean=False)
        # Condition is removed from both index columns and dfbayes
        index_cols.remove(condition_name)
        df_bayes = df_bayes.drop(condition_name, axis=1, errors='ignore')
    # Set multiindex and combine data with posterior
    if index_cols is not None:
        df_bayes = df_bayes.set_index(index_cols)
        df = df.set_index(index_cols)
    # TODO this misses first rows of data
    df_both = df.append(df_bayes, sort=False).reset_index()
    # This replaces zeros in bayes columns with nans TODO super strange. Try merge in line above instead
    df_both.loc[(df_both.shape[0] - df_bayes.shape[0]):, 'mouse_code'] = np.nan
    # df_both = pd.concat([df, df_bayes])
    return df_both


def humanize_df(df):
    """
    Rename dataframe columns from_underscore To human readable
    :param df: dataframe to humanize
    :return:
    """

    assert (df.shape[0] > 0), 'Dataframe is empty'
    df.columns = [inflection.humanize(col) for col in df.columns]
    return df


def trace2df(trace, df, b_name='b_stim_per_condition', group_name='condition_code'):
    """
    # Convert to dataframe and fill in original conditions
    :param trace:
    :param df:
    :param b_name:
    :return:
    """

    # TODO this can be done by simply using Dataset.replace({})
    if df[group_name].dtype != 'int':
        warnings.warn(f"Was {group_name} a string? It's safer to recast it as integer. I'll try to do that...")
        df[group_name] = df[group_name].astype(int)

    def fill_row(group_val, rows):
        # row = rows#.iloc[0].to_dict()
        this_hdi = df_bayes.loc[df_bayes[group_name] == group_val]
        for col in ['lower HDI', 'higher HDI', 'mean HDI']:
            rows.insert(df.shape[1] - 1, col, this_hdi[col].values.squeeze())
        # try:
        #     row['lower HDI'] = df_bayes.sel(Condition=row[group_name], CI='lower').values
        # except KeyError:  # Sometimes it's string whereas it should be a number
        #     row[group_name] = int(row[group_name])
        #     row['lower HDI'] = df_bayes.sel(Condition=row[group_name], CI='lower').values
        # row['higher HDI'] = df_bayes.sel(Condition=row[group_name], CI='higher').values
        # row['mean HDI'] = df_bayes.sel(Condition=row[group_name], CI='mean').values
        return rows

    hdi = az.hdi(trace)[b_name]
    if hdi.ndim == 1:
        mean = xr.DataArray([trace[b_name].mean(['chain', 'draw']).values
                             ], coords={'hdi': ["mean"], }, dims='hdi')
        df_bayes = xr.concat([hdi, mean], 'hdi').to_dataframe().reset_index()
        df_bayes = df_bayes.pivot_table(columns='hdi').reset_index(drop=True)
        df_bayes.columns += ' HDI'
    else:
        hdi = hdi.rename({f'{b_name}_dim_0': group_name})

        mean = xr.DataArray([trace[b_name].mean(['chain', 'draw']).values
                             ], coords={'hdi': ["mean"], group_name: hdi[group_name]}, dims=['hdi', group_name])
        df_bayes = xr.concat([hdi, mean], 'hdi').rename('HDI').to_dataframe()  # .reset_index('hdi')
        df_bayes = df_bayes.pivot_table(index=group_name, columns=['hdi', ]).reset_index()
        # Reset 2-level column from pivot_table:
        df_bayes.columns = [" ".join(np.flip(pair)) for pair in df_bayes.columns]
        df_bayes.rename({f' {group_name}': group_name}, axis=1, inplace=True)

        # hdi = az.hdi(trace[b_name].values)
        # posterior_mean = trace[b_name].mean(['chain', 'draw']).values[:, None]
        #
        # hpd_condition = xr.DataArray(np.concatenate((hdi, posterior_mean), axis=1),
        #                              dims=['Condition', 'CI'],
        #                              coords=[range(hdi.shape[0]), ['ci_start', 'ci_end', 'mean']])
        #

        rows = [fill_row(group_val, rows) for group_val, rows in df.groupby([group_name])]

        df_bayes = pd.concat(rows)
    # df_bayes = humanize_df(df_bayes)
    return df_bayes


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

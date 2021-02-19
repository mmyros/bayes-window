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
                          index_cols=('Brain region', 'Stim phase', 'Event', 'Fid', 'Subject', 'Inversion'),
                          condition_name='Event',
                          conditions=None,  # eg ('stim_on', 'stim_stop')
                          b_name='b_stim_per_condition',  # for posterior
                          group_name='Condition code',  # for posterior
                          do_make_change=True,
                          do_mean_over_trials=True,
                          ):
    index_cols = list(index_cols)
    if conditions is None:
        conditions = df[condition_name].drop_duplicates().sort_values().values
    assert len(conditions) == 2, f'{condition_name}={conditions}. Should be only two instead!'
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
                                 do_take_mean=False)
        # Condition is removed from both index columns and dfbayes
        index_cols.remove(condition_name)
        df_bayes = df_bayes.drop(condition_name, axis=1, errors='ignore')
    # Set multiindex and combine data with posterior
    df_bayes = df_bayes.set_index(index_cols)
    df = df.set_index(index_cols)
    # TODO this misses first rows of data
    df_both = df.append(df_bayes, sort=False).reset_index()
    # This replaces zeros in bayes columns with nans TODO super strange. Try merge in line above instead
    df_both.loc[(df_both.shape[0] - df_bayes.shape[0]):, 'mouse_code'] = np.nan
    # list(set(index_cols)-{condition_name})
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

    def fill_row(rows):
        row = rows.iloc[0].to_dict()
        try:
            row['Bayes condition CI0'] = hpd_condition.sel(Condition=row[group_name], CI='ci_start').values
        except KeyError:  # Sometimes it's string whereas it should be a number
            row[group_name] = int(row[group_name])
            row['Bayes condition CI0'] = hpd_condition.sel(Condition=row[group_name], CI='ci_start').values
        row['Bayes condition CI1'] = hpd_condition.sel(Condition=row[group_name], CI='ci_end').values
        row['Bayes condition mean'] = hpd_condition.sel(Condition=row[group_name], CI='mean').values
        return row

    posterior = trace[b_name]
    hpd = az.hdi(posterior.values)
    hpd_condition = xr.DataArray(np.concatenate((hpd, posterior.mean(['chain', 'draw']).values[:, None]), axis=1),
                                 dims=['Condition', 'CI'],
                                 coords=[range(hpd.shape[0]), ['ci_start', 'ci_end', 'mean']])

    df_bayes = pd.DataFrame.from_records([fill_row(rows) for i, rows in df.groupby([group_name])])
    # df_bayes = humanize_df(df_bayes)
    return df_bayes


def make_fold_change(df, y='log_firing_rate', index_cols=('Brain region', 'Stim phase'),
                     condition_name='stim', conditions=(0, 1), do_take_mean=False):
    # for index_col in index_cols:
    #     assert type(df[index_col].iloc[0]) != str, f'Make sure {index_col} contains not strings!'
    for condition in conditions:
        assert condition in df[condition_name].unique(), f'{condition} not in {df[condition_name].unique()}'
    if y not in df.columns:
        raise ValueError(f'{y} is not a column in this dataset: {df.columns}')
    if do_take_mean:
        # Take mean of trials:
        df = df.groupby(list(index_cols)).mean().reset_index()
    # Make multiindex
    mdf = df.set_index(list(set(index_cols) - {'i_spike'})).copy()
    # mdf.xs(0, level='stim') - mdf.xs(1, level='stim')
    if (mdf.xs(conditions[1], level=condition_name).size !=
        mdf.xs(conditions[0], level=condition_name).size):
        raise IndexError(f'Uneven number of entries in conditions! Try setting do_take_mean=True'
                         f'{mdf.xs(conditions[0], level=condition_name).size, mdf.xs(conditions[1], level=condition_name).size}')

    # Subtract/divide
    try:
        data = (mdf.xs(conditions[1], level=condition_name) -
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

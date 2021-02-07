import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
import inflection
from sklearn.preprocessing import LabelEncoder
trans = LabelEncoder().fit_transform


def add_data_to_posterior(df,
                          trace,
                          y='Coherence magnitude at',
                          index_cols=['Brain region', 'Stim phase', 'Event', 'Fid', 'Subject', 'Inversion'],
                          condition_name='Event',
                          conditions=['stim_on', 'stim_stop'],
                          b_name='b_stim_per_condition',  # for posterior
                          group_name='Condition code'  # for posterior
                          ):
    # Make fold change #TODO add option of divide or subtract
    df = df.groupby(index_cols).mean().reset_index()
    delta_y, y = make_fold_change(df,
                                  y=y,
                                  index_cols=index_cols,
                                  condition_name=condition_name,
                                  conditions=conditions,
                                  do_take_mean=False)

    # Convert to dataframe and fill in original conditions:
    df_bayes = trace2df(trace, df, b_name=b_name, group_name=group_name)
    df_bayes = df_bayes.drop(condition_name, axis=1, errors='ignore')  # TODO move to trace2df

    # Set multiindex and combine data with posterior
    index_cols = list(set(index_cols) - set(condition_name)) # pop might error
    df_bayes = df_bayes.set_index(index_cols)
    delta_y = delta_y.set_index(index_cols)
    df_both = df_bayes.append(delta_y, sort=False).reset_index()
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
        row = rows.iloc[0]
        try:
            row['Bayes condition CI0'] = hpd_condition.sel(Condition=row[group_name], CI='ci_start').values
        except KeyError:  # Sometimes it's string whereas it should be a number
            row[group_name] = int(row[group_name])
        row['Bayes condition CI1'] = hpd_condition.sel(Condition=row[group_name], CI='ci_end').values
        row['Bayes condition mean'] = hpd_condition.sel(Condition=row[group_name], CI='mean').values
        return row

    posterior = trace[b_name]
    hpd = az.hpd(posterior.values.squeeze())
    hpd_condition = xr.DataArray(np.concatenate((hpd, posterior.mean(['chain', 'draw']).values[:, None]), axis=1),
                                 dims=['Condition', 'CI'],
                                 coords=[range(hpd.shape[0]), ['ci_start', 'ci_end', 'mean']])

    df_bayes = pd.DataFrame.from_records([fill_row(rows) for i, rows in df.groupby([group_name])])
    # df_bayes = humanize_df(df_bayes)
    return df_bayes


def make_fold_change(df, y='log_firing_rate', index_cols=('Brain region', 'Stim phase'),
                     condition_name='stim', conditions=(0, 1), do_take_mean=False):
    if do_take_mean:
        # Take mean of trials:
        df = df.groupby(index_cols).mean().reset_index()
    # Make multiindex
    mdf = df.set_index(list(set(index_cols) - {'i_spike'})).copy()
    # mdf.xs(0, level='stim') - mdf.xs(1, level='stim')
    if (mdf.xs(conditions[0], level=condition_name).size !=
        mdf.xs(conditions[1], level=condition_name).size):
        raise IndexError(f'Uneven number of entries in conditions! Try setting do_take_mean=True'
                         f'{mdf.xs(conditions[0], level=condition_name).size, mdf.xs(conditions[1], level=condition_name).size}')

    # Subtract/divide
    data = (mdf.xs(conditions[0], level=condition_name) -
            mdf.xs(conditions[1], level=condition_name)
            ).reset_index()
    y1 = f'{y.split(" ")[0]} diff'
    data.rename({y: y1}, axis=1, inplace=True)
    y = y1
    return data, y

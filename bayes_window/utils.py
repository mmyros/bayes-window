import warnings
from importlib import reload
from pdb import set_trace
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xarray as xr
from joblib import Parallel, delayed

# Works:

import numpyro
from jax import random
import jax.numpy as jnp
import numpyro.distributions as dist

from numpyro.infer import MCMC, NUTS, Predictive, log_likelihood

import arviz as az

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform
import inflection


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
        # row['bayes_condition_CI0'] = hpd_condition.sel(Condition=row[group_name], CI='ci_start').values
        # row['bayes_condition_CI1'] = hpd_condition.sel(Condition=row[group_name], CI='ci_end').values
        row['bayes_condition_CI0'] = hpd_condition.sel(Condition=int(row[group_name]), CI='ci_start').values
        row['bayes_condition_CI1'] = hpd_condition.sel(Condition=int(row[group_name]), CI='ci_end').values
        row['bayes_condition_mean'] = hpd_condition.sel(Condition=int(row[group_name]), CI='mean').values
        return row

    posterior = trace[b_name]
    hpd = az.hpd(posterior.values.squeeze())
    hpd_condition = xr.DataArray(np.concatenate((hpd, posterior.mean(['chain', 'draw']).values[:, None]), axis=1),
                                 dims=['Condition', 'CI'],
                                 coords=[range(hpd.shape[0]), ['ci_start', 'ci_end', 'mean']])

    df_bayes = pd.DataFrame.from_records([fill_row(rows) for i, rows in df.groupby([group_name])])
    df_bayes = humanize_df(df_bayes)
    return df_bayes

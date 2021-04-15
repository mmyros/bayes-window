import warnings

import arviz as az
import numpy as np
import pandas as pd
from arviz.plots.plot_utils import calculate_point_estimate
from sklearn.preprocessing import LabelEncoder


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


def add_data_to_posterior(df_data, posterior, y=None, fold_change_index_cols=None, treatment_name='Event',
                          treatments=None, b_name='slope_per_condition', posterior_index_name='',
                          do_make_change='subtract', do_mean_over_trials=True, group_name='subject'):
    # group_name should be conditions
    if type(fold_change_index_cols) == str:
        fold_change_index_cols = [fold_change_index_cols]
    fold_change_index_cols = list(fold_change_index_cols)
    treatments = treatments or df_data[treatment_name].drop_duplicates().sort_values().values

    assert len(treatments) == 2, f'{treatment_name}={treatments}. Should be only two instead!'
    assert do_make_change in [False, 'subtract', 'divide']
    if not (treatment_name in fold_change_index_cols):
        fold_change_index_cols.append(treatment_name)
    if not ('combined_condition' in fold_change_index_cols) and ('combined_condition' in df_data.columns):
        fold_change_index_cols.append('combined_condition')
    # if not (group_name in index_cols):
    #     index_cols.append(group_name)
    if do_mean_over_trials:
        df_data = df_data.groupby(fold_change_index_cols).mean().reset_index()
    if do_make_change:
        # Make (fold) change
        try:
            df_data, _ = make_fold_change(df_data,
                                          y=y,
                                          index_cols=fold_change_index_cols,
                                          treatment_name=treatment_name,
                                          treatments=treatments,
                                          fold_change_method=do_make_change,
                                          do_take_mean=False)
        except Exception as e:
            print(e)
    # Convert to dataframe and fill in data:
    df_bayes, posterior = trace2df(posterior, df_data, b_name=b_name, posterior_index_name=posterior_index_name,
                                   group_name=group_name)
    return df_bayes, posterior


def fill_row(condition_val, data_rows, df_bayes, condition_name):
    # Look up where posterior has the same condition value as in data
    this_hdi = df_bayes.loc[df_bayes[condition_name] == condition_val]
    if this_hdi.shape[0] == 0:
        raise ValueError(f"No such value {condition_val} in estimate's {condition_name}: "
                         f"it's {df_bayes[condition_name].unique()}")
    # Insert posterior into data at the corresponding location
    for col in ['lower interval', 'higher interval', 'center interval']:
        data_rows.insert(data_rows.shape[1] - 1,  # Last column
                         col,  # lower or higher or center interval
                         this_hdi[col].values.squeeze()  # The value of posterior we just looked up
                         )
    return data_rows


def hdi2df_many_conditions(df_bayes, posterior_index_name, df_data):
    # Check
    if len(df_data[posterior_index_name].unique()) != len(df_bayes[posterior_index_name].unique()):
        warnings.warn('Groups were constructed differently for estimation and data. Cant add data for plots')
    rows = [fill_row(group_val, data_rows, df_bayes, posterior_index_name)
            for group_val, data_rows in df_data.groupby([posterior_index_name])]
    return pd.concat(rows)


def hdi2df_one_condition(df_bayes, df_data):
    # df_bayes[group_name] = df_data[group_name].iloc[0]
    for col in ['lower interval', 'higher interval', 'center interval']:
        df_data.insert(df_data.shape[1], col, df_bayes[col].values.squeeze())
    return df_data


def get_hdi_map(posterior, circular=False, prefix=''):
    # HDI and mean over draws (will replace with MAP)
    hdi = az.hdi(posterior).to_dataframe()
    var_name = hdi.columns[-1]
    if posterior.ndim == 2:
        # Get MAP
        max_a_p = calculate_point_estimate('mode', posterior.values.flatten(), bw="default", circular=circular)

        hdi = pd.DataFrame({f'{prefix}higher interval': hdi.loc['higher', var_name],
                            f'{prefix}lower interval': hdi.loc['lower', var_name],
                            f'{prefix}center interval': max_a_p,
                            },
                           index=[0])
    else:
        # The dimension name, other than draws and chains (eg mouse)
        dim = posterior.dims[-1]
        # Name of the variable we are estimating (eg intercept_per_group)

        m_a_p = posterior.mean(['chain', 'draw']).to_dataframe()

        # Get MAP
        m_a_p[var_name] = [calculate_point_estimate('mode', b.values.flatten(), bw="default", circular=circular)
                           for _, b in posterior.groupby(dim)]

        # Merge HDI and MAP
        hdi = hdi.rename({var_name: 'interval'}, axis=1)
        hdi = hdi.pivot_table(index=dim, columns=['hdi', ])
        # Reset column multiindex: Join 'interval' with 'higher
        hdi.columns = [prefix + (' '.join(np.flip(col))) for col in hdi.columns]
        hdi = hdi.join(m_a_p.rename({var_name: f'{prefix}center interval'}, axis=1)).reset_index()
    return hdi


def trace2df(trace, df_data, b_name='slope_per_condition', posterior_index_name='combined_condition',
             group_name='subject'):
    """
    # Convert to dataframe and fill in original conditions
    group name is whatever was used to index posterior
    """
    # Rename axis names to what they actually represent:
    if f'{b_name}_dim_0' in trace:
        trace = trace.rename({f'{b_name}_dim_0': posterior_index_name})
    if f'intercept_per_group_dim_0' in trace:
        trace = trace.rename({f'intercept_per_group_dim_0': group_name})
    if f'slope_per_group_dim_0' in trace:
        trace = trace.rename({f'slope_per_group_dim_0': f"{group_name}_"})  # underscore so it doesnt conflict
    if posterior_index_name and (df_data[posterior_index_name].dtype != 'int'):
        warnings.warn(f"Was {posterior_index_name} a string? It's safer to recast it as integer. I'll try to do that")
        df_data[posterior_index_name] = df_data[posterior_index_name].astype(int)

    # HDI and MAP:
    # df_bayes = get_hdi_map(trace[b_name])
    df_bayes = pd.concat([get_hdi_map(trace[var], prefix=f'{var} ' if var != b_name else '')
                          for var in trace.data_vars], axis=1)
    # Insert posteriors into data:
    if trace[b_name].ndim == 2:  # No extra dimension, just chains and draws
        return hdi2df_one_condition(df_bayes, df_data), trace
    else:
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
        debug_info = (mdf.xs(treatments[0], level=treatment_name).size,
                      mdf.xs(treatments[1], level=treatment_name).size)
        raise IndexError(
            f'Uneven number of entries in conditions! This will lead to nans in data (window.data[\"{y} diff"'
            f'{debug_info}')
        # Some posteriors wont work then
        # if fold_change_method == 'subtract':
        #     data = (
        #         mdf.xs(treatments[1], )[y] -
        #         mdf.xs(treatments[0], )[y]
        #     ).reset_index()
        # else:
        #     data = (mdf.xs(treatments[1], )[y] /
        #             mdf.xs(treatments[0], )[y]
        #             ).reset_index()
    else:
        # Subtract/divide
        if fold_change_method == 'subtract':
            data = (
                mdf.xs(treatments[1], level=treatment_name)[y] -
                mdf.xs(treatments[0], level=treatment_name)[y]
            ).reset_index()
        else:
            data = (mdf.xs(treatments[1], level=treatment_name)[y] /
                    mdf.xs(treatments[0], level=treatment_name)[y]
                    ).reset_index()
    y1 = f'{y} diff'
    data.rename({y: y1}, axis=1, inplace=True)
    # Some debug
    # if np.isnan(data[y1]).all():
    #     print(f'For {treatments}, data has all-nan {y1}: {data.head()}')
    #     print(f'Condition 1: {mdf.xs(treatments[1], level=treatment_name)[y].head()}')
    #     print(f'Condition 2: {mdf.xs(treatments[0], level=treatment_name)[y].head()}')
    #     raise ValueError(f'For {treatments}, data has all-nan {y1}. Ensure there a similar treatment to {y} does not'
    #                      f'shadow it!')

    return data, y1


def scrub_lme_result(result, include_condition, condition, data, treatment):
    res = result.summary().tables[1]
    if include_condition:
        # Only take relevant estimates
        res = res.loc[[index for index in res.index
                       if (condition[:-3] in index)]]
        if res.shape[0] > len(data[condition].unique()):
            # If  conditionals are included, remove non-conditionals
            res = res.loc[[index for index in res.index
                           if (condition[:-3] in index)
                           # if (index[:len(condition)] == condition)
                           and ('|' in index)]]
        # Restore condition names
        res[condition] = [data[condition].unique()[i] for i, index in enumerate(res.index)]
        res = res.reset_index(drop=True).set_index(condition)  # To prevent changing condition to float below
    else:
        # Only take relevant estimates
        res = res.loc[[index for index in res.index if (res.loc[index, 'z'] != '') & (treatment in index)]]
    try:
        res = res.astype(float)  # [['P>|z|', 'Coef.', '[0.025', '0.975]']]
    except Exception as e:
        warnings.warn(f'somehow LME failed to estimate CIs for one or more variables. Replacing with nan:'
                      f' {e} \n=>\n {res}')
        res.replace({'': np.nan}).astype(float)
    if res.shape[0] == 0:
        print(result.summary().tables[1])
        print(res)
        # import pdb;pdb.set_trace()

    assert res.shape[0] > 0, 'There is no result'
    res = res.reset_index()
    res = res.rename({'P>|z|': 'p',
                      'Coef.': 'center interval',
                      '[0.025': 'higher interval',
                      '0.975]': 'lower interval'}, axis=1)
    return res


def add_data_to_lme(do_make_change, include_condition, res, condition, data, y, levels, treatment):
    # Sanity check:
    if not include_condition:
        if not res['index'].str.contains(treatment).any():
            raise KeyError(f'No {treatment}-containing row in :\n{res}')
    if do_make_change and include_condition:
        # Make (fold) change
        change_data, _ = make_fold_change(data,
                                          y=y,
                                          index_cols=levels,
                                          treatment_name=treatment,
                                          fold_change_method=do_make_change,
                                          do_take_mean=False)

        # like in hdi2df:
        rows = [fill_row(group_val, rows, res, condition_name=condition)
                for group_val, rows in change_data.groupby([condition])]
        data_and_posterior = pd.concat(rows)
    elif include_condition:
        # like in hdi2df:
        rows = [fill_row(group_val, rows, res, condition_name=condition)
                for group_val, rows in data.groupby([condition])]
        data_and_posterior = pd.concat(rows)
    elif do_make_change:
        # Make (fold) change
        change_data, _ = make_fold_change(data,
                                          y=y,
                                          index_cols=levels,
                                          treatment_name=treatment,
                                          fold_change_method=do_make_change,
                                          do_take_mean=False)
        # like in hdi2df_one_condition():
        data_and_posterior = change_data.copy()
        for col in ['lower interval', 'higher interval', 'center interval']:
            data_and_posterior.insert(data_and_posterior.shape[1],
                                      col,
                                      res.loc[res['index'].str.contains(treatment), col]
                                      )
    else:
        # like in hdi2df_one_condition():
        data_and_posterior = data.copy()
        for col in ['lower interval', 'higher interval', 'center interval']:
            data_and_posterior.insert(data.shape[1],
                                      col,
                                      res.loc[res['index'].str.contains(treatment), col])

    return data_and_posterior


def combined_condition(df: pd.DataFrame, conditions: list):
    # String-valued combined condition
    # df['combined_condition'] = utils.df_index_compress(df, index_columns=self.levels)[1]
    if conditions[0] is None:
        df['combined_condition'] = np.ones(df.shape[0])
        return df, None

    df['combined_condition'] = df[conditions[0]].astype('str')
    for level in conditions[1:]:
        df['combined_condition'] += df[level].astype('str')
    data = df.copy()
    # Transform conditions to integers as required by numpyro:
    labeler = LabelEncoder()
    data['combined_condition'] = labeler.fit_transform(df['combined_condition'])

    # Keep key for later use
    key = dict()
    for level in conditions:
        key[level] = dict(zip(range(len(labeler.classes_)), labeler.classes_))
    return data, key


def load_radon():
    # !pip install pymc3
    import numpy as np
    import pandas as pd
    # import pymc3 as pm
    # Import radon data
    try:
        srrs2 = pd.read_csv("../docs/example_notebooks/radon_example/srrs2.dat", error_bad_lines=False)
    except FileNotFoundError:
        try:
            srrs2 = pd.read_csv("docs/example_notebooks/radon_example/srrs2.dat", error_bad_lines=False)
        except FileNotFoundError:
            srrs2 = pd.read_csv("srrs2.dat", error_bad_lines=False)
    srrs2.columns = srrs2.columns.map(str.strip)
    srrs_mn = srrs2[srrs2.state == "MN"].copy()

    # Next, obtain the county-level predictor, uranium, by combining two variables.

    srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
    try:
        cty = pd.read_csv("../docs/example_notebooks/radon_example/cty.dat", error_bad_lines=False)
    except FileNotFoundError:
        try:
            cty = pd.read_csv("docs/example_notebooks/radon_example/cty.dat", error_bad_lines=False)
        except FileNotFoundError:
            cty = pd.read_csv("cty.dat", error_bad_lines=False)
    cty_mn = cty[cty.st == "MN"].copy()
    cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips

    # Use the merge method to combine home- and county-level information in a single DataFrame.

    srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
    srrs_mn = srrs_mn.drop_duplicates(subset="idnum")
    u = np.log(srrs_mn.Uppm).unique()

    n = len(srrs_mn)

    srrs_mn.head()

    srrs_mn.county = srrs_mn.county.map(str.strip)
    mn_counties = srrs_mn.county.unique()
    counties = len(mn_counties)
    county_lookup = dict(zip(mn_counties, range(counties)))

    # Finally, create local copies of variables.

    county = srrs_mn["county_code"] = srrs_mn.county.replace(county_lookup).values
    radon = srrs_mn.activity
    srrs_mn["log_radon"] = log_radon = np.log(radon + 0.1).values
    floor = srrs_mn.floor.values

    return pd.DataFrame({'county': county, 'radon': radon, 'floor': floor})

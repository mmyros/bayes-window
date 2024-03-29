import warnings

import arviz as az
import numpy as np
import pandas as pd
from arviz.plots.plot_utils import calculate_point_estimate
from sklearn.preprocessing import LabelEncoder
import xarray as xr


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


def fill_row(condition_val, data_rows, df_bayes, condition_name):
    """

    Parameters
    ----------
    condition_val: Current value of condition_name to look up in df_bayes
    data_rows: rows to fill with corresponding entries from df_bayes (will be concated later)
    df_bayes: dataframe to fill from
    condition_name: eg combined_condition

    Returns
    -------

    """
    # Look up where posterior has the same condition value as in data
    # index = np.ones(df_bayes.shape[0])
    # for cond,val in df_bayes.groupby(condition_name):
    #     index=index&(df_bayes[cond] == val)
    this_hdi = df_bayes.loc[df_bayes[condition_name] == condition_val]
    if this_hdi.shape[0] == 0:
        raise ValueError(f"No such value {condition_val} in estimate's {condition_name}: "
                         f"it's {df_bayes[condition_name].unique()}")
    # Insert posterior into data at the corresponding location
    for col in df_bayes.columns[df_bayes.columns.str.contains('interval')]:
        data_rows.insert(data_rows.shape[1] - 1,  # Insert into the Last column
                         col,  # lower or higher or center interval
                         this_hdi[col].values.squeeze()  # The value of posterior we just looked up
                         )
    return data_rows


def get_hdi_map(posterior, circular=False, prefix=''):
    # HDI and mean over draws
    hdi95 = az.hdi(posterior, hdi_prob=0.95).to_dataframe()
    hdi75 = az.hdi(posterior, hdi_prob=0.75).to_dataframe()
    var_name = hdi95.columns[-1]
    if posterior.ndim == 2:
        # Get MAP

        max_a_p = calculate_point_estimate('auto', posterior.values.flatten(), bw="default", circular=circular)

        hdi95 = pd.DataFrame({f'{prefix}higher interval': hdi95.loc['higher', var_name],
                              f'{prefix}lower interval': hdi95.loc['lower', var_name],
                              f'{prefix}center interval': max_a_p,
                              },
                             index=[0])
    else:
        # The dimension name, other than draws and chains (eg mouse)
        # Take the last dimension name. If there is more than one dimension in the data, this may be problematic
        dims = posterior.dims
        # dims = list(set(dims) - {'chain', 'draw'})
        dim = [dim for dim in dims if ('_dim_' not in dim  # and dim[-1] != '_'
                                       ) and dim not in ['chain', 'draw']][-1]
        #        if len(posterior.dims) > 3:
        #            print(f"Untransformed dimension in {[dim for dim in dims if dim not in ['chain', 'draw']]} may be "
        #                  f"a problem. If you made a new numpyro model, look in utils.rename_posterior() ")
        #            print(posterior)
        #            print(dims)
        #            print(dim)
        #            return

        # Name of the variable we are estimating (eg intercept_per_group)

        m_a_p = posterior.mean(['chain', 'draw']).to_dataframe()

        # Get MAP

        for _, b in posterior.groupby(dim):
            try:
                m_a_p[var_name] = calculate_point_estimate('auto', b.values.flatten(), bw="default", circular=circular)
            except ValueError:
                print(posterior)
                print(posterior.values)
                print(posterior.values.flatten())
        m_a_p[var_name] = [calculate_point_estimate('mode', b.values.flatten(), bw="default", circular=circular)
                           for _, b in posterior.groupby(dim)]

        # Merge HDI and MAP
        hdi95 = hdi95.rename({var_name: 'interval'}, axis=1)
        hdi95 = hdi95.pivot_table(index=dim, columns=['hdi', ])
        # Reset column multiindex: Join 'interval' with 'higher
        hdi95.columns = [prefix + (' '.join(np.flip(col))) for col in hdi95.columns]

        # Repeat for 75% HDI:
        hdi75 = hdi75.rename({var_name: 'interval75'}, axis=1)
        hdi75 = hdi75.pivot_table(index=dim, columns=['hdi', ])
        # Reset column multiindex: Join 'interval' with 'higher
        hdi75.columns = [prefix + (' '.join(np.flip(col))) for col in hdi75.columns]

        # 95% HDI and MAP:
        hdi95 = hdi95.join(m_a_p.rename({var_name: f'{prefix}center interval'}, axis=1)).reset_index()
        # Add 75% HDI:
        hdi95 = hdi95.join(hdi75)
    return hdi95


# def recode(posterior, levels, data, original_data, condition):
#     """ This is really a replace operation on a dataframe. """
#     for column in levels + ['combined_condition']:
#         if column not in posterior.columns:
#             continue
#
#         if column == 'combined_condition':
#             original_columns = condition
#         else:
#             original_columns = [column]
#
#
#         # loop over values in posterior variable (column of df)
#         # and replace posterior values with original values
#         # TODO should be just looking up self.original_labels
#         if False:
#             for val in posterior[column].unique():
#                 original_val = original_data.loc[original_data_index, original_columns].iloc[0]
#                 posterior[column]=posterior[column].replace({val: original_val},axis=1)
#         else:
#             for i, val in posterior[column].iteritems():
#                 original_data_index = data[data[column] == val].index
#                 # Where were original locations?
#                 original_vals = original_data.loc[original_data_index, original_columns]
#                 if original_vals.shape[0] == 0:
#                     continue
#                 # for col in original_vals.columns:
#                 #     assert original_vals[col].unique().size < 2, f'non-unique {col} in: {original_vals}'
#                 posterior.loc[i, original_columns] = original_vals.iloc[0].values
#     return posterior

def replace_in_xarray(xar, index_var, key):
    ddf = xar.to_dataframe()
    index_cols = ddf.index.names
    ddf = ddf.reset_index()
    ddf[index_var] = ddf[index_var].replace(key)
    return ddf.set_index(index_cols).to_xarray()


def recode_posterior(posterior, levels, original_label_values):
    for index_var in set(levels):
        if index_var not in original_label_values.keys():
            continue
        key = original_label_values[index_var]
        # key_as_str = {str(old_val): new_val for old_val, new_val in key.items()}
        # from pdb import set_trace; set_trace()

        if (type(posterior) == xr.Dataset) | (type(posterior) == xr.DataArray):
            if index_var not in posterior.coords:
                continue
            posterior = replace_in_xarray(posterior, index_var, key)
            # posterior = replace_in_xarray(posterior, index_var, key_as_str)

        else:  # Otherwise, if dataframe:
            if index_var not in posterior.keys():
                # print(f'Not recoding {index_var}: {index_var} not in {posterior.keys()}')
                continue
            posterior[index_var] = posterior[index_var].replace(key)
            # Also try old value as string:
            # try:
            #     posterior[index_var] = posterior[index_var].replace(key_as_str)
            # except TypeError as e:
            #     print(e)
            #     pass

    return posterior


# def recode_posterior(posteriors, levels, data, original_data, condition):
#     # Recode index variables to their original state
#     return {p_name: recode(posterior, levels, data, original_data, condition)
#             for p_name, posterior in posteriors.items()}

#
# def recode_trace(traces, levels, data, original_data, condition):
#     # Recode index variables to their original state
#     recoded_traces = []
#     for p_name in traces.data_vars:
#         trace = recode(traces[p_name].to_dataframe().reset_index(), levels, data, original_data, condition)
#         try:
#             trace_xar = trace.set_index(list(set(trace.columns)-{p_name} -{'combined_condition'} )).to_xarray()
#         except ValueError as e:
#             print(e, trace.columns, p_name)
#             try:
#                 trace_xar = trace.set_index(list(set(trace.columns)-{p_name})).to_xarray()
#             except ValueError as e:
#                 print(e, f'. Giving up on {p_name}')
#                 continue
#         recoded_traces.append(trace_xar[p_name])
#     return xr.merge(recoded_traces)


def insert_posterior_into_data(posteriors, data, group, group2):
    for posterior_name, posterior in posteriors.items():

        # Remove underscore from get_hdi_map():
        posterior.rename({f'{group}_': group}, axis=1, inplace=True)
        posterior.rename({f'{group2}_': group2}, axis=1, inplace=True)
        posterior.rename({f'combined_condition__': 'combined_condition'}, axis=1, inplace=True)

        # Sanity check

        posterior_index_cols = list(posterior.columns[~posterior.columns.str.contains('interval')])
        posterior_value_cols = list(posterior.columns[posterior.columns.str.contains('interval')])

        if 'combined_condition' in posterior_index_cols and 'combined_condition' not in data.columns:
            # remove combined condition if needed: it's not in data_and_posterior for example
            posterior_index_cols = list(set(posterior_index_cols) - {'combined_condition'})

        if 'zero' in posterior_index_cols:
            posterior_index_cols.remove('zero')
        # assert len(posterior_index_cols) <= 2, f'Should be [combined condition, {group}]. was {posterior_index_cols}'

        if len(posterior_index_cols) == 0:  # No indices, unidimensional
            for posterior_value_col in posterior_value_cols:
                data.loc[data.index[0], posterior_value_col] = posterior.iloc[0][posterior_value_col]
                # print(data.loc[data.index[0], posterior_value_col])
            continue

        # Fill in:
        for posterior_val, subset_posterior in posterior.groupby(posterior_index_cols):
            assert subset_posterior.shape[0] == 1, f'Non-unique! {subset_posterior}'
            subset_posterior = subset_posterior.iloc[0]
            # Take the first time eg 'higher interval' needs to be placed
            data_index = \
                np.where(data[posterior_index_cols] == posterior_val)[0]
            if len(data_index) == 0:
                data_index = np.where(data[posterior_index_cols].dropna(
                    subset=posterior_index_cols).astype(type(posterior_val)) == posterior_val)[0]
            if len(data_index) == 0:
                # print(f'no {posterior_index_cols} in {posterior_name}: {data[posterior_index_cols]}=={posterior_val}')
                continue
            for posterior_value_col in posterior_value_cols:
                data.loc[data.index[data_index[0]], posterior_value_col] = subset_posterior[posterior_value_col]
    return data


def rename_posterior(trace, b_name, posterior_index_name, group_name, treatment_name='treatment', group2_name='group2'):
    # Rename axis names to what they actually represent:
    if f'{b_name}_dim_0' in trace:
        trace = trace.rename({f'{b_name}_dim_0': posterior_index_name})
    if f'{b_name}_per_condition_dim_0' in trace:
        trace = trace.rename({f'{b_name}_per_condition_dim_0': posterior_index_name})
    if 'intercept_per_condition_dim_0' in trace:
        trace = trace.rename(
            {f'intercept_per_condition_dim_0': f"{posterior_index_name}__"})  # underscore so it doesnt conflict
    if 'mu_intercept_per_group_dim_0' in trace:
        trace = trace.rename({'mu_intercept_per_group_dim_0': group_name})
    if 'mu_intercept_per_treatment_dim_0' in trace:
        trace = trace.rename({'mu_intercept_per_treatment_dim_0': treatment_name})
    if 'mu_intercept_per_treatment_dim_1' in trace:
        trace = trace.rename({'mu_intercept_per_treatment_dim_1': 'combined_condition_'})
    if 'mu_intercept_per_group_dim_0' in trace:
        trace = trace.rename({'mu_intercept_per_group_dim_0': group_name})
    if 'slope_per_condition_dim_0' in trace:
        trace = trace.rename({'slope_per_condition_dim_0': f"combined_condition"})  # underscore so it doesnt conflict
    if 'slope_per_group_dim_0' in trace:
        trace = trace.rename({'slope_per_group_dim_0': f"{group_name}_"})  # underscore so it doesnt conflict
    if 'slope_per_group2_dim_0' in trace:
        trace = trace.rename({'slope_per_group2_dim_0': f"{group2_name}_"})  # underscore so it doesnt conflict
    # Check
    var_names = trace.to_dataframe().reset_index().columns
    if var_names.str.contains('_dim_0').any():
        warnings.warn(f'Unhandled dimension {var_names[var_names.str.contains("_dim_0")]}')
    return trace


def make_fold_change(df, y='log_firing_rate', index_cols=('Brain region', 'Stim phase'),
                     treatment_name='stim', treatments=None, do_take_mean=True, fold_change_method='divide'):
    treatments = treatments or df[treatment_name].drop_duplicates().sort_values().values

    # assert len(treatments) == 2, f'{treatment_name}={treatments}. Should be only two instead!'
    assert fold_change_method in [False, 'subtract', 'divide']
    index_cols = list(index_cols)
    if not (treatment_name in index_cols):
        index_cols.append(treatment_name)
    if not ('combined_condition' in index_cols) and ('combined_condition' in df.columns):
        index_cols.append('combined_condition')
    # if not (group_name in index_cols):
    #     index_cols.append(group_name)

    for treatment in treatments:
        assert treatment in df[treatment_name].unique(), f'{treatment} not in {df[treatment_name].unique()}'
    if y not in df.columns:
        raise ValueError(f'{y} is not a column in this dataset: {df.columns}')
    # Take mean of trials:
    if do_take_mean:
        df = df.groupby(list(set(index_cols)
                             # -{treatment_name}
                             )).mean().reset_index()

    # Make multiindex
    mdf = df.set_index(index_cols).copy()
    if (mdf.xs(treatments[1], level=treatment_name).size !=
        mdf.xs(treatments[0], level=treatment_name).size):  # Warning
        debug_info = (mdf.xs(treatments[0], level=treatment_name).size,
                      mdf.xs(treatments[1], level=treatment_name).size)
        # print(f'Careful if overlaying boxplots over posterior: Uneven number of entries in conditions f"{debug_info}"')
        data = mdf.xs(treatments[0], level=treatment_name).reset_index()
    else:  # all good
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


def decode_combined_condition(combined_condition: pd.Series, conditions: list,
                              combined_condition_labeler: LabelEncoder):
    decoded_df = pd.Series(combined_condition_labeler.inverse_transform(combined_condition)).str.split(',', expand=True)
    if len(decoded_df.columns) != len(conditions):
        raise KeyError(f'Combined condition stored in labeler is '
                       f'{combined_condition_labeler.inverse_transform(combined_condition)[0]}, '
                       f'but conditions requested are {conditions}')
    decoded_df.columns = conditions
    return decoded_df


def combined_condition(df: pd.DataFrame, conditions: list):
    # String-valued combined condition
    # df['combined_condition'] = utils.df_index_compress(df, index_columns=self.levels)[1]
    if len(conditions) == 0:
        return df, None
    if conditions[0] is None:
        df['combined_condition'] = np.ones(df.shape[0])
        return df, None

    df['combined_condition'] = df[conditions[0]].astype('str')
    for level in conditions[1:]:
        df['combined_condition'] += (',' + df[level].astype(str))
    data = df.copy()
    # Transform conditions to integers as required by numpyro:
    labeler = LabelEncoder()
    data['combined_condition'] = labeler.fit_transform(df['combined_condition'])

    # Keep key to combined_condition for later use in recode_posterior
    # combined_condition_labeler = labeler
    # combined_condition_key = dict()
    # for level in conditions:
    #     combined_condition_key[level] = dict(zip(range(len(labeler.classes_)), labeler.classes_))
    return data, labeler


def load_radon():
    # Import radon data
    try:
        srrs2 = pd.read_csv("../tests/test_data/srrs2.dat", error_bad_lines=False)
    except FileNotFoundError:
        try:
            srrs2 = pd.read_csv("tests/test_data/srrs2.dat", error_bad_lines=False)
        except FileNotFoundError:
            srrs2 = pd.read_csv("test_data/srrs2.dat", error_bad_lines=False)
    srrs2.columns = srrs2.columns.map(str.strip)
    srrs_mn = srrs2[srrs2.state == "MN"].copy()

    # Next, obtain the county-level predictor, uranium, by combining two variables.

    srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
    try:
        cty = pd.read_csv("../tests/test_data/cty.dat", error_bad_lines=False)
    except FileNotFoundError:
        try:
            cty = pd.read_csv("tests/test_data/cty.dat", error_bad_lines=False)
        except FileNotFoundError:
            cty = pd.read_csv("test_data/cty.dat", error_bad_lines=False)
    cty_mn = cty[cty.st == "MN"].copy()
    cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips

    # Use the merge method to combine home- and county-level information in a single DataFrame.
    srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
    srrs_mn = srrs_mn.drop_duplicates(subset="idnum")
    # srrs_mn.county = srrs_mn.county.map(str.strip)
    # mn_counties = srrs_mn.county.unique()
    # counties = len(mn_counties)
    # county_lookup = dict(zip(mn_counties, range(counties)))

    # Finally, create local copies of variables.

    # county = srrs_mn["county_code"] = srrs_mn.county.replace(county_lookup).values
    # radon = srrs_mn.activity
    # srrs_mn["log_radon"] = np.log(radon + 0.1).values
    # floor = srrs_mn.floor.values

    return pd.DataFrame({'county': srrs_mn.county, 'radon': srrs_mn.activity, 'floor': srrs_mn.floor.values})


def query_posterior(trace, posterior, query=None):
    # Query posterior since we have access to sane conditions there:

    # Restrict if requested:
    if query:
        posterior = posterior.query(query)
        if posterior['combined_condition'].unique().size == 0:
            raise KeyError(f"Your query {query} returned nothing")

    # Use posterior query to Select combined condition:
    trace_post_query = trace.sel(combined_condition=posterior['combined_condition'].unique())
    return trace_post_query

from copy import copy
from typing import List, Any

import altair as alt
import arviz as az
import pandas as pd
import xarray as xr
from sklearn.preprocessing import LabelEncoder

from bayes_window import models, BayesWindow
from bayes_window import utils
from bayes_window import visualization
from bayes_window.fitting import fit_numpyro


class BayesConditions:
    b_name: str
    chart_data_line: alt.Chart
    chart_posterior_kde: alt.Chart
    chart_zero: alt.Chart
    posterior_intercept: alt.Chart
    chart: alt.Chart
    chart_data_boxplot: alt.Chart
    chart_posterior_whiskers: alt.Chart
    chart_posterior_center: alt.Chart
    chart_base_posterior: alt.Chart
    charts_for_facet: List[Any]
    chart_posterior_hdi_no_data: alt.LayerChart
    add_data: bool
    data_and_posterior: pd.DataFrame
    posterior: dict
    trace: xr.Dataset
    mcmc: Any
    independent_axes: bool

    def __init__(self, window=None, add_data=True, **kwargs):
        if type(window) == pd.DataFrame:  # User must want to specify df, not window
            kwargs['df'] = window
            window = None
        window = copy(window) if window is not None else BayesWindow(transform_treatment=True, **kwargs)
        window.add_data = add_data
        self.window = window

    def fit(self, model=models.model_single, fit_fn=fit_numpyro, **kwargs):

        self.model = model
        self.b_name = 'mu_per_condition'

        # add all levels into condition
        # if self.window.group and self.window.group not in self.window.condition:
        #    self.window.condition += [self.window.group]

        if not self.window.condition[0]:
            self.window.condition = [self.window.treatment]
        if self.window.treatment not in self.window.condition:
            self.window.condition += [self.window.treatment]

        # Recode dummy condition taking into account all levels
        # self.window.data, self._key = utils.combined_condition(self.window.original_data.copy(), self.window.condition)

        # Transform group to integers as required by numpyro:
        # if self.window.group:
        #     self.window.data[self.window.group] = LabelEncoder().fit_transform(self.window.data[self.window.group])
        # if self.window.treatment:
        #     self.window.data[self.window.treatment] = LabelEncoder().fit_transform(
        #         self.window.data[self.window.treatment])

        # Estimate model
        self.trace, self.mcmc = fit_fn(y=self.window.data[self.window.y].values,
                                       condition=self.window.data['combined_condition'].values,
                                       group=self.window.data[self.window.group].values if self.window.group else None,
                                       treatment=self.window.data[self.window.treatment].values,
                                       model=model,
                                       **kwargs
                                       )

        # Add data back
        self.trace.posterior = utils.rename_posterior(self.trace.posterior, self.b_name,
                                                      posterior_index_name='combined_condition',
                                                      group_name=self.window.group,
                                                      treatment_name=self.window.treatment
                                                      )

        # HDI and MAP:
        # self.posterior = {var: utils.get_hdi_map(
        #     self.trace.posterior[var],
        #     prefix=f'{var} ' if (var != self.b_name) #and (var not in ['slope_per_condition'])
        #                         and (self.trace.posterior[var] is not None) else '')
        #     for var in self.trace.posterior.data_vars if var not in ['mu_intercept_per_treatment']} \
        self.posterior = {}
        for var in self.trace.posterior.data_vars:
            if var in ['mu_intercept_per_treatment'] or (self.trace.posterior[var] is None):
                continue
            self.posterior[var] = utils.get_hdi_map(self.trace.posterior[var],
                                                    prefix=f'{var} ' if (var != self.b_name) else '')


        # Fill posterior into data
        self.data_and_posterior = utils.insert_posterior_into_data(posteriors=self.posterior,
                                                                   group=self.window.group,
                                                                   group2=self.window.group2,
                                                                   data=self.window.data.copy())
        self.data_and_posterior = utils.recode_posterior(self.data_and_posterior,
                                                         self.window.levels,
                                                         self.window.original_label_values)


        self.posterior['mu_per_condition'] = utils.recode_posterior(self.posterior['mu_per_condition'],
                                                                    self.window.levels,
                                                                    self.window.original_label_values)

        # Decode back combined_condition for posterior:
        for posterior_name in self.posterior.keys():
            # Recode posterior:
            if 'combined_condition' in self.posterior[posterior_name].keys():
                self.posterior[posterior_name] = pd.concat(
                    [self.posterior[posterior_name],
                     utils.decode_combined_condition(
                         combined_condition=self.posterior[posterior_name]['combined_condition'],
                         conditions=self.window.original_label_values.keys(),
                         combined_condition_labeler=self.window.combined_condition_labeler
                     )], axis=1)

            # Recode trace:
            if 'combined_condition' in self.trace.posterior[posterior_name].coords:
                self.trace.posterior[posterior_name] = pd.concat(
                    [self.trace.posterior[posterior_name].to_dataframe().reset_index(),
                     utils.decode_combined_condition(
                         combined_condition=self.trace.posterior[posterior_name].to_dataframe().reset_index()['combined_condition'],
                         conditions=self.window.original_label_values.keys(),
                         combined_condition_labeler=self.window.combined_condition_labeler
                     )], axis=1).set_index(['chain','draw', ] + list(self.window.original_label_values.keys())
                     ).to_xarray()[posterior_name]

        # Make slope from conditions to emulate regression:
        try:
            self.trace.posterior['slope'] = (self.trace.posterior['mu_per_condition'].sel(
                {self.window.treatment: self.trace.posterior['mu_per_condition'][self.window.treatment].max()}) -
                                             self.trace.posterior['mu_per_condition'].sel({
                                                 self.window.treatment:
                                                     self.trace.posterior['mu_per_condition'][
                                                         self.window.treatment].min()}))

            # HDI and MAP for slope:
            self.posterior['slope'] = utils.get_hdi_map(self.trace.posterior['slope'], prefix='slope')
        except (KeyError,) as e:
            print(f"Cant make fake slope :")
            print(e)
        except ValueError:
            print(f"Cant make fake slope from {self.trace.posterior['slope']}")
        return self

    def plot(self,
             x=None,
             add_data=False,
             independent_axes=True,
             color=None,
             detail=':O',
             auto_facet=False,
             **kwargs):
        self.independent_axes = independent_axes
        x = x or self.window.treatment or self.window.condition[0]
        detail = detail or self.window.detail
        if self.window.treatment:
            color = color or self.window.condition[0]
        elif len(self.window.condition) > 1:
            color = color or self.window.condition[1]
        # TODO default for detail

        # Determine wheteher to use self.data_and_posterior or self.posterior
        if not hasattr(self, 'posterior') or self.posterior is None:
            add_data = True  # Otherwise nothing to do
            base_chart = alt.Chart(self.window.data)
            posterior = None
        elif self.window.add_data:
            posterior = self.data_and_posterior
        elif 'mu_per_condition' in self.posterior.keys():
            posterior = self.posterior['mu_per_condition']
        else:
            posterior = self.data_and_posterior


        chart_p = None
        if posterior is not None:

            if 'mu' in self.posterior.keys():
                for key in posterior.columns:
                    if 'interval' in key:
                        posterior[key] += self.posterior['mu']['mu center interval'].iloc[0]
                        # from pdb import set_trace; set_trace()

            base_chart = alt.Chart(posterior)  # TODO self.data_and_posterior is broken
            # Plot posterior
            chart_p = alt.layer(*visualization.plot_posterior(x=x,
                                                              do_make_change=True,
                                                              y_title=f'{self.window.y} estimate',
                                                              base_chart=base_chart,
                                                              color=color,
                                                              **kwargs
                                                              ))
            if not add_data:  # done
                self.chart = chart_p

        if add_data:
            # Make data plot:
            chart_d = visualization.plot_data_slope_trials(x=x,
                                                           y=self.window.y,
                                                           color=color,
                                                           detail=detail,
                                                           base_chart=base_chart)

            if not hasattr(self, 'data_and_posterior') or self.data_and_posterior is None:
                self.chart = chart_d  # we're done
            else:
                self.chart = chart_p + chart_d
        if auto_facet:
            if ('column' in kwargs) or ('row' in kwargs):
                return visualization.facet(self.chart, **kwargs)
            elif len(self.window.condition) > 2:  # Auto facet
                return visualization.facet(self.chart, **visualization.auto_facet(self.window.condition[2]))

        return self.chart

    def forest(self, query='opsin=="chr2" & delay_length==60'):
        trace_post_query = utils.query_posterior(trace=self.trace, posterior=self.posterior, query=query) if query else \
            self.trace.posterior['mu_per_condition']
        az.plot_forest(trace_post_query,
                       combined=True,
                       kind='ridgeplot',
                       ridgeplot_alpha=.5
                       )

    def plot_BEST(self, query=None, rope=(-1, 1), **kwargs):
        """
        eg query = 'opsin=="chr2" & delay_length==60'
        """
        trace_post_query = utils.query_posterior(query, self.trace.posterior) if query else self.trace.posterior
        # TODO querying trace.posterior will have to wait for replacing actual values of index with originals
        # trace_post_query = trace.query()
        az.plot_posterior(
            (trace_post_query.sel(
                {self.window.treatment: self.trace.posterior['mu_per_condition'][self.window.treatment].max()}) -
             trace_post_query.sel(
                 {self.window.treatment: self.trace.posterior['mu_per_condition'][self.window.treatment].min()})),
            'mu_per_condition',
            rope=rope,
            ref_val=0,
            **kwargs
        )

    def explore_models(self, **kwargs):
        from bayes_window.model_comparison import compare_models
        if self.b_name is None:
            raise ValueError('Fit a model first')
        elif self.b_name == 'mu_per_condition':
            return compare_models(df=self.data,
                                  models={
                                      'no_condition': self.model,
                                      'full_normal': self.model,
                                      'full_student': self.model,
                                      'full_lognormal': self.model,
                                  },
                                  extra_model_args=[
                                      {'condition': None},
                                      {'condition': self.condition},
                                      {'condition': self.condition},
                                      {'condition': self.condition},
                                  ],
                                  y=self.y,
                                  parallel=True,
                                  **kwargs
                                  )

    def explore_model_kinds(self, **kwargs):
        from bayes_window.model_comparison import compare_models
        if self.b_name is None:
            raise ValueError('Fit a model first')
        else:
            assert self.b_name == 'mu_per_condition'
            return compare_models(df=self.window.data,
                                  models={
                                      'no_condition': self.model,
                                  },
                                  extra_model_args=[
                                      {'condition': None},
                                  ],
                                  y=self.window.y,
                                  parallel=True,
                                  **kwargs
                                  )

    def plot_model_quality(self, var_names=None, **kwargs):
        assert hasattr(self, 'trace'), 'Run bayesian fitting first!'
        az.plot_trace(self.trace, var_names=var_names, show=True, **kwargs)
        az.plot_pair(
            self.trace,
            var_names=var_names,
            kind="hexbin",
            # coords=coords,
            colorbar=False,
            divergences=True,
            # backend="bokeh",
        )

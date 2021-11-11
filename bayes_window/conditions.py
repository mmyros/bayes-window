from typing import List, Any

import altair as alt
import arviz as az
import pandas as pd
import xarray as xr
from bayes_window import models, BayesWindow
from bayes_window import utils
from bayes_window import visualization
from bayes_window.fitting import fit_numpyro
from sklearn.preprocessing import LabelEncoder


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
    independent_axes: bool

    def __init__(self, window=None, add_data=False, **kwargs):
        window = window or BayesWindow(**kwargs)
        window.add_data = add_data
        self.window = window

    def fit(self, model=models.model_single, fit_fn=fit_numpyro, **kwargs):

        self.model = model
        self.b_name = 'mu_per_condition'

        # add all levels into condition
        # if self.window.group and self.window.group not in self.window.condition:
        #    self.window.condition += [self.window.group]
        if self.window.treatment not in self.window.condition:
            self.window.condition += [self.window.treatment]

        # Recode dummy condition taking into account all levels
        self.window.data, self._key = utils.combined_condition(self.window.original_data.copy(), self.window.condition)

        # Transform group to integers as required by numpyro:
        if self.window.group:
            self.window.data[self.window.group] = LabelEncoder().fit_transform(self.window.data[self.window.group])
        if self.window.treatment:
            self.window.data[self.window.treatment] = LabelEncoder().fit_transform(
                self.window.data[self.window.treatment])

        # Estimate model
        self.trace = fit_fn(y=self.window.data[self.window.y].values,
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
        self.posterior = {var: utils.get_hdi_map(
            self.trace.posterior[var],
            prefix=f'{var} ' if (var != self.b_name) and
                                (var not in ['slope_per_condition']) else '')
            for var in self.trace.posterior.data_vars if var not in ['mu_intercept_per_treatment']}

        # Fill posterior into data
        self.data_and_posterior = utils.insert_posterior_into_data(posteriors=self.posterior,
                                                                   group=self.window.group,
                                                                   group2=self.window.group2,
                                                                   data=self.window.original_data.copy())

        self.posterior = utils.recode_posterior(self.posterior, self.window.levels, self.window.data,
                                                self.window.original_data,
                                                self.window.condition)

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
        else:
            posterior = self.posterior['mu_per_condition']
        chart_p = None
        if posterior is not None:
            base_chart = alt.Chart(posterior)  # TODO self.data_and_posterior is broken
            # Plot posterior
            chart_p = alt.layer(*visualization.plot_posterior(x=x,
                                                              do_make_change=False,
                                                              title=f'{self.window.y} estimate',
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

    def query_posterior(self, query):
        query_combined_condition = self.posterior['mu_per_condition'].query(query)['combined_condition']
        posterior_post_query = self.trace.posterior['mu_per_condition'].sel(
            combined_condition=slice(query_combined_condition.min(),
                                     query_combined_condition.max()))
        return posterior_post_query

    def forest(self, query='opsin=="chr2" & delay_length==60'):
        posterior_post_query = self.query_posterior(query)
        az.plot_forest(posterior_post_query,
                       combined=True,
                       kind='ridgeplot',
                       ridgeplot_alpha=.5
                       )

    def compare_conditions(self, query='opsin=="chr2" & delay_length==60'):
        posterior_post_query = self.query_posterior(query)
        az.plot_posterior(
            posterior_post_query.sel(combined_condition=posterior_post_query['combined_condition'].max()) -
            posterior_post_query.sel(combined_condition=posterior_post_query['combined_condition'].max() - 1),
            rope=(-1, 1),
            ref_val=0
        )

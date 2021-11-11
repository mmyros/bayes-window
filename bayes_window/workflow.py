"""Main module."""
import warnings
from importlib import reload
from typing import List, Any

import altair as alt
import arviz as az
import numpy as np
import pandas as pd
from altair import LayerChart, Chart
from bayes_window import models
from bayes_window import utils
from bayes_window import visualization
from sklearn.preprocessing import LabelEncoder

reload(visualization)
reload(utils)
reload(models)


class BayesWindow:
    chart_zero: Chart
    posterior_intercept: Chart
    chart: Chart
    chart_data_boxplot: Chart
    chart_posterior_whiskers: Chart
    chart_posterior_center: Chart
    chart_base_posterior: Chart
    charts_for_facet: List[Any]
    chart_posterior_hdi_no_data: LayerChart
    add_data: bool

    def __init__(self,
                 df: pd.DataFrame,
                 y: str,
                 treatment: str,
                 condition: str or list = None,
                 group: str = None,
                 group2: str = None,
                 detail=':O',
                 add_data=False
                 ):
        assert y in df.columns
        assert treatment in df.columns
        if group:
            assert group in df.columns
        self.add_data = add_data
        self.treatment = treatment  # if type(treatment)=='list' else [treatment]  # self.levels[2]
        self.group = group  # if type(group)=='list' else [group] # self.levels[1]  # Eg subject
        self.group2 = group2  # if type(group)=='list' else [group] # self.levels[1]  # Eg subject
        self.condition = condition if type(condition) == list else [condition]
        if self.condition[0]:
            assert self.condition[0] in df.columns, f'{self.condition[0]} is not in {df.columns}'
        self.levels = utils.parse_levels(self.treatment, self.condition, self.group, self.group2)

        # Combined condition
        self.data, self._key = utils.combined_condition(df.copy(), self.condition)
        self.original_data = self.data.copy()
        self.detail = detail
        self.y = y

        # Transform conditions to integers as required by numpyro:
        for level in self.levels:
            self.data[level] = LabelEncoder().fit_transform(self.data[level])
        # TODO this is never transformed back at this time? Try using self.original data for plotting

        # Preallocate attributes:
        self.b_name = None  # Depends on model we'll fit
        self.do_make_change = None  # Depends on plotting input
        self.independent_axes = None
        # self.data_and_posterior = None
        # self.posterior = None
        self.trace = None
        self.model = None
        self.group2 = None
        self.model_args = None

        # Preallocate charts
        base_chart = alt.Chart(self.data)

        # Default empty chart:
        empty_chart = base_chart.mark_rule().encode()
        self.chart_posterior_hdi = empty_chart
        self.chart_data_line = empty_chart
        self.chart_posterior_kde = empty_chart
        self.chart_data_box_detail = empty_chart
        self.chart_data_detail = empty_chart
        self.charts = []

        # Some charts of data that don't need fitting
        self.data_box_detail()

    def data_box_detail(self, data=None, color=None, autofacet=False):
        if data is None:
            data = self.data
        y_domain = list(np.quantile(data[self.y], [.05, .95]))
        chart_data_box_for_detail = visualization.plot_data(
            df=data, x=self.treatment, y=self.y, y_domain=y_domain)[0].properties(width=60)

        if (self.detail in self.data.columns) and (len(self.condition) > 1) and not color:
            # Color=condition
            self.chart_data_detail = visualization.plot_data_slope_trials(df=data, x=self.treatment, y=self.y,
                                                                          color=self.condition[1],
                                                                          detail=self.detail,
                                                                          y_domain=y_domain, )
        if self.detail in self.data.columns:
            # default color
            self.chart_data_detail = visualization.plot_data_slope_trials(df=data, x=self.treatment, y=self.y,
                                                                          color=color,
                                                                          detail=self.detail,
                                                                          y_domain=y_domain,
                                                                          )
        else:  # Empty chart; potentially override with new data
            self.chart_data_detail = alt.Chart(data).mark_rule().encode()
        if self.detail in self.data.columns:
            self.chart_data_box_detail = alt.layer(chart_data_box_for_detail, self.chart_data_detail)
        else:
            self.chart_data_box_detail = chart_data_box_for_detail
        if autofacet:
            self.chart_data_box_detail = self.chart_data_box_detail.facet(**visualization.auto_facet(self.group,
                                                                                                     self.condition))
        return self.chart_data_box_detail

    def plot(self, **kwargs):
        # Convenience function
        warnings.warn('No model has been fit. Plotting raw data. Use BayesRegression or LMERegression etc')
        # x = self.levels[0] if 'x' not in kwargs else None
        # color = color or (self.levels[1] if len(self.levels) > 1 else None),
        return visualization.plot_data(self.data,
                                       y=self.y,
                                       **kwargs)[0]

    def facet(self, width=150, height=160, **kwargs):
        assert ('row' in kwargs) or ('column' in kwargs), 'Give facet either row, or column'
        if self.independent_axes is None:
            # TODO let's not force users to plot. have a sensible default
            raise RuntimeError('Plot first, then you can use facet')
        if self.independent_axes:
            facetchart = visualization.facet(self.chart, width=width, height=height, **kwargs)
        else:
            facetchart = self.chart.properties(width=width, height=height).facet(**kwargs)
        return facetchart

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

    def explore_models(self, parallel=True, add_group_slope=False, **kwargs):
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
        elif 'slope' in self.b_name:
            models = {
                'full_normal': self.model,
                'no_condition': self.model,
                'no_condition_or_treatment': self.model,
                'no-treatment': self.model,
                'no_group': self.model,
                'full_student': self.model,
                'full_lognormal': self.model,
                'full_gamma': self.model,
                'full_exponential': self.model,
            }
            extra_model_args = [
                {'treatment': self.treatment, 'condition': self.condition, 'group': self.group},
                {'treatment': self.treatment, 'condition': None},
                {'treatment': None, 'condition': None},
                {'treatment': None, 'condition': self.condition},
                {'treatment': self.treatment, 'condition': self.condition, 'group': None},
                {'treatment': self.treatment, 'condition': self.condition, 'group': self.group, 'dist_y': 'student'},
                {'treatment': self.treatment, 'condition': self.condition, 'group': self.group, 'dist_y': 'lognormal'},
                {'treatment': self.treatment, 'condition': self.condition, 'group': self.group, 'dist_y': 'gamma'},
                {'treatment': self.treatment, 'condition': self.condition, 'group': self.group,
                 'dist_y': 'exponential'},
            ]
            if add_group_slope:
                if self.group is None:
                    raise KeyError('You asked to include group slope. Initalize BayesWindow object with group input')
                models['with_group_slope'] = self.model
                # add_group_slope is False by default in model_hierarchical
                extra_model_args.extend([{'treatment': self.treatment, 'condition': self.condition, 'group': self.group,
                                          'add_group_slope': True}])
            return compare_models(
                df=self.data,
                models=models,
                extra_model_args=extra_model_args,
                y=self.y,
                parallel=parallel,
                **kwargs
            )

    def explore_model_kinds(self, parallel=True, add_group_slope=True, **kwargs):
        from bayes_window.model_comparison import compare_models
        if self.b_name is None:
            raise ValueError('Fit a model first')
        elif self.b_name == 'mu_per_condition':
            return compare_models(df=self.data,
                                  models={
                                      'no_condition': self.model,
                                  },
                                  extra_model_args=[
                                      {'condition': None},
                                  ],
                                  y=self.y,
                                  parallel=True,
                                  **kwargs
                                  )
        elif 'slope' in self.b_name:
            models = {
                'full': self.model,
                'no_condition': self.model,
                'no_condition_or_treatment': self.model,
                'no-treatment': self.model,
                'no_group': self.model,
            }
            extra_model_args = [
                {'treatment': self.treatment, 'condition': self.condition, 'group': self.group},
                {'treatment': self.treatment, 'condition': None},
                {'treatment': None, 'condition': None},
                {'treatment': None, 'condition': self.condition},
                {'treatment': self.treatment, 'condition': self.condition, 'group': None},
            ]
            if add_group_slope and self.group is not None:
                models['with_group_slope'] = self.model
                # add_group_slope is False by default in model_hierarchical
                extra_model_args.extend([{'treatment': self.treatment, 'condition': self.condition, 'group': self.group,
                                          'add_group_slope': True}])
            return compare_models(
                df=self.data,
                models=models,
                extra_model_args=extra_model_args,
                y=self.y,
                parallel=parallel,
                dist_y=self.model_args['dist_y'] if self.model_args else None,
                **kwargs
            )

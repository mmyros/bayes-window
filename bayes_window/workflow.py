"""Main module."""
import warnings
from importlib import reload
from typing import List, Any

import altair as alt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from bayes_window import models
from bayes_window import utils
from bayes_window import visualization

reload(visualization)
reload(utils)
reload(models)


class BayesWindow:
    chart_zero: alt.Chart
    posterior_intercept: alt.Chart
    original_label_values: dict
    chart: alt.Chart
    chart_data_boxplot: alt.Chart
    chart_posterior_whiskers: alt.Chart
    chart_posterior_center: alt.Chart
    chart_base_posterior: alt.Chart
    charts_for_facet: List[Any]
    chart_posterior_hdi_no_data: alt.LayerChart
    add_data: bool

    def __init__(self,
                 df: pd.DataFrame,
                 y: str,
                 treatment: str,
                 condition: str or list = None,
                 group: str = None,
                 group2: str = None,
                 detail=':O',
                 add_data=False,
                 transform_treatment=False
                 ):
        assert y in df.columns
        if df[y].isna().any():
            raise ValueError(f'Outcome variable {y} should not contain nans. Please clean first')
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
        self.levels = list(set(utils.parse_levels(self.treatment, self.condition, self.group, self.group2)))

        # Combined condition
        levels_to_transform = self.levels if transform_treatment else set(self.levels) - {self.treatment}
        self.data, self.combined_condition_labeler = utils.combined_condition(df.copy(), list(levels_to_transform))
        # self.combined_condition_labeler.labels = levels_to_transform
        # if len(self.combined_condition_labeler.classes_[0].split(',')) != len(self.condition):
        #     raise KeyError(f"{self.combined_condition_labeler.classes_[0].split(', ')} but {self.condition}")
        self.original_data = self.data.copy()
        self.detail = detail
        self.y = y

        # Transform conditions to integers as required by numpyro:
        labeler = LabelEncoder()
        self.original_label_values = {}

        # Transform all except treatment if not transform_treatment
        for level in set(levels_to_transform):
            self.data[level] = labeler.fit_transform(self.data[level])
            # Keep key for later use
            self.original_label_values[level] = dict(zip(range(len(labeler.classes_)), labeler.classes_))
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
        try:
            self.data_box_detail()
        except IndexError:
            pass

    def data_box_detail(self, data=None, color=None, autofacet=False):
        if data is None:
            data = self.data
        y_domain = list(np.quantile(data[self.y], [.05, .95]))
        chart_data_box_for_detail = visualization.plot_data(
            df=data, x=self.treatment, y=self.y, y_domain=y_domain)[0][0].properties(width=60)

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
        return alt.layer(*visualization.plot_data(self.data,
                                                  y=self.y,
                                                  **kwargs)[0])

    def facet(self, width=150, height=160, independent_axes=False, **kwargs):
        assert ('row' in kwargs) or ('column' in kwargs), 'Give facet either row, or column'
        if not hasattr(self, 'chart') or self.chart is None:
            # TODO let's not force users to plot. have a sensible default
            raise RuntimeError('Plot first, then you can use facet')
        elif type(self.chart.data) != pd.DataFrame:
            facetchart = visualization.facet(alt.layer(*self.charts_for_facet), width=width, height=height, **kwargs)
        elif independent_axes or type(self.chart) == alt.LayerChart:
            facetchart = visualization.facet(self.chart, width=width, height=height, **kwargs)
        else:
            try:
                facetchart = self.chart.properties(width=width, height=height).facet(**kwargs)
            except ValueError as e:
                assert 'Facet charts require' in str(e)
                facetchart = visualization.facet(self.chart, width=width, height=height, **kwargs)
        return facetchart

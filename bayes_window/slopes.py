import xarray as xr
import warnings
from importlib import reload
from typing import List, Any

import altair as alt
import numpy as np
import pandas as pd

from bayes_window import models, BayesWindow
from bayes_window import utils
from bayes_window import visualization
from bayes_window.fitting import fit_numpyro
from .visualization import plot_posterior


class BayesRegression:
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


    def __init__(self, window=None, add_data=True, **kwargs):
        window = window or BayesWindow(**kwargs)
        window.add_data = add_data
        self.window = window

    def fit(self, model=models.model_hierarchical, do_make_change='subtract', fold_change_index_cols=None,
            do_mean_over_trials=True, fit_method=fit_numpyro, add_condition_slope=True, **kwargs):
        self.window.model_args = kwargs
        if do_make_change not in ['subtract', 'divide', False]:
            raise ValueError(f'do_make_change should be subtract or divide, not {do_make_change}')
        if not add_condition_slope:
            warnings.warn(
                f'add_condition_slope is not requested. Slopes will be the same across {self.window.condition}')
        # if self.b_name is not None:
        #     raise SyntaxError("A model is already present in this BayesWindow object. "
        #                       "Please create a new one by calling BayesWindow(...) again")
        self.window.do_make_change = do_make_change
        self.window.model = model
        if fold_change_index_cols is None:
            fold_change_index_cols = self.window.levels
        fold_change_index_cols = list(fold_change_index_cols)
        if self.window.detail and (self.window.detail in self.window.data.columns) and (
            self.window.detail not in fold_change_index_cols):
            fold_change_index_cols += [self.window.detail]
        if add_condition_slope:
            add_condition_slope = self.window.condition[0] and (
                    np.unique(self.window.data['combined_condition']).size > 1)
            fold_change_index_cols.append('combined_condition')
        self.b_name = 'slope_per_condition' if add_condition_slope else 'slope'
        if add_condition_slope and (not self.window.condition[0] in fold_change_index_cols):
            [fold_change_index_cols.extend([condition]) for condition in self.window.condition
             if not (condition in fold_change_index_cols)]

        # Fit
        self.trace = fit_method(y=self.window.data[self.window.y].values,
                                treatment=self.window.data[self.window.treatment].values,
                                # condition=self.window.data[self.window.condition[0]].values if self.window.condition[0] else None,
                                condition=self.window.data['combined_condition'].values if self.window.condition[
                                    0] else None,
                                group=self.window.data[self.window.group].values if self.window.group else None,
                                model=model,
                                add_condition_slope=add_condition_slope,
                                **kwargs)
        df_data = self.window.data.copy()
        if do_mean_over_trials:
            df_data = df_data.groupby(fold_change_index_cols).mean().reset_index()

        # Make (fold) change
        if do_make_change:
            try:
                df_data, _ = utils.make_fold_change(df_data, y=self.window.y, index_cols=fold_change_index_cols,
                                                    treatment_name=self.window.treatment,
                                                    fold_change_method=do_make_change)
            except Exception as e:
                print(e)

        reload(utils)
        self.trace.posterior = utils.rename_posterior(self.trace.posterior, self.b_name,
                                                      posterior_index_name='combined_condition',
                                                      group_name=self.window.group, group2_name=self.window.group2)

        # HDI and MAP:
        self.posterior = {var: utils.get_hdi_map(self.trace.posterior[var],
                                                 prefix=f'{var} '
                                                 if (var != self.b_name) and (
                                                         var != 'slope_per_condition') else '')
                          for var in self.trace.posterior.data_vars}

        # Fill posterior into data
        self.data_and_posterior = utils.insert_posterior_into_data(posteriors=self.posterior,
                                                                   data=df_data.copy(),
                                                                   group=self.window.group,
                                                                   group2=self.window.group2)

        try:
            self.posterior = utils.recode_posterior(self.posterior, self.window.levels, self.window.data,
                                                    self.window.original_data,
                                                    self.window.condition)
        except Exception as e:
            print(e)

        self.default_regression_charts()
        return self

    def plot(self, x: str = ':O', color: str = ':N', detail: str = ':N', independent_axes=None,
             add_data=None,
             **kwargs):

        # Set some options
        if (x == '') or (x[-2] != ':'):
            x = f'{x}:O'
        if color[-2] != ':':
            color = f'{color}:N'
        if add_data is None:
            add_data = self.window.add_data
        if add_data or self.posterior is None:  # LME
            posterior = self.data_and_posterior
        elif 'slope_per_condition' in self.posterior.keys():
            posterior = self.posterior['slope_per_condition']
        elif 'mu_intercept_per_group' in self.posterior.keys():
            posterior = self.posterior['mu_intercept_per_group']  # TODO fix data_and_posterior
        else:
            posterior = self.data_and_posterior
        if len(x) > 2 and len(posterior[x[:-2]].unique() == 1):
            add_x_axis = True
            x = f'{self.window.condition[0]}:O'
        else:
            add_x_axis = False

        if (x != ':O') and (x != ':N') and x[:-2] in posterior.columns and len(posterior[x[:-2]].unique()) < 10:
            long_x_axis = False
        else:
            long_x_axis = True
            x = f'{x[:-1]}Q'  # Change to quantitative encoding
        # If we are only plotting posterior and not data, independenet axis does not make sense:
        self.window.independent_axes = independent_axes or f'{self.window.y} diff' in posterior
        self.charts = []

        # 1. Plot posterior
        if posterior is not None:
            base_chart = alt.Chart(posterior)
            # Add zero for zero line
            base_chart.data['zero'] = 0

            self.chart_base_posterior = base_chart
            # No-data plot
            (self.chart_posterior_whiskers, self.chart_posterior_whiskers75,
             self.chart_posterior_center, self.chart_zero) = plot_posterior(title=f'{self.window.y}',
                                                                            x=x,
                                                                            base_chart=base_chart,
                                                                            do_make_change=self.window.do_make_change)

            # if no self.data_and_posterior, use self.posterior to build slope per condition:
            if (self.b_name != 'lme') and (type(self.posterior) == dict):
                main_effect = (self.posterior[self.b_name] if self.posterior[self.b_name] is not None
                               else self.posterior['slope_per_condition'])
                self.chart_posterior_hdi_no_data = alt.layer(
                    *plot_posterior(df=main_effect, title=f'{self.window.y}', x=x,
                                    do_make_change=self.window.do_make_change))

            self.chart_posterior_hdi = alt.layer(self.chart_posterior_whiskers, self.chart_posterior_whiskers75,
                                                 self.chart_posterior_center)
            self.charts.append(self.chart_posterior_whiskers)
            self.charts.append(self.chart_posterior_center)
            self.charts.append(self.chart_zero)
            self.charts_for_facet = self.charts.copy()  # KDE cannot be faceted so don't add it
            if (self.b_name != 'lme') and not add_x_axis:
                # Y Axis limits to match self.chart
                minmax = [float(posterior['lower interval'].min()), 0,
                          float(posterior['higher interval'].max())]
                y_domain = [min(minmax), max(minmax)]
                self.chart_posterior_kde = visualization.plot_posterior_density(base_chart, self.window.y, y_domain,
                                                                                self.trace,
                                                                                posterior,
                                                                                self.b_name,
                                                                                do_make_change=self.window.do_make_change)
                self.charts.append(self.chart_posterior_kde)
                # self.charts_for_facet.append(self.chart_posterior_kde) # kde cannot be faceted
        else:
            base_chart = alt.Chart(self.window.data)

        # 2. Plot data
        y = f'{self.window.y} diff'
        if y in posterior:
            if (detail != ':N') and (detail != ':O'):
                assert detail in self.window.data

            # Plot data:
            y_domain = list(np.quantile(base_chart.data[y], [.05, .95]))
            if x != ':O':
                self.chart_data_line, chart_data_points = visualization.line_with_highlight(base_chart, x, y,
                                                                                            color, detail,
                                                                                            highlight=False)
                self.charts.append(self.chart_data_line)
                self.charts.append(chart_data_points)
                self.charts_for_facet.append(chart_data_points)
                self.charts_for_facet.append(self.chart_data_line)

            self.chart_data_boxplot = base_chart.mark_boxplot(
                clip=True, opacity=.3, size=9, color='black',
                median=alt.MarkConfig(color='red', strokeWidth=20)
            ).encode(
                x=x,
                y=alt.Y(f'{y}:Q',
                        axis=alt.Axis(orient='right', title=''),
                        scale=alt.Scale(zero=False, domain=y_domain)
                        )
            )
            self.charts.append(self.chart_data_boxplot)
            self.charts_for_facet.append(self.chart_data_boxplot)
        else:  # No data overlay
            warnings.warn("Did you have Uneven number of entries in conditions? I can't add data overlay")

        # Layer and facet:
        self.chart = visualization.auto_layer_and_facet(
            self.charts, self.charts_for_facet, self.window.independent_axes, **kwargs)
        # self.chart_posterior_hdi_no_data = visualization.auto_layer_and_facet(
        #     self.chart_posterior_hdi_no_data, charts_for_facet=None, independent_axes=self.window.independent_axes, **kwargs)

        # 4. Make overlay for data_detail_plot
        # self.plot_slopes_shading()
        return self.chart

    def plot_slopes_shading(self):  # TODO this method is WIP
        # 0. Use
        pd.concat([utils.get_hdi_map(self.trace.posterior[var], prefix=f'{var} ')
                   for var in self.trace.posterior.data_vars], axis=1)
        # 1. intercepts for stim=1
        self.data_and_posterior['mu_intercept_per_group center interval']
        # 2. slopes+ intercepts
        self.data_and_posterior['intercept'] * self.data_and_posterior['slope']
        # 3. Overlay with
        self.chart_data_detail
        # 4. color by dimension of slope (condition (and group if self.window.group))

    def plot_intercepts(self, x=':O', y='mu_intercept_per_group center interval', **kwargs):
        """
        Plot intercepts of a regression model, mostly for a better understanding of slopes
        Parameters
        ----------
        x
        y
        kwargs

        Returns
        -------

        """
        assert self.posterior is not None
        if self.window.do_make_change:
            # combine posterior with original data instead, not diff TODO
            # Fill posterior into data
            data_and_posterior = utils.insert_posterior_into_data(posteriors=self.posterior,
                                                                  data=self.window.original_data.copy(),
                                                                  group=self.window.group,
                                                                  group2=self.window.group2)
        else:
            data_and_posterior = self.data_and_posterior

        # Redo boxplot (no need to show):
        self.window.data_box_detail(data=data_and_posterior, autofacet=False)

        # Make stand-alone posterior intercept chart:
        self.posterior_intercept = visualization.posterior_intercept_chart(data_and_posterior,
                                                                           x=x, y=y,
                                                                           group=self.window.group)
        # Redo chart_intercept with x=treatment for overlay with self.chart_data_box_detail:
        chart_intercept = visualization.posterior_intercept_chart(data_and_posterior,
                                                                  x=':O', y=y,
                                                                  group=self.window.group)
        chart = alt.layer(chart_intercept, self.window.chart_data_box_detail).resolve_scale(y='independent')

        # Check
        if len(chart.data) == 0:
            raise IndexError('was layer chart from different sources?')
        if ('column' in kwargs) or ('row' in kwargs):
            return visualization.facet(chart, **kwargs)
        else:  # Auto facet
            return visualization.facet(chart, **visualization.auto_facet(self.window.group, self.window.condition))

    def default_regression_charts(self, **kwargs):
        reload(visualization)
        # Default plots:
        # try:
        # facet_kwargs=visualization.auto_facet(self.window.group,self,condition)
        if self.window.condition[0] and len(self.window.condition) > 2:
            try:
                return self.plot(x=self.window.condition[0], column=self.window.condition[1],
                                 row=self.window.condition[2],
                                 **kwargs)
            except KeyError:
                return self.plot(x=self.window.condition[0], row=self.window.condition[1], **kwargs)
        elif self.window.condition[0] and len(self.window.condition) > 1:
            try:
                return self.plot(x=self.window.condition[0], column=self.window.group, row=self.window.condition[1],
                                 **kwargs)
            except KeyError:
                return self.plot(x=self.window.condition[0], row=self.window.condition[1], **kwargs)
        elif self.window.condition[0] and self.b_name != 'lme':
            try:
                return self.plot(x=self.window.condition[0], column=self.window.group, **kwargs)
            except KeyError:
                return self.plot(x=self.window.condition[0], **kwargs)
        else:  # self.window.group:
            return self.plot(x=self.window.condition[0] if self.window.condition[0] else ':O', **kwargs)
        #    self.regression_charts(column=self.window.group)
        # except Exception as e:  # In case I haven't thought of something
        #     print(f'Please use window.regression_charts(): {e}')
        #     # import traceback
        #     # traceback.(e)

    def facet(self, **kwargs):
        return BayesWindow.facet(self, **kwargs)

#     def fit_twostep(self, dist_y_step_one='gamma', **kwargs):
#         if self.detail not in self.window.condition:
#             self.window.condition += [self.detail]
#         window_step_one = self.fit_conditions(dist_y=dist_y_step_one)

#         window_step_two = BayesWindow(window_step_one.posterior['mu_per_condition'],
#                                       y='center interval', treatment=self.treatment,
#                                       condition=list(set(self.window.condition) - {self.treatment, self.window.group, self.detail}),
#                                       group=self.window.group, detail=self.detail)
#         window_step_two.window_step_one = window_step_one
#         window_step_two.fit_slopes(model=models.model_hierarchical,
#                                    **kwargs
#                                    # fold_change_index_cols=('stim', 'mouse', 'neuron_x_mouse')
#                                    )

#         return window_step_two

#     def fit_twostep_by_group(self, dist_y_step_one='gamma', groupby=None, dist_y='student', parallel=True, **kwargs):
#         from joblib import Parallel, delayed
#         assert self.detail is not None

#         def fit_subset(df_m_n, i):
#             window_step_one = BayesWindow(df_m_n, y=self.window.y, treatment=self.treatment,
#                                           condition=[self.detail], group=self.window.group)
#             window_step_one.fit_conditions(dist_y=dist_y_step_one, n_draws=1000, num_chains=1)
#             posterior = window_step_one.posterior['mu_per_condition'].copy()
#             posterior[groupby] = i
#             return posterior

#         groupby = groupby or self.window.condition[0]
#         if parallel:
#             step1_res = Parallel(n_jobs=12, verbose=1)(delayed(fit_subset)(df_m_n, i)
#                                                        for i, df_m_n in self.window.data.groupby(groupby))
#         else:
#             from tqdm import tqdm
#             step1_res = [fit_subset(df_m_n, i) for i, df_m_n in tqdm(self.window.data.groupby(groupby))]

#         window_step_two = BayesWindow(pd.concat(step1_res).rename({'center interval': self.window.y}, axis=1),
#                                       y=self.window.y, treatment=self.treatment,
#                                       condition=list(set(self.window.condition) - {self.treatment, self.window.group, self.detail}),
#                                       group=self.window.group, detail=self.detail)
#         window_step_two.fit_slopes(model=models.model_hierarchical,
#                                    dist_y=dist_y,
#                                    robust_slopes=False,
#                                    add_group_intercept=False,
#                                    add_group_slope=False,
#                                    **kwargs
#                                    )
#         return window_step_two

import warnings
from importlib import reload
from typing import List, Any

import altair as alt
import arviz as az
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from altair import LayerChart, Chart
from sklearn.preprocessing import LabelEncoder

from bayes_window import models
from bayes_window import utils
from bayes_window import visualization
from bayes_window.fitting import fit_numpyro
from bayes_window.model_comparison import compare_models
from bayes_window.visualization import plot_posterior

reload(visualization)
reload(utils)


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

    def __init__(self,
                 df: pd.DataFrame,
                 y: str,
                 treatment: str,
                 condition: str or list = None,
                 group: str = None,
                 detail=':O'
                 ):
        try:
            az.plots.backends.output_notebook(hide_banner=True)
        except Exception as e:
            print("Bokeh not found, it's no big deal")
            print(e)
        assert y in df.columns
        assert treatment in df.columns
        if group:
            assert group in df.columns
        self.treatment = treatment  # if type(treatment)=='list' else [treatment]  # self.levels[2]
        self.group = group  # if type(group)=='list' else [group] # self.levels[1]  # Eg subject
        self.condition = condition if type(condition) == list else [condition]
        if self.condition[0]:
            assert self.condition[0] in df.columns, f'{self.condition[0]} is not in {df.columns}'
        self.levels = utils.parse_levels(self.treatment, self.condition, self.group)

        # Combined condition
        self.data, self._key = utils.combined_condition(df.copy(), self.condition)
        self.original_data = self.data.copy()
        self.detail = detail
        self.y = y

        # Transform conditions to integers as required by numpyro:
        for level in self.levels:
            self.data[level] = LabelEncoder().fit_transform(self.data[level])

        # Preallocate attributes:
        self.b_name = None  # Depends on model we'll fit
        self.do_make_change = None  # Depends on plotting input
        self.independent_axes = None
        self.data_and_posterior = None
        self.posterior = None
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
        self.chart_data_box_for_detail = empty_chart
        self.charts = []

        # Some charts of data that don't need fitting
        self.data_box_detail()

    def data_box_detail(self, data=None, color=None, autofacet=False):
        if data is None:
            data = self.data
        y_domain = list(np.quantile(data[self.y], [.05, .95]))
        self.chart_data_box_for_detail = visualization.plot_data(
            df=data, x=self.treatment, y=self.y, y_domain=y_domain)[0].properties(width=60)
        if (self.detail in self.data.columns) and (len(self.condition) > 1):
            self.chart_data_detail = visualization.plot_data_slope_trials(df=data, x=self.treatment, y=self.y,
                                                                          color=self.condition[1],
                                                                          detail=self.detail,
                                                                          y_domain=y_domain, )
        if self.detail in self.data.columns:
            self.chart_data_detail = visualization.plot_data_slope_trials(df=data, x=self.treatment, y=self.y,
                                                                          color=color,
                                                                          detail=self.detail,
                                                                          y_domain=y_domain,
                                                                          )
        else:  # Empty chart; potentially override with new data
            self.chart_data_detail = alt.Chart(data).mark_rule().encode()
        self.chart_data_box_detail = alt.layer(self.chart_data_box_for_detail, self.chart_data_detail)
        if autofacet:
            self.chart_data_box_detail = self.chart_data_box_detail.facet(**visualization.auto_facet(self.group,
                                                                                                     self.condition))
        return self.chart_data_box_detail

    def fit_anova(self, formula=None, **kwargs):
        from statsmodels.stats.anova import anova_lm
        if self.group:
            # Average over group:
            data = self.data.groupby([self.group, self.treatment]).mean().reset_index()
        else:
            data = self.data
        # dehumanize all columns and variable names for statsmodels:
        [data.rename({col: col.replace(" ", "_")}, axis=1, inplace=True) for col in data.columns]
        self.y = self.y.replace(" ", "_")
        if not formula:
            formula = f'{self.y}~{self.treatment}'
        lm = sm.ols(formula, data=data).fit()
        print(f'{formula}\n {anova_lm(lm, typ=2, **kwargs).round(2)}')
        return anova_lm(lm, typ=2)['PR(>F)'][self.treatment] < 0.05

    def fit_twostep(self, dist_y_step_one='gamma', **kwargs):
        if self.detail not in self.condition:
            self.condition += [self.detail]
        window_step_one = self.fit_conditions(dist_y=dist_y_step_one)

        window_step_two = BayesWindow(window_step_one.posterior['mu_per_condition'],
                                      y='center interval', treatment=self.treatment,
                                      condition=list(set(self.condition) - {self.treatment, self.group, self.detail}),
                                      group=self.group, detail=self.detail)
        window_step_two.window_step_one = window_step_one
        window_step_two.fit_slopes(model=models.model_hierarchical,
                                   **kwargs
                                   # fold_change_index_cols=('stim', 'mouse', 'neuron_x_mouse')
                                   )

        return window_step_two

    def fit_twostep_by_group(self, dist_y_step_one='gamma', groupby=None, dist_y='student', parallel=True, **kwargs):
        from joblib import Parallel, delayed
        assert self.detail is not None

        def fit_subset(df_m_n, i):
            window_step_one = BayesWindow(df_m_n, y=self.y, treatment=self.treatment,
                                          condition=[self.detail], group=self.group)
            window_step_one.fit_conditions(dist_y=dist_y_step_one, n_draws=1000, num_chains=1)
            posterior = window_step_one.posterior['mu_per_condition'].copy()
            posterior[groupby] = i
            return posterior

        groupby = groupby or self.condition[0]
        if parallel:
            step1_res = Parallel(n_jobs=12, verbose=1)(delayed(fit_subset)(df_m_n, i)
                                                       for i, df_m_n in self.data.groupby(groupby))
        else:
            from tqdm import tqdm
            step1_res = [fit_subset(df_m_n, i) for i, df_m_n in tqdm(self.data.groupby(groupby))]

        window_step_two = BayesWindow(pd.concat(step1_res).rename({'center interval': self.y}, axis=1),
                                      y=self.y, treatment=self.treatment,
                                      condition=list(set(self.condition) - {self.treatment, self.group, self.detail}),
                                      group=self.group, detail=self.detail)
        window_step_two.fit_slopes(model=models.model_hierarchical,
                                   dist_y=dist_y,
                                   robust_slopes=False,
                                   add_group_intercept=False,
                                   add_group_slope=False,
                                   **kwargs
                                   )
        return window_step_two

    def fit_lme(self, do_make_change='divide', add_interaction=False, add_data=False, formula=None,
                add_group_intercept=True, add_group_slope=False, add_nested_group=False):
        # model = MixedLM(endog=self.data[self.y],
        #                 exog=self.data[self.condition],
        #                 groups=self.data[self.group],
        #                 # exog_re=exog.iloc[:, 0]
        #                 )
        self.b_name = 'lme'
        # dehumanize all columns and variable names for statsmodels:
        [self.data.rename({col: col.replace(" ", "_")}, axis=1, inplace=True) for col in self.data.columns]
        self.y = self.y.replace(" ", "_")
        self.group = self.group.replace(" ", "_")
        self.treatment = self.treatment.replace(" ", "_")
        self.do_make_change = do_make_change
        include_condition = False  # in all but the following cases:
        if self.condition[0]:
            self.condition[0] = self.condition[0].replace(" ", "_")
            if len(self.condition) > 1:
                self.condition[1] = self.condition[1].replace(" ", "_")
            if len(self.data[self.condition[0]].unique()) > 1:
                include_condition = True
        # condition = None  # Preallocate

        # Make formula
        if include_condition and not formula:
            if len(self.condition) > 1:
                raise NotImplementedError(f'conditions {self.condition}. Use combined_condition')
                # This would need a combined condition dummy variable and an index of condition in patsy:
                # formula = f"{self.y} ~ 1+ {self.condition}(condition_index) | {self.treatment}"

            # Make dummy variables for each level in condition:
            self.data = pd.concat((self.data,
                                   pd.get_dummies(self.data[self.condition[0]],
                                                  prefix=self.condition[0],
                                                  prefix_sep='__',
                                                  drop_first=False)), axis=1)
            dummy_conditions = [cond for cond in self.data.columns
                                if cond[:len(self.condition[0]) + 2] == f'{self.condition[0]}__']
            if add_group_intercept and not add_group_slope and not add_nested_group:
                formula = f"{self.y} ~ (1|{self.group}) + {self.treatment}| {dummy_conditions[0]}"
                # eg 'firing_rate ~ stim|neuron_x_mouse__0 +stim|neuron_x_mouse__1 ... + ( 1 |mouse )'
                for dummy_condition in dummy_conditions[1:]:
                    formula += f" + {self.treatment}|{dummy_condition}"
            elif add_group_intercept and add_group_slope and not add_nested_group:
                formula = (f"{self.y} ~ ({self.treatment}|{self.group}) "
                           f" + {self.treatment}| {dummy_conditions[0]}"
                           )
                for dummy_condition in dummy_conditions[1:]:
                    formula += f" + {self.treatment}|{dummy_condition}"
            elif add_group_intercept and add_group_slope and add_nested_group:
                formula = (f"{self.y} ~ ({self.treatment}|{self.group}) + "
                           f"{self.treatment}| {dummy_conditions[0]}:{self.group}")
                for dummy_condition in dummy_conditions[1:]:
                    formula += f" + {self.treatment}|{dummy_condition}:{self.group}"

            # if add_interaction:
            #     formula += f"+ {condition} * {self.treatment}"

        elif self.group and not formula:
            # Random intercepts and slopes (and their correlation): (Variable | Group)
            formula = f'{self.y} ~  C({self.treatment}, Treatment) + (1 | {self.group})'  # (1 | {self.group}) +
            # Random intercepts and slopes (without their correlation): (1 | Group) + (0 + Variable | Group)
            # formula += f' + (1 | {self.group}) + (0 + {self.treatment} | {self.group})'
        elif not formula:
            formula = f"{self.y} ~ C({self.treatment}, Treatment)"

        print(f'Using formula {formula}')
        result = sm.mixedlm(formula,
                            self.data,
                            groups=self.data[self.group]).fit()
        print(result.summary().tables[1])
        self.data_and_posterior = utils.scrub_lme_result(result, include_condition, self.condition[0], self.data,
                                                         self.treatment)
        if add_data:
            self.data_and_posterior = utils.add_data_to_lme(do_make_change, include_condition, self.posterior,
                                                            self.condition[0], self.data, self.y, self.levels,
                                                            self.treatment)
        self.default_regression_charts()
        return self

    def fit_conditions(self, model=models.model_single, fit_fn=fit_numpyro, **kwargs):

        self.model = model
        self.b_name = 'mu_per_condition'

        # add all levels into condition
        if self.group not in self.condition:
            self.condition += [self.group]
        if self.treatment not in self.condition:
            self.condition += [self.treatment]

        # Recode dummy condition taking into account all levels
        self.data, self._key = utils.combined_condition(self.original_data.copy(), self.condition)
        # Estimate model
        self.trace = fit_fn(y=self.data[self.y].values,
                            condition=self.data['combined_condition'].values,
                            # treatment=self.data[self.treatment].values,
                            model=model,
                            **kwargs
                            )

        # Add data back
        self.trace.posterior = utils.rename_posterior(self.trace.posterior, self.b_name,
                                                      posterior_index_name='combined_condition',
                                                      group_name=self.group)

        # HDI and MAP:
        self.posterior = {var: utils.get_hdi_map(self.trace.posterior[var],
                                                 prefix=f'{var} '
                                                 if (var != self.b_name) and (var != 'slope_per_condition') else '')
                          for var in self.trace.posterior.data_vars}

        # Fill posterior into data
        self.data_and_posterior = utils.insert_posterior_into_data(posteriors=self.posterior,
                                                                   data=self.data.copy(),
                                                                   group=self.group)

        self.posterior = utils.recode_posterior(self.posterior, self.levels, self.data, self.original_data,
                                                self.condition)

        return self

    def fit_slopes(self, model=models.model_hierarchical, do_make_change='subtract', fold_change_index_cols=None,
                   do_mean_over_trials=True, fit_method=fit_numpyro, add_condition_slope=True, **kwargs):
        self.model_args = kwargs
        if do_make_change not in ['subtract', 'divide', False]:
            raise ValueError(f'do_make_change should be subtract or divide, not {do_make_change}')
        if not add_condition_slope:
            warnings.warn(f'add_condition_slope is not requested. Slopes will be the same across {self.condition}')
        if self.b_name is not None:
            raise SyntaxError("A model is already present in this BayesWindow object. "
                              "Please create a new one by calling BayesWindow(...) again")
        self.do_make_change = do_make_change
        self.model = model
        if fold_change_index_cols is None:
            fold_change_index_cols = self.levels
        fold_change_index_cols = list(fold_change_index_cols)
        if self.detail and (self.detail in self.data.columns) and (self.detail not in fold_change_index_cols):
            fold_change_index_cols += [self.detail]
        if add_condition_slope:
            add_condition_slope = self.condition[0] and (np.unique(self.data['combined_condition']).size > 1)
            fold_change_index_cols.append('combined_condition')
        self.b_name = 'slope_per_condition' if add_condition_slope else 'slope'
        if add_condition_slope and (not self.condition[0] in fold_change_index_cols):
            [fold_change_index_cols.extend([condition]) for condition in self.condition
             if not (condition in fold_change_index_cols)]

        # Fit
        self.trace = fit_method(y=self.data[self.y].values,
                                treatment=self.data[self.treatment].values,
                                # condition=self.data[self.condition[0]].values if self.condition[0] else None,
                                condition=self.data['combined_condition'].values if self.condition[0] else None,
                                group=self.data[self.group].values if self.group else None,
                                model=model,
                                add_condition_slope=add_condition_slope,
                                **kwargs)
        df_data = self.data.copy()
        if do_mean_over_trials:
            df_data = df_data.groupby(fold_change_index_cols).mean().reset_index()

        # Make (fold) change
        if do_make_change:
            try:
                df_data, _ = utils.make_fold_change(df_data, y=self.y, index_cols=fold_change_index_cols,
                                                    treatment_name=self.treatment, fold_change_method=do_make_change)
            except Exception as e:
                print(e)

        reload(utils)
        self.trace.posterior = utils.rename_posterior(self.trace.posterior, self.b_name,
                                                      posterior_index_name='combined_condition',
                                                      group_name=self.group, group2_name=self.group2)

        # HDI and MAP:
        self.posterior = {var: utils.get_hdi_map(self.trace.posterior[var],
                                                 prefix=f'{var} '
                                                 if (var != self.b_name) and (var != 'slope_per_condition') else '')
                          for var in self.trace.posterior.data_vars}

        # Fill posterior into data
        self.data_and_posterior = utils.insert_posterior_into_data(posteriors=self.posterior,
                                                                   data=df_data.copy(),
                                                                   group=self.group)

        assert 'lower interval' in self.data_and_posterior.columns
        try:
            self.posterior = utils.recode_posterior(self.posterior, self.levels, self.data, self.original_data,
                                                    self.condition)
        except Exception as e:
            print(e)

        self.default_regression_charts()
        return self

    def default_regression_charts(self, **kwargs):
        # Default plots:
        try:
            # facet_kwargs=visualization.auto_facet(self.group,self,condition)
            if self.condition[0] and len(self.condition) > 1:
                return self.regression_charts(x=self.condition[0], column=self.group, row=self.condition[1], **kwargs)
            elif self.condition[0]:
                return self.regression_charts(x=self.condition[0], column=self.group, **kwargs)
            else:  # self.group:
                return self.regression_charts(x=self.condition[0] if self.condition[0] else ':O', **kwargs)
            #    self.regression_charts(column=self.group)
        except Exception as e:  # In case I haven't thought of something
            print(f'Please use window.regression_charts(): {e}')

    def regression_charts(self, x: str = ':O', color: str = ':N', detail: str = ':N', independent_axes=None, **kwargs):
        # Set some options
        if (x == '') or (x[-2] != ':'):
            x = f'{x}:O'
        if color[-2] != ':':
            color = f'{color}:N'
        posterior = self.data_and_posterior
        if len(x) > 2 and len(posterior[x[:-2]].unique() == 1):
            add_x_axis = True
            x = f'{self.condition[0]}:O'
        else:
            add_x_axis = False
        # If we are only plotting posterior and not data, independenet axis does not make sense:
        self.independent_axes = independent_axes or f'{self.y} diff' in posterior
        self.charts = []

        # 1. Plot posterior
        if posterior is not None:
            base_chart = alt.Chart(posterior)
            # Add zero for zero line
            base_chart.data['zero'] = 0

            self.chart_base_posterior = base_chart
            # No-data plot
            (self.chart_posterior_whiskers,
             self.chart_posterior_center, self.chart_zero) = plot_posterior(title=f'{self.y}',
                                                                            x=x,
                                                                            base_chart=base_chart,
                                                                            do_make_change=self.do_make_change)

            # if no self.data_and_posterior, use self.posterior to build slope per condition:
            if (self.b_name != 'lme') and (type(self.posterior) == dict):
                main_effect = (self.posterior[self.b_name] if self.posterior[self.b_name] is not None
                               else self.posterior['slope_per_condition'])
                self.chart_posterior_hdi_no_data = alt.layer(*plot_posterior(df=main_effect, title=f'{self.y}', x=x,
                                                                             do_make_change=self.do_make_change))

            self.chart_posterior_hdi = alt.layer(self.chart_posterior_whiskers, self.chart_posterior_center)
            self.charts.append(self.chart_posterior_whiskers)
            self.charts.append(self.chart_posterior_center)
            self.charts.append(self.chart_zero)
            self.charts_for_facet = self.charts.copy()  # KDE cannot be faceted so don't add it
            if (self.b_name != 'lme') and not add_x_axis:
                # Y Axis limits to match self.chart
                minmax = [float(posterior['lower interval'].min()), 0,
                          float(posterior['higher interval'].max())]
                y_domain = [min(minmax), max(minmax)]
                self.chart_posterior_kde = visualization.plot_posterior_density(base_chart, self.y, y_domain,
                                                                                self.trace,
                                                                                posterior,
                                                                                self.b_name,
                                                                                do_make_change=self.do_make_change)
                self.charts.append(self.chart_posterior_kde)
                # self.charts_for_facet.append(self.chart_posterior_kde) # kde cannot be faceted
        else:
            base_chart = alt.Chart(self.data)

        # 2. Plot data
        y = f'{self.y} diff'
        if y in posterior:
            if (detail != ':N') and (detail != ':O'):
                assert detail in self.data

            # Plot data:
            y_domain = list(np.quantile(base_chart.data[y], [.05, .95]))
            if x != ':O':
                self.chart_data_line, self.chart_data_points = visualization.line_with_highlight(base_chart, x, y,
                                                                                                 color, detail,
                                                                                                 highlight=False)
                self.charts.append(self.chart_data_line)
                self.charts.append(self.chart_data_points)
                self.charts_for_facet.append(self.chart_data_points)
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

        facet_requested = ('column' in kwargs.keys()) or ('row' in kwargs.keys())
        _ = alt.layer(*self.charts_for_facet).facet()  # Sanity check

        # 3. Make layered chart out of posterior and data
        if facet_requested:
            self.chart = alt.layer(*self.charts_for_facet)
        else:
            self.chart = alt.layer(*self.charts)

        if self.independent_axes:
            self.chart = self.chart.resolve_scale(y='independent')

        if facet_requested:
            self.chart = visualization.facet(self.chart, **kwargs)

        # 4. Make overlay for data_detail_plot
        # self.plot_slopes_shading()
        return self.chart

    def plot_slopes_intercepts(self, x=':O', y='mu_intercept_per_group center interval', **kwargs):
        assert self.posterior is not None
        if self.do_make_change:
            # combine posterior with original data instead, not diff TODO
            # Fill posterior into data
            data_and_posterior = utils.insert_posterior_into_data(posteriors=self.posterior,
                                                                  data=self.original_data.copy(),
                                                                  group=self.group)
        else:
            data_and_posterior = self.data_and_posterior

        # Redo boxplot (no need to show):
        self.data_box_detail(data=data_and_posterior, autofacet=False)

        # Make stand-alone posterior intercept chart:
        self.posterior_intercept = visualization.posterior_intercept_chart(data_and_posterior,
                                                                           x=x, y=y,
                                                                           group=self.group)
        # Redo chart_intercept with x=treatment for overlay with self.chart_data_box_detail:
        chart_intercept = visualization.posterior_intercept_chart(data_and_posterior,
                                                                  x=':O', y=y,
                                                                  group=self.group)
        chart = alt.layer(chart_intercept, self.chart_data_box_detail).resolve_scale(y='independent')

        # Check
        if len(chart.data) == 0:
            raise IndexError('was layer chart from different sources?')
        if ('column' in kwargs) or ('row' in kwargs):
            return visualization.facet(chart, **kwargs)
        else:  # Auto facet
            return visualization.facet(chart, **visualization.auto_facet(self.group, self.condition))

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
        # 4. color by dimension of slope (condition (and group if self.group))

    def plot_posteriors_no_slope(self,
                                 x=None,
                                 add_data=False,
                                 independent_axes=True,
                                 color=None,
                                 detail=':O',
                                 **kwargs):
        self.independent_axes = independent_axes
        x = x or self.treatment
        detail = detail or self.detail
        color = color or self.condition[0]
        # TODO default for detail
        chart_p = None
        if self.data_and_posterior is not None:
            base_chart = alt.Chart(self.data_and_posterior)
            # Plot posterior
            chart_p = alt.layer(*visualization.plot_posterior(x=x,
                                                              do_make_change=False,
                                                              add_data=add_data,
                                                              title=f'{self.y} estimate',
                                                              base_chart=base_chart,
                                                              **kwargs
                                                              ))
            if not add_data:  # done
                self.chart = chart_p
        else:
            add_data = True  # Otherwise nothing to do
            base_chart = alt.Chart(self.data)

        if add_data:
            # Make data plot:
            chart_d = visualization.plot_data_slope_trials(x=x,
                                                           y=self.y,
                                                           color=color,
                                                           detail=detail,
                                                           base_chart=base_chart)

            if self.data_and_posterior is None:
                self.chart = chart_d  # we're done
            else:
                self.chart = chart_p + chart_d

        return self.chart

    def plot(self, **kwargs):
        # Convenience function
        if not self.b_name:
            warnings.warn('No model has been fit. Defaulting to plotting "slopes" for data. Use .plot_slopes'
                          'or .plot_posteriors_no_slope to be explicit ')
            # x = self.levels[0] if 'x' not in kwargs else None
            # color = color or (self.levels[1] if len(self.levels) > 1 else None),
            return visualization.plot_data(self.data,
                                           y=self.y,
                                           **kwargs)[0]

        elif self.b_name == 'lme':
            if 'add_data' in kwargs.keys():
                warnings.warn('add_data keyword is not implemented for LME')
                kwargs.pop('add_data')
            return BayesWindow.regression_charts(self, **kwargs)
        elif 'slope' in self.b_name:
            return BayesWindow.regression_charts(self, **kwargs)
        elif self.b_name == 'mu_per_condition':
            return BayesWindow.plot_posteriors_no_slope(self, **kwargs)

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

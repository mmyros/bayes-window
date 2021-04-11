import warnings
from importlib import reload

import altair as alt
import arviz as az
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.preprocessing import LabelEncoder

from bayes_window import models
from bayes_window import utils
from bayes_window import visualization
from bayes_window.fitting import fit_numpyro
from bayes_window.model_comparison import compare_models
from bayes_window.visualization import plot_posterior

reload(visualization)


class BayesWindow:
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
            assert self.condition[0] in df.columns
        self.levels = utils.parse_levels(self.treatment, self.condition, self.group)
        # Combined condition
        # if self.condition[0] is None:
        #     self.data, self._key = df, None
        # else:
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

    def fit_lme(self, do_make_change='divide', add_interaction=False, add_data=False, override_formula=None,
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
            if len(self.data[self.condition[0]].unique()) > 1:
                include_condition = True

        # Make formula
        if include_condition:
            if len(self.condition) > 1:
                raise NotImplementedError(f'conditions {self.condition}. Use combined_condition')
                # This would need a combined condition dummy variable and an index of condition in patsy:
                # formula = f"{self.y} ~ 1+ {self.condition}(condition_index) | {self.treatment}"
            condition = self.condition[0].replace(" ", "_")
            # Make dummy variables for each level in condition:
            self.data = pd.concat((self.data,
                                   pd.get_dummies(self.data[condition],
                                                  prefix=condition,
                                                  prefix_sep='__',
                                                  drop_first=False)), axis=1)
            dummy_conditions = [cond for cond in self.data.columns if cond[:len(condition) + 2] == f'{condition}__']
            if add_group_intercept and not add_group_slope and not add_nested_group:
                formula = f"{self.y} ~ (1|{self.group}) + {self.treatment}| {dummy_conditions[0]}"
                # eg 'firing_rate ~ stim|neuron_x_mouse__0 +stim|neuron_x_mouse__1 ... + ( 1 |mouse )'
                for dummy_condition in dummy_conditions[1:]:
                    formula += f" + {self.treatment}|{dummy_condition}"
            elif add_group_intercept and add_group_slope and not add_nested_group:
                formula = (f"{self.y} ~ ({self.treatment}|{self.group}) + "
                           f"{self.treatment}| {dummy_conditions[0]}")
                for dummy_condition in dummy_conditions[1:]:
                    formula += f" + {self.treatment}|{dummy_condition}"
            elif add_group_intercept and add_group_slope and add_nested_group:
                formula = (f"{self.y} ~ ({self.treatment}|{self.group}) + "
                           f"{self.treatment}| {dummy_conditions[0]}:{self.group}")
                for dummy_condition in dummy_conditions[1:]:
                    formula += f" + {self.treatment}|{dummy_condition}:{self.group}"

            # if add_interaction:
            #     formula += f"+ {condition} * {self.treatment}"

        elif self.group:
            condition = None
            # Random intercepts and slopes (and their correlation): (Variable | Group)
            formula = f'{self.y} ~  {self.treatment} + (1 | {self.group})'  # (1 | {self.group}) +
            # Random intercepts and slopes (without their correlation): (1 | Group) + (0 + Variable | Group)
            # formula += f' + (1 | {self.group}) + (0 + {self.treatment} | {self.group})'
        else:
            condition = None
            formula = f"{self.y} ~ {self.treatment}"
        if override_formula:
            formula = override_formula
        print(f'Using formula {formula}')
        result = sm.mixedlm(formula,
                            self.data,
                            groups=self.data[self.group]).fit()
        print(result.summary().tables[1])
        self.data_and_posterior = utils.scrub_lme_result(result, include_condition, condition, self.data,
                                                         self.treatment)
        if add_data:
            self.data_and_posterior = utils.add_data_to_lme(do_make_change, include_condition, self.posterior,
                                                            condition, self.data, self.y, self.levels, self.treatment)
        return self

    def fit_conditions(self, model=models.model_single, add_data=True, **kwargs):

        self.model = model
        self.b_name = 'mu_per_condition'
        # Estimate model
        self.trace = fit_numpyro(y=self.data[self.y].values,
                                 condition=self.data['combined_condition'].values,
                                 # treatment=self.data[self.treatment].values,
                                 model=model,
                                 **kwargs
                                 ).posterior

        # Add data back
        if add_data:
            self.data_and_posterior, self.trace = utils.add_data_to_posterior(df_data=self.data, posterior=self.trace,
                                                                              y=self.y,
                                                                              fold_change_index_cols=self.levels[:3],
                                                                              treatment_name=self.levels[0],
                                                                              b_name=self.b_name,
                                                                              posterior_index_name='combined_condition',
                                                                              do_make_change=False,
                                                                              do_mean_over_trials=False,
                                                                              group_name=self.group)
        return self

    def fit_slopes(self, model=models.model_hierarchical, do_make_change='subtract', fold_change_index_cols=None,
                   do_mean_over_trials=True, fit_method=fit_numpyro, add_condition_slope=True, **kwargs):
        # if do_make_change not in ['subtract', 'divide']:
        #     raise ValueError(f'do_make_change should be subtract or divide, not {do_make_change}')
        if not add_condition_slope:
            warnings.warn(f'add_condition_slope is not requested. Slopes will be the same across {self.condition}')
        if self.b_name is not None:
            raise SyntaxError("A model is already present in this BayesWindow object. "
                              "Please create a new one by calling BayesWindow(...) again")
        self.do_make_change = do_make_change
        self.model = model
        # TODO handle no-group case
        if fold_change_index_cols is None:
            # TODO case with no plot_index_cols should include any multiindex?
            fold_change_index_cols = self.levels
        fold_change_index_cols = list(fold_change_index_cols)
        if add_condition_slope:
            add_condition_slope = self.condition[0] and (np.unique(self.data['combined_condition']).size > 1)
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
        # Add data back
        df_result, self.trace.posterior = utils.add_data_to_posterior(df_data=self.data.copy(),
                                                                      posterior=self.trace.posterior, y=self.y,
                                                                      fold_change_index_cols=fold_change_index_cols,
                                                                      treatment_name=self.treatment,
                                                                      b_name=self.b_name,
                                                                      posterior_index_name='combined_condition',
                                                                      do_make_change=do_make_change,
                                                                      do_mean_over_trials=do_mean_over_trials,
                                                                      group_name=self.group)

        # Back to human-readable labels
        if ('combined_condition' in self.original_data.columns) and ('combined_condition' in df_result.columns):
            levels_to_replace = list(set(self.levels) - {self.treatment})
            for level_values, data_subset in self.original_data.groupby(levels_to_replace):
                if not hasattr(level_values, '__len__'):  # This level is a scalar
                    level_values = [level_values]
                for level_name, level_value in zip(levels_to_replace, level_values):
                    df_result.loc[df_result['combined_condition'] == data_subset['combined_condition'].iloc[0],
                                  level_name] = level_value
        else:
            levels_to_replace = None

        self.data_and_posterior = df_result
        # sanity check:
        if self.data_and_posterior.shape[0] * 2 != self.data.shape[0]:
            print(f'We lost some detail in the data. This does not matter for posterior, but plotting data '
                  f'may suffer. Did was there another index column (like i_trial) other than {levels_to_replace}?')
        self.fold_change_index_cols = fold_change_index_cols
        return self

    def create_posterior_charts(self, x=':O', color=':N', detail=':N', independent_axes=False):
        # Set some options
        self.charts = []
        self.independent_axes = independent_axes
        x = x or self.levels[-1]
        if x[-2] != ':':
            x += ':O'
        color = color or self.levels[0]

        if (x == '') or (x[-2] != ':'):
            x = f'{x}:O'
        if color is None:
            color = ':N'
        if color[-2] != ':':
            color = f'{color}:N'

        posterior = self.posterior if self.data_and_posterior is None else self.data_and_posterior
        add_x_axis = False
        if len(x) > 2:
            if len(posterior[x[:-2]].unique() == 1):
                add_x_axis = True

        # 1. Plot posterior
        if posterior is not None:
            base_chart = alt.Chart(posterior)
            self.base_chart = base_chart
            self.chart_posterior_whiskers, self.chart_posterior_center = plot_posterior(
                title=f'{self.y}',
                x=x,
                base_chart=base_chart,
                do_make_change=self.do_make_change)
            self.chart_posterior = alt.layer(self.chart_posterior_whiskers, self.chart_posterior_center)
            self.charts.append(self.chart_posterior_whiskers)
            self.charts.append(self.chart_posterior_center)
            self.charts_for_facet = self.charts.copy()  # KDE cannot be faceted so don't add it
            if (self.b_name != 'lme') and not add_x_axis:
                # Y Axis limits to match self.chart_posterior
                minmax = [float(posterior['lower interval'].min()), 0,
                          float(posterior['higher interval'].max())]
                y_domain = [min(minmax), max(minmax)]
                self.chart_posterior_kde = visualization.plot_posterior_density(base_chart, self.y, y_domain,
                                                                                self.trace,
                                                                                posterior,
                                                                                self.b_name,
                                                                                do_make_change=self.do_make_change)
                self.charts.append(self.chart_posterior_kde)
            else:
                self.chart_posterior_kde = base_chart.mark_rule().encode()  # Empty chart

        else:
            base_chart = alt.Chart(self.data)
            self.chart_posterior = base_chart.mark_rule().encode()  # Empty chart

        # Default empty chart:
        empty_chart = base_chart.mark_rule().encode()

        # 2. Plot data
        y = f'{self.y} diff'
        if y not in posterior:
            warnings.warn("Did you have Uneven number of entries in conditions? I can't add data overlay")
            return self

        if (detail != ':N') and (detail != ':O'):
            assert detail in self.data
            assert detail in self.fold_change_index_cols

        # Plot data:
        y_domain = list(np.quantile(base_chart.data[y], [.05, .95]))
        if add_x_axis:
            self.chart_data_line = visualization.line_with_highlight(base_chart, x, y, color, detail, highlight=False)
            self.charts.extend(self.chart_data_line)
            self.charts_for_facet.extend(self.chart_data_line)
        else:
            self.chart_data_line = empty_chart

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
        return self

    def plot_posteriors_slopes(self, x=':O', color=':N', detail=':N', add_box=True, add_data=True,
                               independent_axes=False,
                               add_posterior_density=True,
                               **kwargs):
        # Set some options
        self.independent_axes = independent_axes
        x = x or self.levels[-1]
        if x[-2] != ':':
            x += ':O'
        color = color or self.levels[0]
        posterior = self.posterior if self.data_and_posterior is None else self.data_and_posterior

        # Plot posterior
        if posterior is not None:
            add_data = add_data and add_box
            base_chart = alt.Chart(posterior)
            chart_p = alt.layer(*plot_posterior(title=f'{self.y}',
                                                x=x,
                                                base_chart=base_chart,
                                                do_make_change=self.do_make_change))
        else:
            base_chart = alt.Chart(self.data)
            add_data = True  # Otherwise nothing to do
            chart_p = None
        if add_data and f'{self.y} diff' not in self.data_and_posterior.columns:
            warnings.warn(f'change in data was not added, but add_data requested:'
                          f'{self.y} diff is not in {self.data_and_posterior.columns}')
            add_data = False

        if add_data:
            assert self.data_and_posterior is not None
            y = f'{self.y} diff'
            if (detail != ':N') and (detail != ':O'):
                assert detail in self.data
                assert detail in self.fold_change_index_cols

            chart_d, y_scale = visualization.plot_data(x=x, y=y, color=color, add_box=add_box,
                                                       detail=detail,
                                                       base_chart=base_chart)
            self.chart = chart_d + chart_p
        else:
            y = self.y
            y_scale = None
            self.chart = chart_p

        if np.unique(self.data['combined_condition']).size > 1:
            add_posterior_density = False
        if x != ':O':
            if (len(self.data[x[:-2]].unique()) > 1):
                # That would be too dense. Override add_posterior_density
                add_posterior_density = False

        if posterior is not None and add_posterior_density and (self.b_name != 'lme'):
            chart_kde = visualization.plot_posterior_density(base_chart, y, y_scale, self.trace, posterior,
                                                             self.b_name, do_make_change=self.do_make_change)

            # Add kde to charts:
            if add_box and add_data:
                self.chart |= chart_kde
            else:
                self.chart += chart_kde
                independent_axes = False
        if independent_axes:
            self.chart = self.chart.resolve_scale(y='independent')
        return self.chart

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
            return BayesWindow.plot_posteriors_slopes(self, add_data=False, **kwargs)
        elif 'slope' in self.b_name:
            return BayesWindow.plot_posteriors_slopes(self, **kwargs)
        elif self.b_name == 'mu_per_condition':
            return BayesWindow.plot_posteriors_no_slope(self, **kwargs)

    def plot_data_details(self, facet=True):
        if (self.detail is None) or (self.detail == ':O'):
            warnings.warn('This plot works best with a "detail" argument when constructing BayesWindow')
        c1 = visualization.plot_data(df=self.data, x=self.treatment, y=self.y)[0].properties(width=60)
        c2 = visualization.plot_data_slope_trials(df=self.data, x=self.treatment, y=self.y, color=None,
                                                  detail=self.detail)
        chart = c1 + c2
        if facet:
            if self.group and self.condition[0]:
                chart = chart.facet(column=self.group, row=self.condition[0])
            elif self.group:
                chart = chart.facet(column=self.group)
            elif self.condition[0]:
                chart = chart.facet(column=self.condition[0])
        return chart

    def facet(self, width=150, height=160, **kwargs):
        assert ('row' in kwargs) or ('column' in kwargs), 'Give facet either row, or column'
        if self.independent_axes is None:
            # TODO let's not force users to plot. have a sensible default
            raise RuntimeError('Plot first, then you can use facet')
        if self.independent_axes:
            self.facetchart = visualization.facet(self.chart, width=width, height=height, **kwargs)
        else:
            self.facetchart = self.chart.properties(width=width, height=height).facet(**kwargs)
        return self.facetchart

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

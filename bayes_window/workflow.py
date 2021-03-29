import warnings
from importlib import reload

import altair as alt
import arviz as az
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

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
        self.data, self._key = utils.combined_condition(df, self.levels)
        self.detail = detail
        self.y = y

        # Preallocate attributes:
        self.b_name = None  # Depends on model we'll fit
        self.do_make_change = None  # Depends on plotting input
        self.add_data = None  # We'll use this in plotting
        self.independent_axes = None
        self.data_and_posterior = None
        self.posterior = None
        self.trace = None
        self.model = None

    def fit_anova(self):
        from statsmodels.stats.anova import anova_lm
        if self.group:
            # Average over group:
            data = self.data.groupby([self.group, self.treatment]).mean().reset_index()
        else:
            data = self.data
        # dehumanize all columns and variable names for statsmodels:
        [data.rename({col: col.replace(" ", "_")}, axis=1, inplace=True) for col in data.columns]
        self.y = self.y.replace(" ", "_")
        formula = f'{self.y}~{self.treatment}'
        lm = sm.ols(formula, data=data).fit()
        print(f'{formula}\n {anova_lm(lm, typ=2)}')
        return anova_lm(lm, typ=2)['PR(>F)'][self.treatment] < 0.05

    def fit_lme(self, add_data=False, do_make_change='divide', add_interaction=False):
        # model = MixedLM(endog=self.data[self.y],
        #                 exog=self.data[self.condition],
        #                 groups=self.data[self.group],
        #                 # exog_re=exog.iloc[:, 0]
        #                 )
        self.b_name = 'lme'
        self.add_data = add_data
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
            # formula = f"{self.y} ~ (1 | {self.group}) + ({dummy_conditions[0]}|{self.group})"
            formula = f"{self.y} ~ 1+ ({dummy_conditions[0]}|{self.group})"
            for dummy_condition in dummy_conditions[1:]:
                formula += f" + ({dummy_condition}|{self.group}) "
                # formula += f"  {dummy_condition}+ (0 + {dummy_condition} | {self.group})"
            # if add_interaction:
            #     formula += f"+ {condition} * {self.treatment}"

        elif self.group:
            condition = None
            # Random intercepts and slopes (and their correlation): (Variable | Group)
            formula = f'{self.y} ~  (0+ {self.treatment} | {self.group})'  # (1 | {self.group}) +
            # Random intercepts and slopes (without their correlation): (1 | Group) + (0 + Variable | Group)
            # formula += f' + (1 | {self.group}) + (0 + {self.treatment} | {self.group})'
        else:
            condition = None
            formula = f"{self.y} ~ 1 + {self.treatment}"
        print(f'Using formula {formula}')
        result = sm.mixedlm(formula,
                            self.data,
                            groups=self.data[self.group]).fit()
        print(result.summary().tables[1])
        self.posterior = utils.scrub_lme_result(result, include_condition, condition, self.data, self.treatment)
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
            self.data_and_posterior, self.trace = utils.add_data_to_posterior(df_data=self.data,
                                                                              posterior=self.trace,
                                                                              y=self.y,
                                                                              fold_change_index_cols=self.levels[:3],
                                                                              treatment_name=self.levels[0],
                                                                              b_name=self.b_name,
                                                                              posterior_index_name='combined_condition',
                                                                              do_mean_over_trials=False,
                                                                              do_make_change=False,
                                                                              add_data=add_data,
                                                                              group_name=self.group,
                                                                              )
        return self

    def fit_slopes(self, model=models.model_hierarchical, do_make_change='subtract',
                   fold_change_index_cols=None, do_mean_over_trials=True, add_data: bool = True, **kwargs):
        # TODO case with no group_name
        if do_make_change not in ['subtract', 'divide']:
            raise ValueError(f'do_make_change should be subtract or divide, not {do_make_change}')

        self.do_make_change = do_make_change
        self.add_data = add_data  # We'll use this in plotting
        self.model = model
        # TODO handle no-group case
        if fold_change_index_cols is None:
            # TODO case with no plot_index_cols should include any multiindex?
            fold_change_index_cols = self.levels  # [-1]
        fold_change_index_cols = list(fold_change_index_cols)
        # if not self.condition[0]:
        #     np.unique(condition).size == 1
        # warnings.warn('Condition was not provided. Assuming there is no additional condition, just treatment')
        # self.condition[0] = 'dummy_condition'
        # self.data.insert(self.data.shape[-1] - 1, 'dummy_condition', np.zeros(self.data.shape[0]).astype(int))
        # fold_change_index_cols.append('dummy_condition')
        include_condition = self.condition[0] and np.unique(self.data[self.condition[0]]).size > 1
        self.b_name = 'b_stim_per_condition' if include_condition else 'b_stim'
        if include_condition and (not self.condition[0] in fold_change_index_cols):
            fold_change_index_cols.extend(self.condition)
        self.trace = fit_numpyro(y=self.data[self.y].values,
                                 treatment=self.data[self.treatment].values,
                                 condition=self.data[self.condition[0]].values if self.condition[0] else None,
                                 group=self.data[self.group].values,
                                 model=model,
                                 **kwargs)
        if add_data:
            # Add data back
            # TODO posterior_index_name=self.condition[0] will not work if need combined_condition
            df_result, self.trace.posterior = utils.add_data_to_posterior(df_data=self.data,
                                                                          posterior=self.trace.posterior,
                                                                          y=self.y,
                                                                          fold_change_index_cols=fold_change_index_cols,
                                                                          treatment_name=self.treatment,
                                                                          b_name=self.b_name,
                                                                          posterior_index_name=self.condition[0],
                                                                          do_make_change=do_make_change,
                                                                          do_mean_over_trials=do_mean_over_trials,
                                                                          add_data=self.add_data,
                                                                          group_name=self.group,
                                                                          )
        else:  # Just convert posterior to dataframe
            from bayes_window.utils import trace2df
            # TODO we add data regardless. Is there a way to not use self.data?
            df_result, self.trace.posterior = trace2df(self.trace.posterior,
                                                       self.data, b_name=self.b_name,
                                                       posterior_index_name=self.condition[0],
                                                       group_name=self.group,
                                                       )

        # Back to human-readable labels
        [df_result[col].replace(self._key[col], inplace=True) for col in self._key.keys()
         if (not col == self.treatment) and (col in df_result)]
        self.data_and_posterior = df_result
        self.fold_change_index_cols = fold_change_index_cols
        return self

    def plot_posteriors_slopes(self, x=':O', color=':O', detail=':O', add_box=True, independent_axes=False,
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
            add_data = self.add_data and add_box
            base_chart = alt.Chart(posterior)
            chart_p = plot_posterior(title=f'{self.y}',
                                     x=x,
                                     base_chart=base_chart,
                                     do_make_change=self.do_make_change)
        else:
            base_chart = alt.Chart(self.data)
            add_data = True  # Otherwise nothing to do
            chart_p = None

        if add_data:
            assert self.data_and_posterior is not None
            y = f'{self.y} diff'
            if y not in self.data_and_posterior:
                raise KeyError(f'change in data was not added, but add_data requested:'
                               f'{self.y} is not in {self.data_and_posterior.keys}')
            if detail != ':O':
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

        if x != ':O':
            if len(self.data[x[:-2]].unique()) > 1:  # That would be too dense. Override add_posterior_density
                add_posterior_density = False

        if posterior is not None and add_posterior_density and (self.b_name != 'lme'):
            chart_kde = visualization.plot_posterior_density(base_chart, y, y_scale, self.trace, posterior, self.b_name)

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
            chart_p = visualization.plot_posterior(x=x,
                                                   do_make_change=False,
                                                   add_data=add_data,
                                                   title=f'{self.y} estimate',  # TODO uncomment
                                                   base_chart=base_chart,
                                                   **kwargs
                                                   )
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
            return visualization.plot_data(self.data, x=self.levels[0], y=self.y,
                                           color=self.levels[1] if len(self.levels) > 1 else None,
                                           **kwargs)[0]

        elif self.b_name == 'lme':
            return BayesWindow.plot_posteriors_slopes(self, **kwargs)
        elif 'b_stim' in self.b_name:
            return BayesWindow.plot_posteriors_slopes(self, **kwargs)
        elif self.b_name == 'mu_per_condition':
            return BayesWindow.plot_posteriors_no_slope(self, **kwargs)

    def facet(self, width=50, height=60, **kwargs):
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
        elif 'b_stim' in self.b_name:
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
                models['no_group_slope'] = self.model
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

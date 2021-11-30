from typing import List, Any

import pandas as pd
import statsmodels.formula.api as sm
from altair import LayerChart, Chart

from bayes_window import BayesWindow
from bayes_window import utils


class LMERegression:
    b_name: str
    chart_data_line: Chart
    chart_posterior_kde: Chart
    chart_zero: Chart
    posterior_intercept: Chart
    chart: Chart
    charts: Chart
    chart_data_boxplot: Chart
    chart_posterior_whiskers: Chart
    chart_posterior_center: Chart
    chart_base_posterior: Chart
    charts_for_facet: List[Any]
    chart_posterior_hdi_no_data: LayerChart
    add_data: bool
    data_and_posterior: pd.DataFrame
    posterior: dict

    def __init__(self, window=None, add_data=True, **kwargs):
        window = window if window is not None else BayesWindow(**kwargs)
        window.add_data = add_data
        self.window = window
        self.independent_axes = False

    def fit(self, do_make_change='divide', add_interaction=False, add_data=False, formula=None,
            add_group_intercept=True, add_group_slope=False, add_nested_group=False, **kwargs):
        # model = MixedLM(endog=self.window.data[self.window.y],
        #                 exog=self.window.data[self.window.condition],
        #                 groups=self.window.data[self.window.group],
        #                 # exog_re=exog.iloc[:, 0]
        #                 )
        self.b_name = 'lme'
        # dehumanize all columns and variable names for statsmodels:
        [self.window.data.rename({col: col.replace(" ", "_")}, axis=1, inplace=True) for col in
         self.window.data.columns]
        self.window.y = self.window.y.replace(" ", "_")
        self.window.group = self.window.group.replace(" ", "_")
        self.window.treatment = self.window.treatment.replace(" ", "_")
        self.window.do_make_change = do_make_change
        include_condition = False  # in all but the following cases:
        if self.window.condition[0]:
            self.window.condition[0] = self.window.condition[0].replace(" ", "_")
            if len(self.window.condition) > 1:
                self.window.condition[1] = self.window.condition[1].replace(" ", "_")
            if len(self.window.data[self.window.condition[0]].unique()) > 1:
                include_condition = True
        # condition = None  # Preallocate

        # Make formula
        if include_condition and not formula:
            if len(self.window.condition) > 1:
                raise NotImplementedError(f'conditions {self.window.condition}. Use combined_condition')
                # This would need a combined condition dummy variable and an index of condition in patsy:
                # formula = f"{self.window.y} ~ 1+ {self.window.condition}(condition_index) | {self.window.treatment}"
                # Combined condition
                # self.window.data, self._key = utils.combined_condition(self.window.data.copy(), self.window.condition)
                # self.window.condition = ['combined_condition']
                # self.window.original_data = self.window.data.copy()

            # Make dummy variables for each level in condition:
            self.window.data = pd.concat((self.window.data,
                                          pd.get_dummies(self.window.data[self.window.condition[0]],
                                                         prefix=self.window.condition[0],
                                                         prefix_sep='__',
                                                         drop_first=False)), axis=1)
            dummy_conditions = [cond for cond in self.window.data.columns
                                if cond[:len(self.window.condition[0]) + 2] == f'{self.window.condition[0]}__']
            if add_group_intercept and not add_group_slope and not add_nested_group:
                formula = f"{self.window.y} ~ (1|{self.window.group}) + {self.window.treatment}| {dummy_conditions[0]}"
                # eg 'firing_rate ~ stim|neuron_x_mouse__0 +stim|neuron_x_mouse__1 ... + ( 1 |mouse )'
                for dummy_condition in dummy_conditions[1:]:
                    formula += f" + {self.window.treatment}|{dummy_condition}"
            elif add_group_intercept and add_group_slope and not add_nested_group:
                formula = (f"{self.window.y} ~ ({self.window.treatment}|{self.window.group}) "
                           f" + {self.window.treatment}| {dummy_conditions[0]}"
                           )
                for dummy_condition in dummy_conditions[1:]:
                    formula += f" + {self.window.treatment}|{dummy_condition}"
            elif add_group_intercept and add_group_slope and add_nested_group:
                formula = (f"{self.window.y} ~ ({self.window.treatment}|{self.window.group}) + "
                           f"{self.window.treatment}| {dummy_conditions[0]}:{self.window.group}")
                for dummy_condition in dummy_conditions[1:]:
                    formula += f" + {self.window.treatment}|{dummy_condition}:{self.window.group}"

            # if add_interaction:
            #     formula += f"+ {condition} * {self.window.treatment}"

        elif self.window.group and not formula:
            # Random intercepts and slopes (and their correlation): (Variable | Group)
            formula = f'{self.window.y} ~  C({self.window.treatment}, Treatment) + (1 | {self.window.group})'  # (1 | {self.window.group}) +
            # Random intercepts and slopes (without their correlation): (1 | Group) + (0 + Variable | Group)
            # formula += f' + (1 | {self.window.group}) + (0 + {self.window.treatment} | {self.window.group})'
        elif not formula:
            formula = f"{self.window.y} ~ C({self.window.treatment}, Treatment)"

        print(f'Using formula {formula}')
        result = sm.mixedlm(formula,
                            self.window.data,
                            groups=self.window.data[self.window.group]).fit()
        print(result.summary().tables[1])
        self.data_and_posterior = utils.scrub_lme_result(result, include_condition, self.window.condition[0],
                                                         self.window.data,
                                                         self.window.treatment)
        if add_data:
            raise NotImplementedError(f'No adding data to LME')
            # self.data_and_posterior = utils.add_data_to_lme(do_make_change, include_condition, self.posterior,
            #                                                 self.window.condition[0], self.window.data,
            #                                                 self.window.y, self.levels,
            #                                                 self.window.treatment)

            # self.trace.posterior = utils.rename_posterior(self.trace.posterior, self.b_name,
            #                                               posterior_index_name='combined_condition',
            #                                               group_name=self.window.group, group2_name=self.window.group2)
            #
            # # HDI and MAP:
            # self.posterior = {var: utils.get_hdi_map(self.trace.posterior[var],
            #                                          prefix=f'{var} '
            #                                          if (var != self.b_name) and (var != 'slope_per_condition') else '')
            #                   for var in self.trace.posterior.data_vars}
            #
            # # Fill posterior into data
            # self.data_and_posterior = utils.insert_posterior_into_data(posteriors=self.posterior,
            #                                                            data=self.window.data.copy(),
            #                                                            group=self.window.group)
            #
            #
            # self.posterior = utils.recode_posterior(self.posterior, self.levels, self.window.data, self.window.original_data,
            #                                             self.window.condition)
        from bayes_window.slopes import BayesRegression
        self.charts = BayesRegression.plot(self)
        return self

    def fit_anova(self, formula=None, **kwargs):
        from statsmodels.stats.anova import anova_lm
        if self.window.group:
            # Average over group:
            data = self.window.data.groupby([self.window.group, self.window.treatment]).mean().reset_index()
        else:
            data = self.window.data
        # dehumanize all columns and variable names for statsmodels:
        [data.rename({col: col.replace(" ", "_")}, axis=1, inplace=True) for col in data.columns]
        self.window.y = self.window.y.replace(" ", "_")
        if not formula:
            formula = f'{self.window.y}~{self.window.treatment}'
        lm = sm.ols(formula, data=data).fit()
        print(f'{formula}\n {anova_lm(lm, typ=2, **kwargs).round(2)}')
        return anova_lm(lm, typ=2)['PR(>F)'][self.window.treatment] < 0.05

    def plot(self, **kwargs):
        from bayes_window import BayesRegression
        return BayesRegression.plot(self, **kwargs)

    def facet(self, **kwargs):
        from bayes_window import BayesRegression
        return BayesRegression.facet(self, **kwargs)

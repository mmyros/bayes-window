import pandas as pd
import statsmodels.formula.api as sm

from bayes_window import BayesWindow
from bayes_window import utils


class LMERegression(BayesWindow):

    def fit(self, do_make_change='divide', add_interaction=False, add_data=False, formula=None,
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
                # Combined condition
                self.data, self._key = utils.combined_condition(self.data.copy(), self.condition)
                self.condition = ['combined_condition']
                self.original_data = self.data.copy()

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
            raise NotImplementedError(f'No adding data to LME')
            self.data_and_posterior = utils.add_data_to_lme(do_make_change, include_condition, self.posterior,
                                                            self.condition[0], self.data, self.y, self.levels,
                                                            self.treatment)

            # self.trace.posterior = utils.rename_posterior(self.trace.posterior, self.b_name,
            #                                               posterior_index_name='combined_condition',
            #                                               group_name=self.group, group2_name=self.group2)
            #
            # # HDI and MAP:
            # self.posterior = {var: utils.get_hdi_map(self.trace.posterior[var],
            #                                          prefix=f'{var} '
            #                                          if (var != self.b_name) and (var != 'slope_per_condition') else '')
            #                   for var in self.trace.posterior.data_vars}
            #
            # # Fill posterior into data
            # self.data_and_posterior = utils.insert_posterior_into_data(posteriors=self.posterior,
            #                                                            data=self.data.copy(),
            #                                                            group=self.group)
            #
            #
            # self.posterior = utils.recode_posterior(self.posterior, self.levels, self.data, self.original_data,
            #                                             self.condition)
        from bayes_window.slopes import BayesRegression
        BayesRegression.plot(self)
        return self



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

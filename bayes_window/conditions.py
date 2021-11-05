import altair as alt
import arviz as az
from bayes_window import models, BayesWindow
from bayes_window import utils
from bayes_window import visualization
from bayes_window.fitting import fit_numpyro
from sklearn.preprocessing import LabelEncoder


class BayesConditions(BayesWindow):

    def __init__(self, add_data=False, **kwargs):
        super().__init__(add_data=add_data, **kwargs)

    def fit(self, model=models.model_single, fit_fn=fit_numpyro, **kwargs):

        self.model = model
        self.b_name = 'mu_per_condition'

        # add all levels into condition
        # if self.group and self.group not in self.condition:
        #    self.condition += [self.group]
        if self.treatment not in self.condition:
            self.condition += [self.treatment]

        # Recode dummy condition taking into account all levels
        self.data, self._key = utils.combined_condition(self.original_data.copy(), self.condition)

        # Transform group to integers as required by numpyro:
        if self.group:
            self.data[self.group] = LabelEncoder().fit_transform(self.data[self.group])
        if self.treatment:
            self.data[self.treatment] = LabelEncoder().fit_transform(self.data[self.treatment])

        # Estimate model
        self.trace = fit_fn(y=self.data[self.y].values,
                            condition=self.data['combined_condition'].values,
                            group=self.data[self.group].values if self.group else None,
                            treatment=self.data[self.treatment].values,
                            model=model,
                            **kwargs
                            )

        # Add data back
        self.trace.posterior = utils.rename_posterior(self.trace.posterior, self.b_name,
                                                      posterior_index_name='combined_condition',
                                                      group_name=self.group,
                                                      treatment_name=self.treatment
                                                      )

        # HDI and MAP:
        self.posterior = {var: utils.get_hdi_map(
            self.trace.posterior[var],
            prefix=f'{var} ' if (var != self.b_name) and
                                (var not in ['slope_per_condition']) else '')
            for var in self.trace.posterior.data_vars if var not in ['mu_intercept_per_treatment']}

        # Fill posterior into data
        self.data_and_posterior = utils.insert_posterior_into_data(posteriors=self.posterior,
                                                                   group=self.group,
                                                                   group2=self.group2,
                                                                   data=self.original_data.copy())

        self.posterior = utils.recode_posterior(self.posterior, self.levels, self.data, self.original_data,
                                                self.condition)

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
        x = x or self.treatment or self.condition[0]
        detail = detail or self.detail
        if self.treatment:
            color = color or self.condition[0]
        elif len(self.condition) > 1:
            color = color or self.condition[1]
        # TODO default for detail
        posterior = self.data_and_posterior if self.add_data else self.posterior['mu_per_condition']
        chart_p = None
        if posterior is not None:
            base_chart = alt.Chart(posterior)  # TODO self.data_and_posterior is broken
            # Plot posterior
            chart_p = alt.layer(*visualization.plot_posterior(x=x,
                                                              do_make_change=False,
                                                              title=f'{self.y} estimate',
                                                              base_chart=base_chart,
                                                              color=color,
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
        if auto_facet:
            if ('column' in kwargs) or ('row' in kwargs):
                return visualization.facet(self.chart, **kwargs)
            elif len(self.condition) > 2:  # Auto facet
                return visualization.facet(self.chart, **visualization.auto_facet(self.condition[2]))

        return self.chart

    def query_posterior(self, query):
        query_combined_condition = self.posterior['mu_per_condition'].query(query)['combined_condition']
        posterior_post_query = self.trace.posterior['mu_per_condition'].sel(combined_condition=
                                                                            slice(query_combined_condition.min(),
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
        );

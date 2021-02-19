import warnings
from importlib import reload

from bayes_window import models
from bayes_window import utils
from bayes_window import visualization
from bayes_window.fitting import fit_numpyro
from bayes_window.visualization import plot_posterior
from sklearn.preprocessing import LabelEncoder

reload(visualization)
reload(utils)
le = LabelEncoder()


def estimate_slope(df, y='isi', levels=('stim', 'mouse', 'neuron'), hold_for_facet=True, plot_index_cols=None,
                   plot_x=None,
                   plot_color=None,
                   model=models.model_hier_normal_stim, add_data=True, **kwargs):
    reload(visualization)
    reload(utils)
    if plot_x is None:
        plot_x = levels[2]
    if plot_color is None:
        plot_color = levels[1]
    if plot_index_cols is None:
        plot_index_cols = levels[:3]
    # By convention, top condition is first in list of levels:
    top_condition = levels[0]
    # Transform conditions to integers as required by numpyro:
    key = dict()
    for level in levels:
        df[level] = le.fit_transform(df[level])
        # Keep key for later use
        key[level] = dict(zip(range(len(le.classes_)), le.classes_))

    # Estimate model
    trace = fit_numpyro(y=df[y].values,
                        stim_on=df[top_condition].values,
                        treat=df[levels[2]].values,
                        subject=df[levels[1]].values,
                        progress_bar=False,
                        model=model,
                        n_draws=1000, num_chains=1)
    if add_data:
        # Add data back
        df_result = utils.add_data_to_posterior(df=df,
                                                trace=trace,
                                                y=y,
                                                index_cols=plot_index_cols,
                                                condition_name=top_condition,
                                                b_name='b_stim_per_condition',
                                                group_name=levels[2],
                                                do_make_change=True,
                                                do_mean_over_trials=True,
                                                )

        [df_result[col].replace(key[col], inplace=True) for col in key.keys() if not col == top_condition]
        # Plot data and posterior
        chart = visualization.plot_data_and_posterior(df=df_result,
                                                      y=f'{y} diff',
                                                      x=plot_x,
                                                      color=plot_color,
                                                      title=y,
                                                      hold_for_facet=hold_for_facet, **kwargs)
    else:
        from bayes_window.utils import trace2df
        df_result = trace2df(trace=trace, df=df, b_name='b_stim_per_condition', group_name=levels[2])
        [df_result[col].replace(key[col], inplace=True) for col in key.keys() if not col == top_condition]
        chart = plot_posterior(df_result, do_make_change=True, x=plot_x, **kwargs)
    return chart, df_result


class BayesWindow():
    def __init__(self,
                 df=None,
                 y='isi',
                 levels=('stim', 'mouse', 'neuron'),
                 ):
        df['combined_condition'] = df[levels[0]].astype('str')
        for level in levels[1:]:
            df['combined_condition'] += df[level].astype('str')

        # Transform conditions to integers as required by numpyro:
        df['combined_condition'] = le.fit_transform(df['combined_condition'])
        # TODO Keep key for later use
        # key = dict(zip(range(len(le.classes_)), le.classes_))
        self.levels = levels
        self.data = df
        self.y = y

    def fit_conditions(self,
                       model=models.model_single_lognormal):

        self.model = model
        self.bname = 'mu_per_condition'
        # Estimate model
        self.trace = fit_numpyro(y=self.data[self.y].values,
                                 treat=self.data['combined_condition'].values,
                                 model=model,
                                 )
        # Add data back
        self.data_and_posterior = utils.add_data_to_posterior(df=self.data,
                                                              trace=self.trace,
                                                              y=self.y,
                                                              index_cols=self.levels[:3],
                                                              condition_name=self.levels[0],
                                                              b_name=self.bname,
                                                              group_name='combined_condition',
                                                              do_mean_over_trials=False,
                                                              do_make_change=False
                                                              )

    def fit_slopes(self, add_data=True, model=models.model_hier_normal_stim,
                   plot_index_cols=None):
        self.bname = 'b_stim_per_condition'

        if plot_index_cols is None:
            plot_index_cols = self.levels[:3]
        # By convention, top condition is first in list of levels:
        top_condition = self.levels[0]
        # Transform conditions to integers as required by numpyro:
        key = dict()
        for level in self.levels:
            self.data[level] = le.fit_transform(self.data[level])
            # Keep key for later use
            key[level] = dict(zip(range(len(le.classes_)), le.classes_))

        # Estimate model
        trace = fit_numpyro(y=self.data[self.y].values,
                            stim_on=self.data[top_condition].values,
                            treat=self.data[self.levels[2]].values,
                            subject=self.data[self.levels[1]].values,
                            progress_bar=False,
                            model=model,
                            n_draws=1000, num_chains=1)
        if add_data:
            # Add data back
            df_result = utils.add_data_to_posterior(df=self.data,
                                                    trace=trace,
                                                    y=self.y,
                                                    index_cols=plot_index_cols,
                                                    condition_name=top_condition,
                                                    b_name=self.bname,
                                                    group_name=self.levels[2],
                                                    do_make_change=True,
                                                    do_mean_over_trials=True,
                                                    )
        else:
            from bayes_window.utils import trace2df
            df_result = trace2df(trace, self.data, b_name='b_stim_per_condition', group_name=self.levels[2])
        [df_result[col].replace(key[col], inplace=True) for col in key.keys() if not col == top_condition]
        self.data_and_posterior = df_result

    def plot_slopes(self, x=None, color=None, hold_for_facet=False, **kwargs):

        if x is None:
            x = self.levels[2]
        if color is None:
            color = self.levels[1]
        chart = visualization.plot_data_and_posterior(df=self.data_and_posterior,
                                                      y=f'{self.y} diff',
                                                      x=x,
                                                      color=color,
                                                      title=self.y,
                                                      hold_for_facet=hold_for_facet, **kwargs)
        return chart

    def plot_posteriors_no_slope(self, x='stim:O', add_data=False,
                                 independent_axes=True,
                                 color='neuron:N',
                                 detail='i_trial',
                                 **kwargs):
        reload(visualization)
        self.independent_axes = independent_axes
        if hasattr(self, 'data_and_posterior'):
            # Plot posterior
            self.chart = visualization.plot_posterior(df=self.data_and_posterior,
                                                      x=x,
                                                      # x=levels[0],
                                                      do_make_change=False,
                                                      add_data=add_data,
                                                      title='Estimate',
                                                      **kwargs
                                                      )
        else:
            add_data = visualization.plot_data_slope_trials(df=self.data,
                                                            x=x,
                                                            y=self.y,
                                                            color=color,
                                                            detail=detail)

        if add_data:
            # Make data plot:
            self.chart += visualization.plot_data_slope_trials(df=self.data,
                                                               x=x,
                                                               y=self.y,
                                                               color=color,
                                                               detail=detail)

        return self.chart

    def plot(self, **kwargs):
        # Convenience function
        if not hasattr(self, 'bname'):
            warnings.warn('No model has been fit. Defaulting to plotting "slopes" for data. Use .plot_slopes'
                          'or .plot_posteriors_no_slope to be explicit ')
            return visualization.plot_data(self.data,x=self.levels[0],y=self.y, color=self.levels[1], **kwargs)

        if self.bname == 'b_stim_per_condition':
            return BayesWindow.plot_slopes(self, **kwargs)
        elif self.bname == 'mu_per_condition':
            return BayesWindow.plot_posteriors_no_slope(self, **kwargs)
        else:
            raise RuntimeError('Unknown model! please modify plotting code')

    def facet(self, row=None, column=None, width=50, height=60):
        assert row or column
        if not hasattr(self, 'independent_axes'):
            raise RuntimeError('Plot first, then you can use facet')
        if self.independent_axes:
            self.facetchart = visualization.facet(self.chart, row=row, column=column, width=width, height=height)

        else:
            self.facetchart = self.chart.facet(row=row, column=column)
        return self.facetchart

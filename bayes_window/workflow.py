import warnings
from importlib import reload

import altair as alt
from sklearn.preprocessing import LabelEncoder

from bayes_window import models
from bayes_window import utils
from bayes_window import visualization
from bayes_window.fitting import fit_numpyro
from bayes_window.visualization import plot_posterior

reload(visualization)
reload(utils)
le = LabelEncoder()


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
        # Keep key for later use- maybe we don't need to in this case
        # key = dict(zip(range(len(le.classes_)), le.classes_))
        self.levels = list(levels)
        self.data = df.copy()
        self.y = y

    def fit_conditions(self, model=models.model_single_lognormal, add_data=True):

        self.model = model
        self.bname = 'mu_per_condition'
        # Estimate model
        self.trace = fit_numpyro(y=self.data[self.y].values,
                                 treat=self.data['combined_condition'].values,
                                 model=model,
                                 )
        # Add data back
        if add_data:
            self.data_and_posterior = utils.add_data_to_posterior(df_data=self.data,
                                                                  trace=self.trace,
                                                                  y=self.y,
                                                                  index_cols=self.levels[:3],
                                                                  condition_name=self.levels[0],
                                                                  b_name=self.bname,
                                                                  group_name='combined_condition',
                                                                  do_mean_over_trials=False,
                                                                  do_make_change=False
                                                                  )

    def fit_slopes(self, add_data=True, model=models.model_hier_normal_stim, do_make_change='subtract',
                   plot_index_cols=None, **kwargs):
        # TODO case with no group_name
        assert do_make_change in ['subtract', 'divide']
        self.bname = 'b_stim_per_condition'
        self.do_make_change = do_make_change
        if plot_index_cols is None:
            plot_index_cols = self.levels  # [-1]
        # By convention, top condition is first in list of levels:
        top_condition = self.levels[0]
        # Transform conditions to integers as required by numpyro:
        key = dict()
        for level in self.levels:
            self.data[level] = le.fit_transform(self.data[level])
            # Keep key for later use
            key[level] = dict(zip(range(len(le.classes_)), le.classes_))
        # Estimate model
        try:
            self.trace = fit_numpyro(y=self.data[self.y].values,
                                     stim_on=self.data[top_condition].values,
                                     treat=self.data[self.levels[2]].values if len(self.levels) > 2 else None,
                                     subject=self.data[self.levels[1]].values,
                                     progress_bar=False,
                                     model=model,
                                     n_draws=1000, num_chains=1,
                                     **kwargs)
        except TypeError:
            # assert that model() has kwarg stim_on, because this is slopes
            raise KeyError(f'Does your model {model} have "stim_on" argument? You asked for slopes!')
        self.levels.remove(top_condition)

        self.add_data = add_data
        if add_data:
            # Add data back
            df_result = utils.add_data_to_posterior(df_data=self.data,
                                                    trace=self.trace,
                                                    y=self.y,
                                                    index_cols=plot_index_cols,
                                                    condition_name=top_condition,
                                                    b_name=self.bname,
                                                    group_name=self.levels[-1],
                                                    do_make_change=do_make_change,
                                                    do_mean_over_trials=True,
                                                    )
        else:  # Just convert posterior to dataframe
            from bayes_window.utils import trace2df
            df_result = trace2df(self.trace, self.data, b_name=self.bname, group_name=self.levels[-1])

        # Back to human-readable labels
        [df_result[col].replace(key[col], inplace=True) for col in key.keys()
         if (not col == top_condition) and (col in df_result)]
        self.data_and_posterior = df_result

    def plot_posteriors_slopes(self, x=':O', color=':O', add_box=True, independent_axes=False, **kwargs):
        # Set some options
        self.independent_axes = independent_axes
        x = x or self.levels[-1]
        if x[-2] != ':':
            x += ':O'
        color = color or self.levels[0]

        # Plot posterior
        if hasattr(self, 'data_and_posterior'):
            add_data = self.add_data
            base_chart = alt.Chart(self.data_and_posterior)
            chart_p = plot_posterior(title=f'{self.y}',
                                     x=x,
                                     base_chart=base_chart,
                                     do_make_change=self.do_make_change)
        else:
            base_chart = alt.Chart(self.data)
            add_data = True  # Otherwise nothing to do

        if add_data:
            assert hasattr(self, 'data_and_posterior')
            chart_d = visualization.plot_data(x=x, y=f'{self.y} diff', color=color, add_box=add_box,
                                              base_chart=base_chart)
            self.chart = chart_p + chart_d
        else:
            self.chart = chart_p
        if independent_axes:
            self.chart = self.chart.resolve_scale(y='independent')
        return self.chart

    # TODO plot_posteriors_slopes and plot_posteriors_no_slope can be one
    def plot_posteriors_no_slope(self,
                                 x=None,
                                 add_data=False,
                                 independent_axes=True,
                                 color=None,
                                 detail='i_trial',
                                 **kwargs):
        reload(visualization)
        self.independent_axes = independent_axes
        x = x or self.levels[-1]
        color = color or self.levels[0]
        # TODO default for detail
        if hasattr(self, 'data_and_posterior'):
            base_chart = alt.Chart(self.data_and_posterior)
            # Plot posterior
            chart_p = visualization.plot_posterior(x=x,
                                                   do_make_change=False,
                                                   add_data=add_data,
                                                   title='Estimate',
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

            if not hasattr(self, 'data_and_posterior'):
                self.chart = chart_d  # we're done
            else:
                self.chart = chart_p + chart_d

        return self.chart

    def plot(self, **kwargs):
        # Convenience function
        if not hasattr(self, 'bname'):
            warnings.warn('No model has been fit. Defaulting to plotting "slopes" for data. Use .plot_slopes'
                          'or .plot_posteriors_no_slope to be explicit ')
            return visualization.plot_data(self.data, x=self.levels[0], y=self.y, color=self.levels[1], **kwargs)

        if self.bname == 'b_stim_per_condition':
            print('Plotting slopes')
            return BayesWindow.plot_posteriors_slopes(self, **kwargs)
        elif self.bname == 'mu_per_condition':
            print('Plotting posteriors')
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
            self.facetchart = self.chart.properties(width=width, height=height).facet(row=row, column=column)
        return self.facetchart

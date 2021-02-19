import bulwark.checks as ck
from altair.vegalite.v4.api import FacetChart, Chart, LayerChart
from bayes_window import models
from bayes_window.fitting import fit_numpyro
from bayes_window.generative_models import generate_fake_spikes
from bayes_window.utils import add_data_to_posterior
from bayes_window.visualization import fake_spikes_explore, plot_data_slope_trials
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform


def test_fake_spikes_explore():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    charts = fake_spikes_explore(df, df_monster, index_cols)
    for chart in charts:
        assert ((type(chart) == FacetChart) |
                (type(chart) == Chart) |
                (type(chart) == LayerChart)), print(f'{type(chart)}')
        chart.display()


def test_plot_data_and_posterior():
    # Make some data
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )

    for y in (set(df.columns) - set(index_cols)):
        # Estimate model
        trace = fit_numpyro(y=df[y].values,
                            stim_on=(df['stim']).astype(int).values,
                            treat=trans(df['neuron']),
                            subject=trans(df['mouse']),
                            progress_bar=True,
                            model=models.model_hier_normal_stim,
                            n_draws=100, num_chains=1, )

        # Add data back
        add_data_to_posterior(df,
                              trace=trace,
                              y=y,
                              index_cols=['neuron', 'stim', 'mouse', ],
                              condition_name='stim',
                              conditions=(0, 1),
                              b_name='b_stim_per_condition',  # for posterior
                              group_name='neuron'  # for posterior
                              )

        # Plot data and posterior
        # chart = plot_data_and_posterior(df=df_both, y=f'{y} diff', x='neuron', color='mouse', title=y)
        # assert ((type(chart) == FacetChart) |
        #         (type(chart) == Chart) |
        #         (type(chart) == LayerChart)), print(f'{type(chart)}')
        trace.to_dataframe().pipe(ck.has_no_nans)

        # chart.display()


def test_plot_data_slope_trials():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    chart = plot_data_slope_trials(df=df, x='stim:O', y='log_firing_rate', color='neuron:N', detail='i_trial')
    chart.display()

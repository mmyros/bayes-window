import bulwark.checks as ck
from altair.vegalite.v4.api import FacetChart, Chart, LayerChart
from bayes_window import models
from bayes_window.fitting import fit_numpyro
from bayes_window.generative_models import generate_fake_spikes
from bayes_window.utils import add_data_to_posterior
from bayes_window.visualization import fake_spikes_explore, plot_data_and_posterior, plot_data_slope_trials
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
        df_both = add_data_to_posterior(df,
                                        trace=trace,
                                        y=y,
                                        index_cols=['neuron', 'stim', 'mouse', ],
                                        condition_name='stim',
                                        conditions=(0, 1),
                                        b_name='b_stim_per_condition',  # for posterior
                                        group_name='neuron'  # for posterior
                                        )

        # Plot data and posterior
        chart = plot_data_and_posterior(df_both, y=f'{y} diff', x='neuron', color='mouse', title=y,
                                        hold_for_facet=False)
        assert ((type(chart) == FacetChart) |
                (type(chart) == Chart) |
                (type(chart) == LayerChart)), print(f'{type(chart)}')
        trace.to_dataframe().pipe(ck.has_no_nans)

        chart.display()


def test_facet_twoaxes_row():
    from bayes_window.visualization import AltairHack
    # Make some data
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    # y='isi'
    # trace = fit_numpyro(y=df[y].values,
    #                     stim_on=(df['stim']).astype(int).values,
    #                     treat=trans(df['neuron']),
    #                     subject=trans(df['mouse']),
    #                     progress_bar=True,
    #                     model=models.model_hier_normal_stim,
    #                     n_draws=100, num_chains=1, )
    chart = plot_data_slope_trials(df, y='firing_rate', x='stim:O',
                                   color='neuron:N',
                                   detail='i_trial')
    chart = AltairHack(charts=chart, data=df)
    chart = chart.facet(row='neuron_code')
    chart.display()
    # TODO hmm row does not work anumore


def test_facet_twoaxes_col():
    from bayes_window.visualization import AltairHack
    # Make some data
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    chart_data = plot_data_slope_trials(df, y='log_firing_rate', x='stim',
                                        color='neuron_code', detail='i_trial')

    chart = AltairHack(data=df, charts=[chart_data])
    chart = chart.facet(column='mouse_code')
    chart.display()


def test_facet_twoaxes_row_and_col():
    from bayes_window.visualization import AltairHack
    # Make some data
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    chart_data = plot_data_slope_trials(df, y='log_firing_rate', x='stim',
                                        color='neuron_code', detail='i_trial')

    chart = AltairHack(data=df, charts=[chart_data])
    chart = chart.facet(column='mouse_code', row='neuron_code')
    chart.display()


def test_plot_data_slope_trials():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    chart = plot_data_slope_trials(df, x='stim:O', y='log_firing_rate', color='neuron:N', detail='i_trial')
    chart.display()

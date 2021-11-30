from altair.vegalite.v4.api import FacetChart, Chart, LayerChart
from sklearn.preprocessing import LabelEncoder

from bayes_window import LMERegression
from bayes_window import fake_spikes_explore
from bayes_window.generative_models import generate_fake_spikes, generate_fake_lfp
from bayes_window.visualization import plot_data_slope_trials

trans = LabelEncoder().fit_transform


def test_lme_with_data():
    df, df_monster, index_cols, _ = generate_fake_lfp(mouse_response_slope=8,
                                                      n_trials=40)
    bw = LMERegression(df=df, y='Log power', treatment='stim', group='mouse')
    bw.fit(do_make_change='subtract')
    bw.plot()


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


def test_plot_data_slope_trials():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    chart = plot_data_slope_trials(df=df, x='stim:O', y='log_firing_rate', color='neuron:N', detail='i_trial')
    chart.display()

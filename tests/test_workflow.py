from bayes_window import models
from bayes_window.generative_models import generate_fake_spikes
from bayes_window.workflow import BayesWindow
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform


def test_estimate_posteriors():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse_code', 'neuron_code'), )
    bw.fit_conditions(model=models.model_single_lognormal)

    chart = bw.plot(x='stim:O', column='neuron', row='mouse', add_data=False)
    chart.display()
    chart = bw.plot(x='stim:O', column='neuron_code', row='mouse_code', add_data=False)
    chart.display()


def test_estimate_posteriors_data_overlay():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse_code', 'neuron_code'))
    bw.fit_conditions(model=models.model_single_lognormal)
    chart = bw.plot(x='stim:O', hold_for_facet=True, independent_axes=False, add_data=True,
                    column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_data_overlay_indep_axes():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse_code', 'neuron_code'))
    bw.fit_conditions(model=models.model_single_lognormal)
    chart = bw.plot(x='stim:O', hold_for_facet=True, independent_axes=True, add_data=True,
                    column='neuron_code', row='mouse_code')
    chart.display()


def test_plot():
    from bayes_window.workflow import BayesWindow

    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    chart = BayesWindow(df).plot(add_data=True)
    chart.display()

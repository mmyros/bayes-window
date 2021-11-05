from bayes_window.generative_models import *
from bayes_window.workflow import BayesWindow

trans = LabelEncoder().fit_transform
from bayes_window.utils import load_radon

import os

os.environ['bayes_window_test_mode'] = 'True'

df_radon = load_radon()

dfl, _, _, _ = generate_fake_lfp(n_trials=5)

df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                n_neurons=3,
                                                                n_mice=4,
                                                                dur=2, )


def test_fit_lme():
    window = BayesWindow(dfl, y='Log power', treatment='stim', group='mouse')
    window.fit_lme()
    window.regression_charts()
    # window.facet(row='mouse') # currently group is coded as a random variable


def test_fit_lme_w_condition():
    from numpy.linalg import LinAlgError
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=20,
                                                                    n_neurons=7,
                                                                    n_mice=6,
                                                                    dur=7,
                                                                    mouse_response_slope=12,
                                                                    overall_stim_response_strength=45)
    try:
        window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse', )
        assert window.fit_lme().data_and_posterior is not None
        window.regression_charts(x=window.condition[0]).display()
        window.plot(x='neuron_x_mouse').display()
        window.facet(column='neuron_x_mouse', width=300).display()
        assert len(window.charts) > 0
    except LinAlgError as e:
        print(e)


def test_fit_lme_w_data():
    window = BayesWindow(dfl, y='Log power', treatment='stim', group='mouse')
    window.fit_lme(do_make_change='divide')
    assert window.data_and_posterior is not None
    window.regression_charts().display()


# @mark.parametrize('add_data', [False]) # Adding data to LME does not work
def test_fit_lme_w_data_condition():
    df, df_monster, index_cols, _ = generate_fake_spikes(n_trials=25)

    window = BayesWindow(df, y='isi', treatment='stim', group='mouse',
                         condition='neuron_x_mouse')

    window.fit_lme(do_make_change='divide', )
    window.regression_charts().display()
    window.facet(column='neuron_x_mouse', width=300).display()

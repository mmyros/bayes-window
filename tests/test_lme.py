import os

from bayes_window.generative_models import generate_fake_spikes, generate_fake_lfp
from bayes_window.lme import LMERegression
from bayes_window.utils import load_radon

os.environ['bayes_window_test_mode'] = 'True'

df_radon = load_radon()

dfl, _, _, _ = generate_fake_lfp(n_trials=5)

df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                n_neurons=3,
                                                                n_mice=4,
                                                                dur=2, )


def test_fit_lme():
    window = LMERegression(df=dfl, y='Log power', treatment='stim', group='mouse')
    window.fit()
    window.plot()
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
        reg = LMERegression(df=df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse', )
        assert reg.fit().data_and_posterior is not None
        reg.plot(x=reg.window.condition[0]).display()
        reg.plot(x='neuron_x_mouse').display()
        reg.facet(column='neuron_x_mouse', width=300).display()
        assert len(reg.charts) > 0
    except LinAlgError as e:
        print(e)


def test_fit_lme_w_data():
    window = LMERegression(df=dfl, y='Log power', treatment='stim', group='mouse')
    window.fit(do_make_change='divide')
    assert window.data_and_posterior is not None
    window.plot().display()


# @mark.parametrize('add_data', [False]) # Adding data to LME does not work
def test_fit_lme_w_data_condition():
    df, df_monster, index_cols, _ = generate_fake_spikes(n_trials=25)

    window = LMERegression(df=df, y='isi', treatment='stim', group='mouse',
                           condition='neuron_x_mouse')

    window.fit(do_make_change='divide', )
    window.plot().display()
    window.facet(column='neuron_x_mouse', width=300).display()

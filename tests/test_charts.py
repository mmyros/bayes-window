from pathlib import Path

from bayes_window import models
from bayes_window.generative_models import *
from bayes_window.visualization import plot_posterior
from bayes_window.workflow import BayesWindow

trans = LabelEncoder().fit_transform
from bayes_window.utils import load_radon

from pytest import mark

df_radon = load_radon()


@mark.parametrize('do_make_change', ['subtract', 'divide', False])
@mark.parametrize('column', ['county', None])
def test_radon(do_make_change, column):
    window = BayesWindow(df_radon, y='radon', treatment='floor', condition=['county'])
    window.fit_slopes(do_make_change=do_make_change,
                      n_draws=100, num_chains=1, num_warmup=100)
    window.create_regression_charts(column=column)


def test_slopes():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=5,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=7,
                                                                    mouse_response_slope=16)
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(model=models.model_hierarchical, do_make_change='subtract',
                  fold_change_index_cols=('stim', 'mouse', 'neuron'))
    bw.create_regression_charts()


def test_fit_lme():
    df, df_monster, index_cols, _ = generate_fake_lfp(n_trials=25)
    bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    bw.fit_lme()

    bw.create_regression_charts()

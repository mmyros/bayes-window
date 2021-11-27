from bayes_window import models
from bayes_window.conditions import BayesConditions
from bayes_window.generative_models import *

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


def test_estimate_posteriors():
    window = BayesConditions(df=df, y='isi', treatment='stim', condition=['neuron_x_mouse', 'i_trial', ],
                             group='mouse', )
    window.fit(model=models.model_single)

    chart = window.plot(x='stim:O', column='neuron', row='mouse', )
    chart.display()
    chart = window.plot(x='stim:O', column='neuron', row='mouse', )
    chart.display()
    #window.plot_BEST()


def test_estimate_posteriors_data_overlay():
    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_single, )
    chart = window.plot(x='stim:O', independent_axes=False,
                        column='neuron', row='mouse')
    chart.display()


def test_estimate_posteriors_data_overlay_indep_axes():
    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_single, )

    chart = window.plot(x='stim:O', independent_axes=True,
                        column='neuron', row='mouse')
    chart.display()

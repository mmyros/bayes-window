import os

from bayes_window import models
from bayes_window.conditions import BayesConditions
from bayes_window.generative_models import generate_fake_spikes, generate_fake_lfp
from bayes_window.utils import load_radon

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
    # window.plot_BEST()


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


def test_conditions2():
    df.neuron = df.neuron.astype(int)
    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse', add_data=True)

    window.fit(model=models.model_single, num_chains=1)
    assert window.window.y in window.data_and_posterior
    window.plot(x='stim:O', independent_axes=False, add_data=True)


def test_fit_conditions():
    # TODO combined condition here somehow
    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit()


def test_plot_generic():
    # conditions:
    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_single)
    window.plot()


def test_facet():
    # conditions:
    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_single)
    window.plot(row='neuron', width=40)
    window.plot(x='neuron').facet(column='mouse')

#def test_model_quality():
    # conditions:
#    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
#    window.fit(model=models.model_single)
#    window.plot_model_quality()
    
    
def test_model_comparison():
    # conditions:
    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_single)
    window.explore_models()
    
    
def test_explore_model_kinds():
    # conditions:
    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_single)
    window.plot_model_quality()

def test_plot_no_slope_data_only():
    window = BayesConditions(df=df, y='isi', treatment='stim')
    chart = window.plot()
    chart.display()


def test_plot_slope_data_only():
    window = BayesConditions(df=df, y='isi', treatment='stim')
    chart = window.plot()
    chart.display()


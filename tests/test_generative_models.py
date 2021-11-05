from bayes_window import BayesRegression
from bayes_window.generative_models import generate_spikes_stim_strength


def test_generate_spikes_stim_strength():
    df = generate_spikes_stim_strength(overall_stim_response_strengths=range(10),
                                       n_trials=2,
                                       n_neurons=3,
                                       n_mice=4,
                                       dur=2, )

    window = BayesRegression(df=df, y='isi', treatment='stim_strength', condition='neuron')
    window.fit()
    window.chart.display()

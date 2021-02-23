from bayes_window.model_comparison import *


def test_run_methods():
    y_scores, true_slopes = run_methods(
        true_slopes=np.hstack([np.zeros(2), np.linspace(.03, 18, 2)]))
    plot_roc(y_scores, true_slopes).display()


def test_compare_models():
    from bayes_window.generative_models import generate_fake_spikes

    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    df['neuron']=df['neuron'].astype(int)
    df=df.rename({'neuron': 'treat', 'isi': 'y'},axis=1)
    df['subject'] = df['mouse_code']
    compare_models({'1': models.model_hier_normal_stim},
                   df=df,
                   data_cols=['stim','subject', 'treat', 'y'], )

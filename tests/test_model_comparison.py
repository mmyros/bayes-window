from bayes_window.model_comparison import *


def test_run_methods():
    res = run_conditions(
        true_slopes=np.hstack([np.zeros(2), np.linspace(8.03, 18, 3)]),
        n_trials=[7],
        parallel=True
    )
    plot_roc(res)[0].display()
    plot_roc(res)[1].display()


def test_compare_models():
    from bayes_window.generative_models import generate_fake_spikes

    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    df['neuron'] = df['neuron'].astype(int)
    df = df.rename({'neuron': 'condition', 'stim': 'treatment', 'isi': 'y'}, axis=1)
    df['subject'] = df['mouse_code']
    compare_models({'1': models.model_hierarchical},
                   df=df,
                   data_cols=['treatment', 'subject', 'condition', 'y'],
                   index_cols=('mouse', 'condition', 'treatment')
                   )

from bayes_window.generative_models import generate_fake_spikes
from bayes_window.model_comparison import *
from pytest import mark


def test_run_methods():
    res = run_conditions(
        true_slopes=np.hstack([np.zeros(2), np.linspace(8.03, 18, 3)]),
        n_trials=[7],
        parallel=True
    )
    plot_roc(res)[0].display()
    plot_roc(res)[1].display()


@mark.parametrize('parallel', [False, True])
def test_compare_models(parallel):
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=10,
                                                                    n_neurons=10,
                                                                    n_mice=4,
                                                                    dur=2, )
    compare_models(df=df,
                   models={'no_neuron': models.model_hierarchical,
                           'no_neuron_or_teratment': models.model_hierarchical,
                           'no-treatment': models.model_hierarchical,
                           'treatment': models.model_hierarchical,
                           'student': models.model_hierarchical,
                           'lognogmal': models.model_hierarchical,

                           },
                   extra_model_args=[{'treatment': 'stim', 'condition': None},
                                     {'treatment': None, 'condition': None},
                                     {'treatment': None, 'condition': 'neuron'},
                                     {'treatment': 'stim', 'condition': 'neuron'},
                                     {'treatment': 'stim', 'condition': 'neuron', 'dist_y': 'student'},
                                     {'treatment': 'stim', 'condition': 'neuron', 'dist_y': 'lognormal'}, ],
                   y='isi',
                   group='mouse',
                   parallel=parallel
                   )


@mark.parametrize('parallel', [False, True])
def test_compare_models2(parallel):
    df, df_monster, index_cols, _ = generate_fake_lfp(mouse_response_slope=13,
                                                      n_trials=40)
    compare_models(df=df,
                   models={
                       'no_teratment': models.model_hierarchical,
                       'no_group': models.model_hierarchical,
                       'full_normal': models.model_hierarchical,
                       'full_student': models.model_hierarchical,
                       'full_lognogmal': models.model_hierarchical,

                   },
                   extra_model_args=[
                       {'treatment': None},
                       {'group': None},
                       {'treatment': 'stim'},
                       {'treatment': 'stim', 'dist_y': 'student'},
                       {'treatment': 'stim', 'dist_y': 'lognormal'},
                   ],
                   y='isi',
                   condition=None,
                   parallel=False
                   )

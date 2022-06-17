from pytest import mark

from bayes_window.generative_models import generate_fake_spikes
from bayes_window.model_comparison import *


@mark.serial
def test_compare_models():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=10,
                                                                    n_neurons=10,
                                                                    n_mice=4,
                                                                    dur=2, )
    compare_models(df=df,
                   models={'no_neuron': models.model_hierarchical,
                           'no_neuron_or_treatment': models.model_hierarchical,
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
                                     {'treatment': 'stim', 'condition': 'neuron', 'dist_y': 'lognormal'}],
                   y='isi',
                   group='mouse',
                   parallel=True,
                   plotose=True,
                   # num_chains=1,
                   num_warmup=100,
                   n_draws=1000,
                   )


@mark.serial
def test_compare_models_serial():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=10,
                                                                    n_neurons=10,
                                                                    n_mice=4,
                                                                    dur=2, )
    compare_models(df=df,
                   models={'no_neuron': models.model_hierarchical,
                           'no_neuron_or_treatment': models.model_hierarchical,
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
                                     {'treatment': 'stim', 'condition': 'neuron', 'dist_y': 'lognormal'}],
                   y='isi',
                   group='mouse',
                   parallel=False,
                   plotose=True,
                   # num_chains=1,
                   num_warmup=100,
                   n_draws=1000,
                   )


@mark.serial
def test_compare_models2():
    df, df_monster, index_cols, _ = generate_fake_lfp(mouse_response_slope=13,
                                                      n_trials=4)
    compare_models(df=df,
                   models={
                       'no_treatment': models.model_hierarchical,
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
                   parallel=True
                   )

@mark.serial
def test_compare_models2_serial():
    df, df_monster, index_cols, _ = generate_fake_lfp(mouse_response_slope=13,
                                                      n_trials=4)
    compare_models(df=df,
                   models={
                       'no_treatment': models.model_hierarchical,
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

# def test_compare_modesl3():
#     from bayes_window import model_comparison
#
#     res = model_comparison.run_conditions(true_slopes=np.hstack([np.zeros(5),
#                                                                  np.tile(np.linspace(.2, 20, 5), 3)]),
#                                           n_trials=np.linspace(15, 70, 5).astype(int),
#                                           trial_baseline_randomness=np.linspace(.2, 11, 3),
#                                           ys=('Power', 'Log power', ),
#                                           parallel=True)



@mark.serial
def test_run_methods():
    res = run_conditions(
        true_slopes=np.hstack([np.zeros(2), np.linspace(8.03, 18, 3)]),
        n_trials=[9],
        parallel=True,
        ys=('Log power', 'Power')
    )
    res=make_roc_auc(res)
    plot_roc(res)[0].display()
    plot_roc(res)[1].display()


def test_run_methods_serial():
    res = run_conditions(
        true_slopes=np.hstack([np.zeros(2), 10]),
        n_trials=[10],
        parallel=False,
        ys=('Power', )
    )
    res_confusion=make_confusion_matrix(res, groups=['method'])
    plot_confusion(res_confusion)
    res=make_roc_auc(res)
    plot_roc(res)[0].display()
    plot_roc(res)[1].display()

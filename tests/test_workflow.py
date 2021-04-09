from pathlib import Path

from bayes_window import models
from bayes_window.generative_models import *
from bayes_window.visualization import plot_posterior
from bayes_window.workflow import BayesWindow

trans = LabelEncoder().fit_transform
from bayes_window.utils import load_radon

from pytest import mark

df_radon = load_radon()


@mark.parametrize('add_posterior_density', [True, False], )
@mark.parametrize('add_data', [True, False])
@mark.parametrize('add_box', [True, False])
@mark.parametrize('add_condition_slope', [True, False])
@mark.parametrize('add_group_slope', [True, False])
@mark.parametrize('do_mean_over_trials', [True, False])
@mark.parametrize('do_make_change', ['subtract', 'divide', False])
def test_radon(add_posterior_density, add_data, add_box, add_condition_slope,
               add_group_slope, do_mean_over_trials, do_make_change
               ):
    window = BayesWindow(df_radon, y='radon', treatment='floor', condition=['county'])
    # window.plot(x='county').facet(row='floor').display()
    window.fit_slopes(add_condition_slope=add_condition_slope, do_mean_over_trials=do_mean_over_trials,
                      add_group_slope=add_group_slope, do_make_change=do_make_change,
                      n_draws=100, num_chains=1, num_warmup=100)
    # window.plot().display()
    window.plot(x=':O', add_data=add_data, add_box=add_box, add_posterior_density=add_posterior_density).display()


def test_slopes_dont_make_change():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=5,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=7,
                                                                    mouse_response_slope=16)
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    try:
        bw.fit_slopes(model=models.model_hierarchical, do_make_change=False,
                      fold_change_index_cols=('stim', 'mouse', 'neuron'))
    except ValueError:
        pass


def test_fit_lme():
    df, df_monster, index_cols, _ = generate_fake_lfp(n_trials=25)
    bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    bw.fit_lme()
    bw.plot_posteriors_slopes()
    bw.facet(row='mouse')


def test_fit_lme_w_condition():
    from numpy.linalg import LinAlgError
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=20,
                                                                    n_neurons=7,
                                                                    n_mice=6,
                                                                    dur=7,
                                                                    mouse_response_slope=12,
                                                                    overall_stim_response_strength=45)
    try:
        bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse', )
        assert bw.fit_lme().data_and_posterior is not None
        bw.plot_posteriors_slopes()
        bw.facet(column='neuron_x_mouse', width=300)
    except LinAlgError as e:
        print(e)


def test_fit_lme_w_data():
    df, df_monster, index_cols, _ = generate_fake_lfp(n_trials=25)

    bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    bw.fit_lme(do_make_change='divide')
    assert bw.data_and_posterior is not None
    bw.plot_posteriors_slopes()


def test_fit_lme_w_data_condition():
    df, df_monster, index_cols, _ = generate_fake_spikes(n_trials=25)

    bw = BayesWindow(df, y='isi', treatment='stim', group='mouse', condition='neuron')

    bw.fit_lme(do_make_change='divide')
    bw.plot_posteriors_slopes()
    bw.facet(column='neuron_x_mouse', width=300)


def test_estimate_posteriors():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse', )
    bw.fit_conditions(model=models.model_single, add_data=False)

    chart = bw.plot(x='stim:O', column='neuron', row='mouse', )
    chart.display()
    chart = bw.plot(x='stim:O', column='neuron_code', row='mouse_code', )
    chart.display()


def test_estimate_posteriors_data_overlay():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse')
    bw.fit_conditions(model=models.model_single, add_data=False)
    chart = bw.plot(x='stim:O', independent_axes=False,
                    column='neuron_code', row='mouse_code')
    chart.display()


# def test_estimate_posteriors_data_overlay_sa():
#     df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
#                                                                     n_neurons=3,
#                                                                     n_mice=4,
#                                                                     dur=2, )
#     from numpyro.infer import HMC, SA, BarkerMH, NUTS
#     bw = BayesWindow(df, y='firing_rate', treatment='stim', condition='neuron_x_mouse', group='mouse')
#     bw.fit_slopes(add_data=True, model=models.model_hierarchical, do_make_change='subtract',
#                   progress_bar=False,
#                   dist_y='exponential',
#                   sampler=SA,
#                   add_group_slope=True, add_group_intercept=False,
#                   fold_change_index_cols=('stim', 'mouse', 'neuron','neuron_x_mouse'))
#
#     bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)
#     bw.facet(column='mouse',width=200,height=200).display()


def test_estimate_posteriors_data_overlay_indep_axes():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse')
    bw.fit_conditions(model=models.model_single, add_data=True, )

    chart = bw.plot(x='stim:O', independent_axes=True,
                    column='neuron_code', row='mouse_code')
    chart.display()


def test_plot():
    from bayes_window.workflow import BayesWindow

    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    chart = BayesWindow(df, y='isi', treatment='stim').plot()
    chart.display()


def test_estimate_posteriors_slope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse', )
    bw.fit_slopes(models.model_hierarchical)

    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse')
    chart.display()
    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_slope_uneven_n_data_per_condition():
    # Trying to reproduce Uneven number of entries in conditions! Try setting do_take_mean=True
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=10,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    df = df.drop(df[(df['i_trial'] == 0) &
                    # (df['neuron_x_mouse'] == '0m0bayes') &
                    (df['stim'] == 0)].index)
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse')
    bw.fit_slopes(models.model_hierarchical, do_make_change='divide')

    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse')
    chart.display()
    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_slope_groupslope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse', )
    bw.fit_slopes(models.model_hierarchical, add_group_slope=True, add_group_intercept=True)

    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse')
    chart.display()
    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse_code')
    chart.display()


# def test_estimate_posteriors_two_conditions_w_wo_data():
#     df = generate_spikes_stim_types(mouse_response_slope=3,
#                                     n_trials=2,
#                                     n_neurons=3,
#                                     n_mice=4,
#                                     dur=2, )
#
#     bw1 = BayesWindow(df, y='isi', treatment='stim', condition=['neuron_code', 'stim_strength'], group='mouse', )
#     bw1.fit_slopes(model=models.model_hierarchical, do_mean_over_trials=False, add_data=False)
#
#     bw2 = BayesWindow(df, y='isi', treatment='stim', condition=['neuron_code', 'stim_strength'], group='mouse', )
#     bw2.fit_slopes(model=models.model_hierarchical, do_mean_over_trials=False, add_data=True)
#
#     assert (bw1.data_and_posterior['neuron_code'] == bw2.data_and_posterior['neuron_code']).all()
#     assert (bw1.data_and_posterior['stim_strength'] == bw2.data_and_posterior['stim_strength']).all()


from pytest import mark


@mark.parametrize('add_data', [False, True])
def test_estimate_posteriors_two_conditions_no_add_data(add_data):
    df = generate_spikes_stim_types(mouse_response_slope=3,
                                    n_trials=2,
                                    n_neurons=3,
                                    n_mice=4,
                                    dur=2, )

    bw = BayesWindow(df, y='isi', treatment='stim', condition=['neuron_code', 'stim_strength'], group='mouse', )
    bw.fit_slopes(model=models.model_hierarchical, fold_change_index_cols=None, do_mean_over_trials=False)
    for condition_name in bw.condition:
        assert condition_name in bw.data_and_posterior.columns, f'{condition_name} not in window.condition'
    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse')
    chart.display()
    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_two_conditions_add_data():
    df = generate_spikes_stim_types(mouse_response_slope=3,
                                    n_trials=2,
                                    n_neurons=3,
                                    n_mice=4,
                                    dur=2, )

    bw = BayesWindow(df, y='isi', treatment='stim', condition=['neuron_code', 'stim_strength'], group='mouse', )
    bw.fit_slopes(model=models.model_hierarchical, fold_change_index_cols=None, do_mean_over_trials=False, num_chains=1,
                  num_warmup=100, n_draws=100)
    for condition_name in bw.condition:
        assert condition_name in bw.data_and_posterior.columns, f'{condition_name} not in window.condition'
    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse')
    chart.display()
    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_data_overlay_slope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse')
    bw.fit_slopes(model=models.model_hierarchical)
    chart = bw.plot_posteriors_slopes(independent_axes=False, add_posterior_density=False)
    chart.display()
    bw.facet(column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_data_overlay_indep_axes_slope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse')
    bw.fit_slopes(model=models.model_hierarchical)
    chart = bw.plot_posteriors_no_slope(independent_axes=True)
    chart.display()
    chart = bw.facet(column='neuron_code', row='mouse')
    chart.display()


def test_plot_no_slope_data_only():
    from bayes_window.workflow import BayesWindow

    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim')
    chart = window.plot_posteriors_no_slope()
    chart.display()


def test_plot_slope_data_only():
    from bayes_window.workflow import BayesWindow

    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim')
    chart = window.plot_posteriors_no_slope()
    chart.display()


def test_fit_conditions():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    # TODO combined condition here somehow
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_conditions(add_data=True)


def test_fit_slopes():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(model=models.model_hierarchical)


def test_plot_slopes():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(model=models.model_hierarchical)
    bw.plot()


def test_plot_slopes_2levelslope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse')
    bw.fit_slopes(model=models.model_hierarchical, add_group_slope=True)
    bw.plot().display()


def test_plot_posteriors_no_slope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(model=models.model_hierarchical)
    bw.plot_posteriors_slopes()


def test_plot_generic():
    # Slopes:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(model=models.model_hierarchical)
    bw.plot()
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(model=models.model_hierarchical)
    bw.plot()
    # conditions:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_conditions(model=models.model_single)
    bw.plot()


def test_facet():
    # Slopes:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(model=models.model_hierarchical)
    bw.plot(add_posterior_density=False).facet('neuron', width=40)

    # conditions:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_conditions(model=models.model_single)
    bw.plot().facet('neuron', width=40)


def test_single_condition_withdata():
    df, df_monster, index_cols, _ = generate_fake_lfp()
    bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    bw.fit_slopes(model=models.model_hier_stim_one_codition, do_make_change='divide', dist_y='normal')
    plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
    bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()

    # Without data again
    bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    bw.fit_slopes(model=models.model_hier_stim_one_codition, do_make_change='divide', dist_y='normal')
    plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
    bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()

    # With data again
    bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    bw.fit_slopes(model=models.model_hier_stim_one_codition, do_make_change='divide', dist_y='normal')
    plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
    bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()


def test_single_condition_nodata():
    df, df_monster, index_cols, _ = generate_fake_lfp()
    bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    bw.fit_slopes(model=models.model_hier_stim_one_codition, do_make_change='divide', dist_y='normal')
    plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
    bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()


def test_single_condition_nodata_dists():
    df, df_monster, index_cols, _ = generate_fake_lfp()
    for dist in ['normal', 'lognormal', 'student']:
        bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
        bw.fit_slopes(model=models.model_hier_stim_one_codition, do_make_change='divide', dist_y=dist)
        plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
        bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()


# @mark.parametrize('condition', [None, 'neuron'])
# @mark.parametrize('parallel', [False, True])
@mark.parametrize('add_group_slope', [False, ])  # True doesnt work in GHA
def test_explore_models(add_group_slope):
    parallel = False
    # Slopes:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    conditions_to_test = [None]
    if add_group_slope:
        conditions_to_test.append('neuron')
    for condition in conditions_to_test:
        bw = BayesWindow(df, y='isi', treatment='stim', condition=condition, group='mouse')
        bw.fit_slopes(model=models.model_hierarchical, num_chains=1)
        bw.explore_models(parallel=parallel, add_group_slope=add_group_slope)


def test_chirp_data():
    df = pd.read_csv(Path('tests') / 'test_data' / 'chirp_power.csv')
    window = BayesWindow(df, y='Log power',
                         treatment='stim_on',
                         condition='Condition code',
                         group='Subject')
    window.fit_slopes(model=models.model_hierarchical, fold_change_index_cols=['Condition code',
                                                                               'Brain region', 'Stim phase', 'stim_on',
                                                                               'Fid', 'Subject', 'Inversion'],
                      do_mean_over_trials=True, num_chains=1, n_draws=100, num_warmup=100)
    window.plot_posteriors_slopes(x='Stim phase', color='Fid', independent_axes=True)

    window.data_and_posterior


def test_chirp_data1():
    df = pd.read_csv(Path('tests') / 'test_data' / 'chirp_power.csv')
    window = BayesWindow(df, y='Log power',
                         treatment='stim_on',
                         condition=['Stim phase', 'Inversion'],
                         group='Subject')
    window.fit_slopes(model=models.model_hierarchical, fold_change_index_cols=['Stim phase', 'Inversion',
                                                                               # 'Brain region', 'stim_on', 'Fid', 'Subject'
                                                                               ], do_mean_over_trials=True,
                      num_chains=1, n_draws=100, num_warmup=100)
    window.plot_posteriors_slopes(x='Stim phase', color='Fid', independent_axes=True)


def test_chirp_data2():
    df = pd.read_csv(Path('tests') / 'test_data' / 'chirp_power.csv')
    window = BayesWindow(df, y='Log power',
                         treatment='stim_on',
                         condition=['Condition code'],
                         group='Subject')
    window.fit_slopes(model=models.model_hierarchical, fold_change_index_cols=[  # 'Condition code',
        'Brain region', 'Stim phase', 'stim_on', 'Fid', 'Subject', 'Inversion'], do_mean_over_trials=True, num_chains=1,
                      n_draws=100, num_warmup=100)
    window.plot_posteriors_slopes(x='Stim phase', color='Fid', independent_axes=True)


def test_conditions2():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=5,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=7,
                                                                    mouse_response_slope=16)
    df.neuron = df.neuron.astype(int)
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')

    window.fit_conditions(model=models.model_single, num_chains=1)
    assert window.y in window.data_and_posterior
    window.plot_posteriors_no_slope(x='stim:O', independent_axes=False, add_data=True)


def random_tests():
    # TODO make a notebook for this
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(model=models.model_hierarchical, num_chains=1)
    bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True, add_box=False)

    bw.plot_posteriors_slopes(add_box=True, independent_axes=False, x='neuron:O', color='mouse')

    bw.plot_posteriors_slopes(add_box=False, independent_axes=True, x='neuron:O', color='mouse')

    bw.plot_posteriors_slopes(independent_axes=False, x='neuron:O', color='mouse')

    chart = bw.plot_posteriors_slopes(add_box=True, independent_axes=True, x='neuron:O', color='mouse')

    chart

    chart.resolve_scale(y='independent')

    bw.facet(column='neuron')

    window = BayesWindow(df, y='isi',
                         treatment='stim',
                         condition='neuron',
                         group='mouse')
    window.fit_slopes(model=models.model_hierarchical, num_chains=1)
    c = window.plot_posteriors_slopes(x='neuron', color='i_trial')

    window.plot_posteriors_slopes()  # x='Stim phase', color='Fid')#,independent_axes=True)
    window.facet(column='neuron', row='mouse')

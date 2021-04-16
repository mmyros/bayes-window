from pathlib import Path

from bayes_window import models
from bayes_window.generative_models import *
from bayes_window.visualization import plot_posterior
from bayes_window.workflow import BayesWindow

trans = LabelEncoder().fit_transform
from bayes_window.utils import load_radon

from pytest import mark

df_radon = load_radon()


# @mark.parametrize('add_data', [True, False])
# @mark.parametrize('add_box', [True, False])
@mark.parametrize('add_condition_slope', [True, False])
@mark.parametrize('add_group_slope', [False, True])
# @mark.parametrize('do_mean_over_trials', [True, False])
@mark.parametrize('do_make_change', ['subtract', 'divide', False])
def test_radon(
               # add_data, add_box,
               add_condition_slope,
               add_group_slope,
               # do_mean_over_trials,
               do_make_change
               ):
    window = BayesWindow(df_radon, y='radon', treatment='floor', condition=['county'])
    # window.plot(x='county').facet(row='floor').display()
    window.fit_slopes(add_condition_slope=add_condition_slope, #do_mean_over_trials=do_mean_over_trials,
                      add_group_slope=add_group_slope, do_make_change=do_make_change,
                      n_draws=100, num_chains=1, num_warmup=100)
    # window.plot().display()
    window.plot(x=':O', #add_data=add_data,
                ).display()
    window.regression_charts()


@mark.parametrize('do_make_change', [False, 'divide', 'subtract'])
@mark.parametrize('detail', [None, ':O', 'i_trial'])
def test_slopes(do_make_change, detail):
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=5,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=7,
                                                                    mouse_response_slope=16)
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse', detail=detail)
    # try:
    window.fit_slopes(model=models.model_hierarchical, do_make_change=do_make_change,)
                      # fold_change_index_cols=('stim', 'mouse', 'neuron_x_mouse'))
    window.chart.display()
    window.chart_data_box_detail.display()


def test_fit_lme():
    df, df_monster, index_cols, _ = generate_fake_lfp(n_trials=25)
    window = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    window.fit_lme()
    window.regression_charts()
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
        window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse', )
        assert window.fit_lme().data_and_posterior is not None
        window.regression_charts().display()
        window.facet(column='neuron_x_mouse', width=300).display()
    except LinAlgError as e:
        print(e)


def test_fit_lme_w_data():
    df, df_monster, index_cols, _ = generate_fake_lfp(n_trials=25)

    window = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    window.fit_lme(do_make_change='divide')
    assert window.data_and_posterior is not None
    window.regression_charts().display()


def test_fit_lme_w_data_condition():
    df, df_monster, index_cols, _ = generate_fake_spikes(n_trials=25)

    window = BayesWindow(df, y='isi', treatment='stim', group='mouse', condition='neuron_x_mouse')

    window.fit_lme(do_make_change='divide')
    window.regression_charts().display()
    window.facet(column='neuron_x_mouse', width=300).display()


def test_estimate_posteriors():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse', )
    window.fit_conditions(model=models.model_single, add_data=False)

    chart = window.plot(x='stim:O', column='neuron', row='mouse', )
    chart.display()
    chart = window.plot(x='stim:O', column='neuron_code', row='mouse_code', )
    chart.display()


def test_estimate_posteriors_data_overlay():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse')
    window.fit_conditions(model=models.model_single, add_data=False)
    chart = window.plot(x='stim:O', independent_axes=False,
                        column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_data_overlay_indep_axes():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse')
    window.fit_conditions(model=models.model_single, add_data=True, )

    chart = window.plot(x='stim:O', independent_axes=True,
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
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse', )
    window.fit_slopes(models.model_hierarchical)

    chart = window.plot(x='neuron_code', column='neuron_code', row='mouse')
    chart.display()
    chart = window.plot(x='neuron_code', column='neuron_code', row='mouse_code')
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
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse')
    window.fit_slopes(models.model_hierarchical, do_make_change='divide')

    chart = window.plot(x='neuron_code', column='neuron_code', row='mouse')
    chart.display()
    chart = window.plot(x='neuron_code', column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_slope_groupslope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse', )
    window.fit_slopes(models.model_hierarchical, add_group_slope=True, add_group_intercept=True)

    chart = window.plot(x='neuron_code', column='neuron_code', row='mouse')
    chart.display()
    chart = window.plot(x='neuron_code', column='neuron_code', row='mouse_code')
    chart.display()


# def test_estimate_posteriors_two_conditions_w_wo_data():
#     df = generate_spikes_stim_types(mouse_response_slope=3,
#                                     n_trials=2,
#                                     n_neurons=3,
#                                     n_mice=4,
#                                     dur=2, )
#
#     window1 = BayesWindow(df, y='isi', treatment='stim', condition=['neuron_code', 'stim_strength'], group='mouse', )
#     window1.fit_slopes(model=models.model_hierarchical, do_mean_over_trials=False, add_data=False)
#
#     window2 = BayesWindow(df, y='isi', treatment='stim', condition=['neuron_code', 'stim_strength'], group='mouse', )
#     window2.fit_slopes(model=models.model_hierarchical, do_mean_over_trials=False, add_data=True)
#
#     assert (window1.data_and_posterior['neuron_code'] == window2.data_and_posterior['neuron_code']).all()
#     assert (window1.data_and_posterior['stim_strength'] == window2.data_and_posterior['stim_strength']).all()


from pytest import mark


@mark.parametrize('add_data', [False, True])
def test_estimate_posteriors_two_conditions_no_add_data(add_data):
    df = generate_spikes_stim_types(mouse_response_slope=3,
                                    n_trials=2,
                                    n_neurons=3,
                                    n_mice=4,
                                    dur=2, )

    window = BayesWindow(df, y='isi', treatment='stim', condition=['neuron_code', 'stim_strength'], group='mouse', )
    window.fit_slopes(model=models.model_hierarchical, fold_change_index_cols=None, do_mean_over_trials=False)
    for condition_name in window.condition:
        assert condition_name in window.data_and_posterior.columns, f'{condition_name} not in window.condition'
    chart = window.plot(x='neuron_code', column='neuron_code', row='mouse')
    chart.display()
    chart = window.plot(x='neuron_code', column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_two_conditions_add_data():
    df = generate_spikes_stim_types(mouse_response_slope=3,
                                    n_trials=2,
                                    n_neurons=3,
                                    n_mice=4,
                                    dur=2, )

    window = BayesWindow(df, y='isi', treatment='stim', condition=['neuron_code', 'stim_strength'], group='mouse', )
    window.fit_slopes(model=models.model_hierarchical, fold_change_index_cols=None, do_mean_over_trials=False,
                      num_chains=1,
                      num_warmup=100, n_draws=100)
    for condition_name in window.condition:
        assert condition_name in window.data_and_posterior.columns, f'{condition_name} not in window.condition'
    chart = window.plot(x='neuron_code', column='neuron_code', row='mouse')
    chart.display()
    chart = window.plot(x='neuron_code', column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_data_overlay_slope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)
    chart = window.regression_charts(independent_axes=False)
    chart.display()
    window.facet(column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_data_overlay_indep_axes_slope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)
    chart = window.plot_posteriors_no_slope(independent_axes=True)
    chart.display()
    chart = window.facet(column='neuron_code', row='mouse')
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
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_conditions(add_data=True)


def test_fit_slopes():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)


def test_plot_slopes():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)
    window.plot()


def test_plot_slopes_2levelslope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse')
    window.fit_slopes(model=models.model_hierarchical, add_group_slope=True)
    window.plot().display()


def test_plot_posteriors_no_slope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)
    window.regression_charts()


def test_plot_generic():
    # Slopes:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)
    window.plot()
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)
    window.plot()
    # conditions:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_conditions(model=models.model_single)
    window.plot()


def test_facet():
    # Slopes:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)
    window.plot().facet('neuron', width=40)

    # conditions:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_conditions(model=models.model_single)
    window.plot().facet('neuron', width=40)


def test_single_condition_withdata():
    df, df_monster, index_cols, _ = generate_fake_lfp()
    window = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    window.fit_slopes(model=models.model_hierarchical, do_make_change='divide', dist_y='normal')
    alt.layer(*plot_posterior(df=window.data_and_posterior, title=f'Log power', )).display()
    window.regression_charts( independent_axes=True).display()

    # Without data again
    window = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    window.fit_slopes(model=models.model_hierarchical, do_make_change='divide', dist_y='normal')
    alt.layer(*plot_posterior(df=window.data_and_posterior, title=f'Log power', )).display()
    window.regression_charts( independent_axes=True).display()

    # With data again
    window = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    window.fit_slopes(model=models.model_hierarchical, do_make_change='divide', dist_y='normal')
    alt.layer(*plot_posterior(df=window.data_and_posterior, title=f'Log power', )).display()
    window.regression_charts( independent_axes=True).display()


def test_single_condition_nodata():
    df, df_monster, index_cols, _ = generate_fake_lfp()
    window = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    window.fit_slopes(model=models.model_hierarchical, do_make_change='divide', dist_y='normal')
    alt.layer(*plot_posterior(df=window.data_and_posterior, title=f'Log power', )).display()
    window.regression_charts( independent_axes=True).display()


def test_single_condition_nodata_dists():
    df, df_monster, index_cols, _ = generate_fake_lfp()
    for dist in ['normal', 'lognormal', 'student']:
        window = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
        window.fit_slopes(model=models.model_hierarchical, do_make_change='divide', dist_y=dist)
        alt.layer(*plot_posterior(df=window.data_and_posterior, title=f'Log power', )).display()
        window.regression_charts( independent_axes=True).display()


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
        window = BayesWindow(df, y='isi', treatment='stim', condition=condition, group='mouse')
        window.fit_slopes(model=models.model_hierarchical, num_chains=1)
        window.explore_models(parallel=parallel, add_group_slope=add_group_slope)


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
    window.regression_charts(x='Stim phase', color='Fid', independent_axes=True)

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
    window.regression_charts(x='Stim phase', color='Fid', independent_axes=True)


def test_chirp_data2():
    df = pd.read_csv(Path('tests') / 'test_data' / 'chirp_power.csv')
    window = BayesWindow(df, y='Log power',
                         treatment='stim_on',
                         condition=['Condition code'],
                         group='Subject')
    window.fit_slopes(model=models.model_hierarchical, fold_change_index_cols=[  # 'Condition code',
        'Brain region', 'Stim phase', 'stim_on', 'Fid', 'Subject', 'Inversion'], do_mean_over_trials=True, num_chains=1,
                      n_draws=100, num_warmup=100)
    window.regression_charts(x='Stim phase', color='Fid', independent_axes=True)


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
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical, num_chains=1)
    window.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)

    window.regression_charts( independent_axes=False, x='neuron:O', color='mouse')

    window.regression_charts(add_box=False, independent_axes=True, x='neuron:O', color='mouse')

    window.regression_charts(independent_axes=False, x='neuron:O', color='mouse')

    chart = window.regression_charts( independent_axes=True, x='neuron:O', color='mouse')

    chart

    chart.resolve_scale(y='independent')

    window.facet(column='neuron')

    window = BayesWindow(df, y='isi',
                         treatment='stim',
                         condition='neuron',
                         group='mouse')
    window.fit_slopes(model=models.model_hierarchical, num_chains=1)
    c = window.regression_charts(x='neuron', color='i_trial')

    window.regression_charts()  # x='Stim phase', color='Fid')#,independent_axes=True)
    window.facet(column='neuron', row='mouse')


def test_data_replacement1():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=40,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=7,
                                                                    mouse_response_slope=16)
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse',
                              detail='i_trial')
    window.fit_slopes(model=models.model_hierarchical, do_make_change='subtract',
                      add_condition_slope=True,
                      add_group_slope=True)
    assert window.data_and_posterior.dropna(subset=['mu_intercept_per_group center interval'])['mouse'].unique().size==4



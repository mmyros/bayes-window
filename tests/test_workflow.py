from pathlib import Path

from bayes_window import models
from bayes_window.conditions import BayesConditions
from bayes_window.generative_models import *
from bayes_window.visualization import plot_posterior
from bayes_window.workflow import BayesWindow

trans = LabelEncoder().fit_transform
from bayes_window.utils import load_radon

from pytest import mark
import os

os.environ['bayes_window_test_mode'] = 'True'

df_radon = load_radon()

dfl, _, _, _ = generate_fake_lfp(n_trials=5)

df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                n_neurons=3,
                                                                n_mice=4,
                                                                dur=2, )


# @mark.parametrize('add_data', [True, False])
# @mark.parametrize('add_box', [True, False])
@mark.parametrize('add_condition_slope', [False, True])
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
    window.fit_slopes(add_condition_slope=add_condition_slope,  # do_mean_over_trials=do_mean_over_trials,
                      add_group_slope=add_group_slope, do_make_change=do_make_change,
                      n_draws=100, num_chains=1, num_warmup=100)
    # window.plot().display()
    window.plot(x=':O',  # add_data=add_data,
                ).display()
    window.regression_charts()
    window.chart_posterior_kde.display()
    # window.chart_data_boxplot.display() # Fold change won't work, bc uneven number of entries
    assert len(window.charts) > 3  # Should include kde


@mark.parametrize('do_make_change', [False, 'divide', 'subtract'])
@mark.parametrize('detail', [None, ':O', 'i_trial'])
def test_slopes(do_make_change, detail):
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse', detail=detail)
    # try:
    window.fit_slopes(model=models.model_hierarchical, do_make_change=do_make_change, )
    # fold_change_index_cols=('stim', 'mouse', 'neuron_x_mouse'))
    window.chart.display()
    window.chart_data_box_detail.display()


def test_estimate_posteriors():
    window = BayesConditions(df=df, y='isi', treatment='stim', condition=['neuron_x_mouse', 'i_trial', ],
                             group='mouse', )
    window.fit(model=models.model_single)

    chart = window.plot(x='stim:O', column='neuron', row='mouse', )
    chart.display()
    chart = window.plot(x='stim:O', column='neuron', row='mouse', )
    chart.display()


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


def test_plot():
    from bayes_window.workflow import BayesWindow

    chart = BayesWindow(df, y='isi', treatment='stim').plot()
    chart.display()


@mark.parametrize('add_condition_slope', [True, False])
def test_estimate_posteriors_slope(add_condition_slope):
    window = BayesWindow(df, y='isi', treatment='stim', condition=['neuron', 'neuron_x_mouse'], group='mouse',
                         add_data=True)
    window.fit_slopes(models.model_hierarchical, add_condition_slope=add_condition_slope)

    chart = window.plot(x='neuron', column='neuron', row='mouse')
    chart.display()
    chart = window.plot(x='neuron', column='neuron', row='mouse')
    chart.display()


# This is not implemented
# def test_lme_two_conditions():
#     window = BayesWindow(df, y='isi', treatment='stim', condition=['neuron', 'neuron_x_mouse'], group='mouse', )
#     window.fit_lme()


def test_estimate_posteriors_slope_uneven_n_data_per_condition():
    # Trying to reproduce Uneven number of entries in conditions! Try setting do_take_mean=True
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=10,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    df = df.drop(df[(df['i_trial'] == 0) &
                    # (df['neuron_x_mouse'] == '0m0bayes') &
                    (df['stim'] == 0)].index)
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse', add_data=True)
    window.fit_slopes(models.model_hierarchical, do_make_change='divide')

    chart = window.plot(x='neuron', column='neuron', row='mouse')
    chart.display()
    chart = window.plot(x='neuron', column='neuron', row='mouse')
    chart.display()


@mark.parametrize('add_group_slope', [False, True])
def test_estimate_posteriors_slope_groupslope(add_group_slope):
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse', add_data=True)
    window.fit_slopes(models.model_hierarchical, add_group_slope=add_group_slope, add_group_intercept=True)

    chart = window.plot(x='neuron', column='neuron', row='mouse')
    chart.display()
    chart = window.plot(x='neuron', column='neuron', row='mouse')
    chart.display()


# def test_estimate_posteriors_two_conditions_w_wo_data():
#     df = generate_spikes_stim_types(mouse_response_slope=3,
#                                     n_trials=2,
#                                     n_neurons=3,
#                                     n_mice=4,
#                                     dur=2, )
#
#     window1 = BayesWindow(df, y='isi', treatment='stim', condition=['neuron', 'stim_strength'], group='mouse', )
#     window1.fit_slopes(model=models.model_hierarchical, do_mean_over_trials=False, add_data=False)
#
#     window2 = BayesWindow(df, y='isi', treatment='stim', condition=['neuron', 'stim_strength'], group='mouse', )
#     window2.fit_slopes(model=models.model_hierarchical, do_mean_over_trials=False, add_data=True)
#
#     assert (window1.data_and_posterior['neuron'] == window2.data_and_posterior['neuron']).all()
#     assert (window1.data_and_posterior['stim_strength'] == window2.data_and_posterior['stim_strength']).all()


def test_estimate_posteriors_two_conditions():
    df = generate_spikes_stim_types(mouse_response_slope=3,
                                    n_trials=2,
                                    n_neurons=3,
                                    n_mice=4,
                                    dur=2, )

    window = BayesWindow(df, y='isi', treatment='stim', condition=['neuron', 'stim_strength'], group='mouse',
                         add_data=True)
    window.fit_slopes(model=models.model_hierarchical, fold_change_index_cols=None, do_mean_over_trials=False)
    for condition_name in window.condition:
        assert condition_name in window.data_and_posterior.columns, f'{condition_name} not in window.condition'
    chart = window.plot(x='neuron', column='neuron', row='mouse')
    chart.display()


def test_two_groups():
    df = generate_spikes_stim_types(mouse_response_slope=3,
                                    n_trials=2,
                                    n_neurons=3,
                                    n_mice=4,
                                    dur=2, )

    window = BayesWindow(df, y='isi', treatment='stim', condition=['stim_strength', 'neuron_x_mouse'],
                         group='mouse', group2='neuron_x_mouse')
    window.fit_slopes(model=models.model_hierarchical, add_group2_slope=True)


def test_estimate_posteriors_data_overlay_slope():
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)
    chart = window.regression_charts(x='neuron', independent_axes=False)
    chart.display()
    window.chart.facet(row='mouse').display()


def test_estimate_posteriors_data_overlay_indep_axes_slope():
    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_hierarchical)
    chart = window.plot(independent_axes=True)
    chart.display()
    chart = window.facet(column='neuron', row='mouse')
    chart.display()


def test_plot_no_slope_data_only():
    window = BayesConditions(df=df, y='isi', treatment='stim')
    chart = window.plot()
    chart.display()


def test_plot_slope_data_only():
    window = BayesConditions(df=df, y='isi', treatment='stim')
    chart = window.plot()
    chart.display()


def test_fit_conditions():
    # TODO combined condition here somehow
    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit()


def test_fit_slopes():
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)


def test_plot_slopes():
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse', add_data=True)
    window.fit_slopes(model=models.model_hierarchical)
    window.plot()


def test_plot_slopes_2levelslope():
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse',
                         add_data=True)
    window.fit_slopes(model=models.model_hierarchical, add_group_slope=True)
    window.plot().display()


def test_plot_posteriors_no_slope():
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)
    window.regression_charts()


def test_plot_generic():
    # Slopes:
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)
    window.plot()
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)
    window.plot()
    # conditions:
    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_single)
    window.plot()


def test_facet():
    # Slopes:
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical)
    window.plot(row='neuron', width=40)
    window.plot(x='neuron').facet(column='mouse')

    # conditions:
    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_single)
    window.plot(row='neuron', width=40)
    window.plot(x='neuron').facet(column='mouse')


def test_single_condition_withdata():
    window = BayesWindow(dfl, y='Log power', treatment='stim', group='mouse')
    window.fit_slopes(model=models.model_hierarchical, do_make_change='divide', dist_y='normal')
    alt.layer(*plot_posterior(df=window.data_and_posterior, title=f'Log power', )).display()
    window.regression_charts(independent_axes=True).display()

    # Without data again
    window = BayesWindow(dfl, y='Log power', treatment='stim', group='mouse')
    window.fit_slopes(model=models.model_hierarchical, do_make_change='divide', dist_y='normal')
    alt.layer(*plot_posterior(df=window.data_and_posterior, title=f'Log power', )).display()
    window.regression_charts(independent_axes=True).display()

    # With data again
    window = BayesWindow(dfl, y='Log power', treatment='stim', group='mouse')
    window.fit_slopes(model=models.model_hierarchical, do_make_change='divide', dist_y='normal')
    alt.layer(*plot_posterior(df=window.data_and_posterior, title=f'Log power', )).display()
    window.regression_charts(independent_axes=True).display()


def test_single_condition_nodata():
    window = BayesWindow(dfl, y='Log power', treatment='stim', group='mouse')
    window.fit_slopes(model=models.model_hierarchical, do_make_change='divide', dist_y='normal')
    alt.layer(*plot_posterior(df=window.data_and_posterior, title=f'Log power', )).display()
    window.regression_charts(independent_axes=True).display()


def test_single_condition_nodata_dists():
    for dist in ['normal', 'lognormal', 'student']:
        window = BayesWindow(dfl, y='Log power', treatment='stim', group='mouse')
        window.fit_slopes(model=models.model_hierarchical, do_make_change='divide', dist_y=dist)
        alt.layer(*plot_posterior(df=window.data_and_posterior, title=f'Log power', )).display()
        window.regression_charts(independent_axes=True).display()


# TODO this does not work in GHA - takes too long
# @mark.parametrize('condition', [None, 'neuron'])
# @mark.parametrize('parallel', [False, True])
# @mark.parametrize('add_group_slope', [False, ])  # True doesnt work in GHA
# def test_explore_models(add_group_slope):
#     parallel = False
#     # Slopes:
#     conditions_to_test = [None]
#     if add_group_slope:
#         conditions_to_test.append('neuron')
#     for condition in conditions_to_test:
#         window = BayesWindow(df, y='isi', treatment='stim', condition=condition, group='mouse')
#         window.fit_slopes(model=models.model_hierarchical, num_chains=1)
#         window.explore_models(parallel=parallel, add_group_slope=add_group_slope)


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
                         group='Subject',
                         add_data=True)
    window.fit_slopes(model=models.model_hierarchical, fold_change_index_cols=[  # 'Condition code',
        'Brain region', 'Stim phase', 'stim_on', 'Fid', 'Subject', 'Inversion'], do_mean_over_trials=True, num_chains=1,
                      n_draws=100, num_warmup=100)
    window.regression_charts(x='Stim phase', color='Fid', independent_axes=True)


def test_conditions2():
    df.neuron = df.neuron.astype(int)
    window = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse', add_data=True)

    window.fit(model=models.model_single, num_chains=1)
    assert window.y in window.data_and_posterior
    window.plot(x='stim:O', independent_axes=False, add_data=True)


def random_tests():
    # TODO make a notebook for this
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit_slopes(model=models.model_hierarchical, num_chains=1)
    window.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)

    window.regression_charts(independent_axes=False, x='neuron:O', color='mouse')

    window.regression_charts(add_box=False, independent_axes=True, x='neuron:O', color='mouse')

    window.regression_charts(independent_axes=False, x='neuron:O', color='mouse')

    chart = window.regression_charts(independent_axes=True, x='neuron:O', color='mouse')

    chart.display()

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


# def test_group_slope(): #TODO
#     window = BayesWindow(df, y='isi', treatment='stim',
#                          condition='neuron_x_mouse',
#                          group='mouse',
#                          detail='i_trial', add_data=True
#                          )
#     window.fit_slopes(model=models.model_hierarchical, do_make_change='subtract',
#                       add_condition_slope=True,
#                       add_group_slope=True
#                       )


def test_data_replacement1():
    window = BayesWindow(df, y='isi', treatment='stim',
                         condition='neuron_x_mouse',
                         group='mouse',
                         detail='i_trial', add_data=True
                         )
    window.fit_slopes(model=models.model_hierarchical, do_make_change='subtract',
                      add_condition_slope=True,
                      # add_group_slope=True
                      )
    assert window.data_and_posterior.dropna(subset=['mu_intercept_per_group center interval'])[
               'mouse'].unique().size == 4


# def test_fit_twostep():
#     df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
#                                                                     n_neurons=2,
#                                                                     n_mice=2,
#                                                                     dur=3,
#                                                                     mouse_response_slope=16)

#     bw = BayesWindow(df_monster, y='isi', treatment='stim', condition=['neuron_x_mouse'], group='mouse',
#                      detail='i_trial')
#     bw = bw.fit_twostep(dist_y_step_one='normal', dist_y='normal', num_chains=1, n_draws=100, num_warmup=100)
#     bw.chart.display()


# def test_fit_twostep_by_group():
#     df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
#                                                                     n_neurons=2,
#                                                                     n_mice=2,
#                                                                     dur=3,
#                                                                     mouse_response_slope=16)

#     bw = BayesWindow(df_monster, y='isi', treatment='stim', condition=['neuron_x_mouse'], group='mouse',
#                      detail='i_trial')
#     bw = bw.fit_twostep_by_group(dist_y_step_one='gamma', parallel=False, dist_y='student')
#     bw.chart.display()


@mark.parametrize('do_make_change', [False, 'subtract'])
def test_plot_slopes_intercepts(do_make_change):
    window = BayesWindow(dfl, y='Power', treatment='stim', group='mouse', add_data=True)
    # Fit:
    window.fit_slopes(model=models.model_hierarchical, add_group_intercept=True,
                      add_group_slope=False, robust_slopes=False,
                      do_make_change=do_make_change, dist_y='gamma', num_chains=1,
                      n_draws=100, num_warmup=100);

    # %

    window.plot_slopes_intercepts(x='mouse').display()
    chart_intercepts = window.posterior_intercept
    chart_intercepts.display()


def test_gpu():
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse',
                         detail='i_trial')
    window.fit_slopes(model=models.model_hierarchical, do_make_change='subtract',
                      add_condition_slope=True,
                      add_group_slope=True,
                      use_gpu=True)
    assert window.data_and_posterior.dropna(subset=['mu_intercept_per_group center interval'])[
               'mouse'].unique().size == 4

# def test_stim_strength():
#     df = []
#     for slope in np.linspace(4, 400, 4):
#         df1 = generate_fake_lfp(mouse_response_slope=slope)[0]
#         df1['stim_strength'] = slope
#         df.append(df1)
#     df = pd.concat(df)

#     BayesWindow(df, 'Power', treatment='stim_strength', group='mouse', detail='i_trial').data_box_detail()
#     window = BayesWindow(df, 'Power', treatment='stim_strength', condition='mouse', detail='i_trial')
#     window.fit_slopes(add_condition_slope=True, center_intercept=True, dist_y='normal', num_chains=1, n_draws=100, num_warmup=100)
#     window.chart

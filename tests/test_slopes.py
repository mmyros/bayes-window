from pathlib import Path

import altair as alt
import pandas as pd

from bayes_window import BayesWindow
from bayes_window import models
from bayes_window.generative_models import generate_spikes_stim_types, generate_fake_spikes, generate_fake_lfp
from bayes_window.slopes import BayesRegression
from bayes_window.visualization import plot_posterior

from bayes_window.utils import load_radon

from pytest import mark
import os

os.environ['bayes_window_test_mode'] = 'True'

df_radon = load_radon()

dfl, _, _, _ = generate_fake_lfp(n_trials=5)

df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                n_neurons=3,
                                                                n_mice=4,
                                                                dur=4, )


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
    window = BayesRegression(df=df_radon, y='radon', treatment='floor', condition=['county'])
    # window.plot(x='county').facet(row='floor').display()
    window.fit(add_condition_slope=add_condition_slope,  # do_mean_over_trials=do_mean_over_trials,
               add_group_slope=add_group_slope, do_make_change=do_make_change,
               n_draws=100, num_chains=1, num_warmup=100)
    # window.plot().display()
    window.plot(x=':O',  # add_data=add_data,
                ).display()
    window.plot()
    window.chart_posterior_kde.display()
    # window.chart_data_boxplot.display() # Fold change won't work, bc uneven number of entries
    assert len(window.charts) > 3  # Should include kde


@mark.parametrize('transform_treatment', [False, True])
@mark.parametrize('do_make_change', [False, 'divide', 'subtract'])
@mark.parametrize('detail', [None, ':O', 'i_trial'])
def test_slopes(transform_treatment, do_make_change, detail):
    window = BayesRegression(df=df, y='isi', treatment='stim', condition='neuron', group='mouse', detail=detail,
                             transform_treatment=transform_treatment)
    # try:
    window.fit(model=models.model_hierarchical, do_make_change=do_make_change, )
    # fold_change_index_cols=('stim', 'mouse', 'neuron_x_mouse'))
    window.chart.display()
    window.window.chart_data_box_detail.display()


def test_plot():
    chart = BayesWindow(df=df, y='isi', treatment='stim').plot()
    chart.display()


@mark.parametrize('add_condition_slope', [True, False])
def test_estimate_posteriors_slope(add_condition_slope):
    window = BayesRegression(df=df, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse',
                             add_data=True)
    window.fit(models.model_hierarchical, add_condition_slope=add_condition_slope)

    chart = window.plot(x='neuron', column='neuron', row='mouse')
    chart.display()
    chart = window.plot(x='neuron', column='neuron', row='mouse')
    chart.display()


# This is not implemented
# def test_lme_two_conditions():
#     window = BayesWindow(df=df, y='isi', treatment='stim', condition=['neuron', '_mouse'], group='mouse', )
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
    window = BayesRegression(df=df, y='isi', treatment='stim', condition='neuron', group='mouse', add_data=True)
    window.fit(models.model_hierarchical, do_make_change='divide')

    chart = window.plot(x='neuron', column='neuron', row='mouse')
    chart.display()
    chart = window.plot(x='neuron', column='neuron', row='mouse')
    chart.display()


@mark.parametrize('add_group_slope', [False, True])
def test_estimate_posteriors_slope_groupslope(add_group_slope):
    window = BayesRegression(df=df, y='isi', treatment='stim', condition='neuron', group='mouse', add_data=True)
    window.fit(models.model_hierarchical, add_group_slope=add_group_slope, add_group_intercept=True)

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
#     window1 = BayesWindow(df=df, y='isi', treatment='stim', condition=['neuron', 'stim_strength'], group='mouse', )
#     window1.fit(model=models.model_hierarchical, do_mean_over_trials=False, add_data=False)
#
#     window2 = BayesWindow(df=df, y='isi', treatment='stim', condition=['neuron', 'stim_strength'], group='mouse', )
#     window2.fit(model=models.model_hierarchical, do_mean_over_trials=False, add_data=True)
#
#     assert (window1.data_and_posterior['neuron'] == window2.data_and_posterior['neuron']).all()
#     assert (window1.data_and_posterior['stim_strength'] == window2.data_and_posterior['stim_strength']).all()


def test_estimate_posteriors_two_conditions():
    df = generate_spikes_stim_types(mouse_response_slope=3,
                                    n_trials=2,
                                    n_neurons=3,
                                    n_mice=4,
                                    dur=2, )

    regression = BayesRegression(BayesWindow(df=df, y='isi', treatment='stim',
                                             condition=['neuron', 'stim_strength'], group='mouse',
                                             add_data=True))
    regression.fit(model=models.model_hierarchical, fold_change_index_cols=None, do_mean_over_trials=False)
    for condition_name in regression.window.condition:
        assert condition_name in regression.data_and_posterior.columns, f'{condition_name} not in window.condition'
    chart = regression.plot(x='neuron', column='neuron', row='mouse')
    chart.display()


def test_two_groups():
    df = generate_spikes_stim_types(mouse_response_slope=3,
                                    n_trials=2,
                                    n_neurons=3,
                                    n_mice=4,
                                    dur=2, )

    window = BayesRegression(df=df, y='isi', treatment='stim', condition=['stim_strength', 'neuron_x_mouse'],
                             group='mouse', group2='neuron')
    window.fit(model=models.model_hierarchical, add_group2_slope=True)


def test_inheritance():
    window = BayesWindow(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window = BayesRegression(window)
    window.fit(model=models.model_hierarchical)
    chart = window.plot(x='neuron', independent_axes=False)
    chart.display()


def test_estimate_posteriors_data_overlay_slope():
    window = BayesRegression(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_hierarchical)
    chart = window.plot(x='neuron', independent_axes=False)
    chart.display()
    window.chart.facet(row='mouse').display()


@mark.parametrize('add_data', [True, False])
@mark.parametrize('add_data_plot', [True, False])
@mark.parametrize('add_group_slope', [True, False])
def test_estimate_posteriors_data_overlay_indep_axes_slope(add_data, add_data_plot, add_group_slope):
    window = BayesRegression(df=df, add_data=add_data, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_hierarchical, add_group_slope=add_group_slope)
    chart = window.plot(independent_axes=True, add_data=add_data_plot)
    chart.display()
    if add_group_slope and add_data_plot:
        chart = window.facet(column='neuron', row='mouse')
    else:
        chart = window.facet(column='neuron')
    chart.display()


def test_fit():
    window = BayesRegression(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_hierarchical)


def test_plot_slopes():
    window = BayesRegression(df=df, y='isi', treatment='stim', condition='neuron', group='mouse', add_data=True)
    window.fit(model=models.model_hierarchical)
    window.plot()
    window.plot_BEST()


def test_plot_slopes_2levelslope():
    window = BayesRegression(df=df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse',
                             add_data=True)
    window.fit(model=models.model_hierarchical, add_group_slope=True)
    window.plot().display()


def test_plot_posteriors_no_slope():
    window = BayesRegression(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_hierarchical)
    window.plot()


def test_plot_generic():
    # Slopes:
    window = BayesRegression(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_hierarchical)
    window.plot()
    window = BayesRegression(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_hierarchical)
    window.plot()


def test_facet():
    # Slopes:
    window = BayesRegression(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_hierarchical)
    window.plot(row='neuron', width=40)
    window.plot(x='neuron').facet(column='mouse')


def test_single_condition_withdata():
    window = BayesRegression(df=dfl, y='Log power', treatment='stim', group='mouse')
    window.fit(model=models.model_hierarchical, do_make_change='divide', dist_y='normal')
    alt.layer(*plot_posterior(df=window.data_and_posterior, y_title='Log power', )).display()
    window.plot(independent_axes=True).display()

    # Without data again
    window = BayesRegression(df=dfl, y='Log power', treatment='stim', group='mouse')
    window.fit(model=models.model_hierarchical, do_make_change='divide', dist_y='normal')
    alt.layer(*plot_posterior(df=window.data_and_posterior, y_title='Log power', )).display()
    window.plot(independent_axes=True).display()

    # With data again
    window = BayesRegression(df=dfl, y='Log power', treatment='stim', group='mouse')
    window.fit(model=models.model_hierarchical, do_make_change='divide', dist_y='normal')
    alt.layer(*plot_posterior(df=window.data_and_posterior, y_title='Log power', )).display()
    window.plot(independent_axes=True).display()


def test_single_condition_nodata():
    window = BayesRegression(df=dfl, y='Log power', treatment='stim', group='mouse')
    window.fit(model=models.model_hierarchical, do_make_change='divide', dist_y='normal')
    alt.layer(*plot_posterior(df=window.data_and_posterior, y_title='Log power', )).display()
    window.plot(independent_axes=True).display()


@mark.parametrize('dist', ['normal', 'lognormal', 'student'])
def test_single_condition_nodata_dists(dist):
    window = BayesRegression(df=dfl, y='Log power', treatment='stim', group='mouse', add_data=True)
    window.fit(model=models.model_hierarchical, do_make_change='divide', dist_y=dist, zscore_y=False, )
    alt.layer(*plot_posterior(df=window.data_and_posterior, y_title='Log power', )).display()
    window.plot(independent_axes=True).display()


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
#         window = BayesRegression(df=df, y='isi', treatment='stim', condition=condition, group='mouse')
#         window.fit(model=models.model_hierarchical, num_chains=1)
#         window.explore_models(parallel=parallel, add_group_slope=add_group_slope)


@mark.parametrize('force_correct_fold_change_index_cols', [False, True])
def test_chirp_data(force_correct_fold_change_index_cols):
    dfdata = pd.read_csv(Path('tests') / 'test_data' / 'chirp_power.csv')
    window = BayesRegression(df=dfdata, y='Log power',
                             treatment='stim_on',
                             condition=['Stim phase', 'Inversion', 'Brain region'],
                             group='Subject')
    if force_correct_fold_change_index_cols:
        window.fit(model=models.model_hierarchical, fold_change_index_cols=['Brain region', 'Stim phase', 'stim_on',
                                                                            'Fid', 'Subject', 'Inversion'],
                   num_chains=1, n_draws=100, num_warmup=100)
    else:
        window.fit(model=models.model_hierarchical, num_chains=1, n_draws=100, num_warmup=100)
    window.plot(x='Stim phase', color='Fid', independent_axes=True).display()


import json


@mark.parametrize('plot_from_data_and_posterior', [True, False])
def test_chirp_data1(plot_from_data_and_posterior):
    dfdata = pd.read_csv(Path('tests') / 'test_data' / 'chirp_power.csv')
    window = BayesRegression(df=dfdata, y='Log power',
                             treatment='stim_on',
                             condition=['Stim phase', 'Inversion'],
                             group='Subject')
    window.fit(model=models.model_hierarchical,
               num_chains=1, n_draws=100, num_warmup=100)
    chart = window.plot(x='Stim phase',  # color='Fid',
                        add_data=plot_from_data_and_posterior,
                        independent_axes=True)
    chart.display()
    chart_data = pd.DataFrame.from_records(list(json.loads(chart.to_json())['datasets'].values())[0])
    assert (chart_data['Stim phase'].unique().astype(float) == dfdata['Stim phase'].unique()).all()
    # assert (chart_data['Inversion'].astype(bool).unique()==dfdata['Inversion'].unique()).all()
    # assert (window.data_and_posterior['higher interval'].dropna().size <
    #         window.data_and_posterior['higher interval'].size)


def test_chirp_data2():
    dfdata = pd.read_csv(Path('tests') / 'test_data' / 'chirp_power.csv')
    window = BayesRegression(df=dfdata, y='Log power',
                             treatment='stim_on',
                             # condition=['Condition code'],
                             group='Subject',
                             add_data=True)
    window.fit(model=models.model_hierarchical, fold_change_index_cols=[  # 'Condition code',
        'Brain region', 'Stim phase', 'stim_on', 'Fid', 'Subject', 'Inversion'], do_mean_over_trials=True, num_chains=1,
               n_draws=100, num_warmup=100)
    window.plot(x='Stim phase', color='Fid', add_data=True, independent_axes=True).display()


def random_tests():
    # TODO make a notebook for this
    window = BayesRegression(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
    window.fit(model=models.model_hierarchical, num_chains=1)
    window.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)

    window.plot(independent_axes=False, x='neuron:O', color='mouse')

    window.plot(add_box=False, independent_axes=True, x='neuron:O', color='mouse')

    window.plot(independent_axes=False, x='neuron:O', color='mouse')

    chart = window.plot(independent_axes=True, x='neuron:O', color='mouse')

    chart.display()

    chart.resolve_scale(y='independent')

    window.facet(column='neuron')

    window = BayesRegression(df=df, y='isi',
                             treatment='stim',
                             condition='neuron',
                             group='mouse')
    window.fit(model=models.model_hierarchical, num_chains=1)
    window.plot(x='neuron', color='i_trial')

    window.plot()  # x='Stim phase', color='Fid')#,independent_axes=True)
    window.facet(column='neuron', row='mouse')


# def test_group_slope(): #TODO
#     window = BayesRegression(df=df, y='isi', treatment='stim',
#                          condition='neuron_x_mouse',
#                          group='mouse',
#                          detail='i_trial', add_data=True
#                          )
#     window.fit(model=models.model_hierarchical, do_make_change='subtract',
#                       add_condition_slope=True,
#                       add_group_slope=True
#                       )


def test_data_replacement1():
    window = BayesRegression(df=df, y='isi', treatment='stim',
                             condition='neuron_x_mouse',
                             group='mouse',
                             detail='i_trial', add_data=True
                             )
    window.fit(model=models.model_hierarchical, do_make_change='subtract',
               add_condition_slope=True,
               # add_group_slope=True
               )
    posterior_no_nan = window.data_and_posterior.dropna(subset=['mu_intercept_per_group center interval'])
    assert posterior_no_nan['mouse'].unique().size == 4


# def test_fit_twostep():
#     df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
#                                                                     n_neurons=2,
#                                                                     n_mice=2,
#                                                                     dur=3,
#                                                                     mouse_response_slope=16)

#     bw = BayesRegression(df=df_monster, y='isi', treatment='stim', condition=['neuron_x_mouse'], group='mouse',
#                      detail='i_trial')
#     bw = bw.fit_twostep(dist_y_step_one='normal', dist_y='normal', num_chains=1, n_draws=100, num_warmup=100)
#     bw.chart.display()


# def test_fit_twostep_by_group():
#     df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
#                                                                     n_neurons=2,
#                                                                     n_mice=2,
#                                                                     dur=3,
#                                                                     mouse_response_slope=16)

#     bw = BayesRegression(df=df_monster, y='isi', treatment='stim', condition=['neuron_x_mouse'], group='mouse',
#                      detail='i_trial')
#     bw = bw.fit_twostep_by_group(dist_y_step_one='gamma', parallel=False, dist_y='student')
#     bw.chart.display()


@mark.parametrize('do_make_change', ['subtract', False, ])
def test_plot_slopes_intercepts(do_make_change):
    window = BayesRegression(df=dfl, y='Power', treatment='stim', group='mouse', add_data=True)
    # Fit:
    window.fit(model=models.model_hierarchical, add_group_intercept=True, zscore_y=False,
               add_group_slope=False, robust_slopes=False,
               do_make_change=do_make_change, dist_y='gamma', num_chains=1,
               n_draws=100, num_warmup=100)

    window.plot_intercepts(x='mouse').display()
    chart_intercepts = window.chart_posterior_intercept
    chart_intercepts.display()


def test_gpu():
    window = BayesRegression(df=df, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse',
                             detail='i_trial')
    window.fit(model=models.model_hierarchical, do_make_change='subtract',
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

#     BayesRegression(df=df, 'Power', treatment='stim_strength', group='mouse', detail='i_trial').data_box_detail()
#     window = BayesRegression(df=df, 'Power', treatment='stim_strength', condition='mouse', detail='i_trial')
#     window.fit(add_condition_slope=True, center_intercept=True, dist_y='normal', num_chains=1, n_draws=100, num_warmup=100)
#     window.chart

window = BayesRegression(df=df, y='isi', treatment='stim', group='mouse')
window.fit(model=models.model_hierarchical)


def test_extra_plots():
    window.plot_intercepts().display()
    window.plot_detail_minus_intercepts().display()

# def test_explore_models():
#     window.explore_models(parallel=False)
#     window.explore_model_kinds(parallel=False)

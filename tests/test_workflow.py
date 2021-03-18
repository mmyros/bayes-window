from pathlib import Path

from bayes_window import models
from bayes_window.generative_models import *
from bayes_window.visualization import plot_posterior
from bayes_window.workflow import BayesWindow

trans = LabelEncoder().fit_transform


def test_slopes_dont_make_change():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=5,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=7,
                                                                    mouse_response_slope=16)
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    try:
        bw.fit_slopes(add_data=False, model=models.model_hierarchical, do_make_change=False,
                      fold_change_index_cols=('stim', 'mouse', 'neuron'))
    except ValueError:
        pass


def test_fit_lme():
    df, df_monster, index_cols, _ = generate_fake_lfp(n_trials=25)
    bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    bw.fit_lme(add_data=False, )
    bw.plot_posteriors_slopes()
    bw.facet(row='mouse')


def test_fit_lme_w_condition():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=20,
                                                                    n_neurons=7,
                                                                    n_mice=6,
                                                                    dur=7,
                                                                    mouse_response_slope=12,
                                                                    overall_stim_response_strength=45)
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse', )
    assert bw.fit_lme().posterior is not None
    bw.plot_posteriors_slopes()
    bw.facet(column='neuron_x_mouse', width=300)


def test_fit_lme_w_data():
    df, df_monster, index_cols, _ = generate_fake_lfp(n_trials=25)

    bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    bw.fit_lme(add_data=False, do_make_change='divide')
    assert bw.posterior is not None
    bw.plot_posteriors_slopes()


def test_fit_lme_w_data_condition():
    df, df_monster, index_cols, _ = generate_fake_spikes(n_trials=25)

    bw = BayesWindow(df, y='isi', treatment='stim', group='mouse', condition='neuron')

    bw.fit_lme(add_data=True, do_make_change='divide')
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
    bw.fit_slopes(models.model_hierarchical, add_data=False)

    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse')
    chart.display()
    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_slope_strengths():
    df = generate_spikes_stim_types(mouse_response_slope=3,
                                    n_trials=2,
                                    n_neurons=3,
                                    n_mice=4,
                                    dur=2, )
    if 1:  # TODO this will be hairy
        bw = BayesWindow(df, y='isi', treatment='stim', condition=['neuron_code', 'stim_strength'], group='mouse', )
        bw.fit_slopes(model=models.model_hierarchical, do_mean_over_trials=False, fold_change_index_cols=None,
                      add_data=False)

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
    bw.fit_slopes(model=models.model_hierarchical, add_data=False)
    chart = bw.plot_posteriors_slopes(independent_axes=False)
    chart.display()
    bw.facet(column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_data_overlay_indep_axes_slope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_code', group='mouse')
    bw.fit_slopes(model=models.model_hierarchical, add_data=True)
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
    chart = BayesWindow(df, y='isi', treatment='stim').plot_posteriors_no_slope()
    chart.display()


def test_plot_slope_data_only():
    from bayes_window.workflow import BayesWindow

    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    chart = BayesWindow(df, y='isi', treatment='stim').plot_posteriors_no_slope()
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
    bw.fit_slopes(add_data=True, model=models.model_hierarchical, )


def test_plot_slopes():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(add_data=True, model=models.model_hierarchical, )
    bw.plot()


def test_plot_slopes_2levelslope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse')
    bw.fit_slopes(add_data=True, model=models.model_hierarchical, add_group_slope=True)
    bw.plot().display()


def test_plot_posteriors_no_slope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(add_data=True, model=models.model_hierarchical, )
    bw.plot_posteriors_slopes()


def test_plot_generic():
    # Slopes:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(add_data=True, model=models.model_hierarchical, )
    bw.plot()
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(add_data=False, model=models.model_hierarchical, )
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
    bw.fit_slopes(add_data=True, model=models.model_hierarchical, )
    bw.plot().facet('neuron', width=40)

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
    bw.fit_slopes(add_data=True, model=models.model_hier_stim_one_codition,
                  do_make_change='divide', dist_y='normal')
    plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
    bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()

    # Without data again
    bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    bw.fit_slopes(add_data=False, model=models.model_hier_stim_one_codition,
                  do_make_change='divide', dist_y='normal')
    plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
    bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()

    # With data again
    bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    bw.fit_slopes(add_data=True, model=models.model_hier_stim_one_codition,
                  do_make_change='divide', dist_y='normal')
    plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
    bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()


def test_single_condition_nodata():
    df, df_monster, index_cols, _ = generate_fake_lfp()
    bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
    bw.fit_slopes(add_data=False, model=models.model_hier_stim_one_codition,
                  do_make_change='divide', dist_y='normal')
    plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
    bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()


def test_single_condition_nodata_dists():
    df, df_monster, index_cols, _ = generate_fake_lfp()
    for dist in ['normal', 'lognormal', 'student']:
        bw = BayesWindow(df, y='Log power', treatment='stim', group='mouse')
        bw.fit_slopes(add_data=False, model=models.model_hier_stim_one_codition,
                      do_make_change='divide', dist_y=dist)
        plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
        bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()


def test_explore_models():
    # Slopes:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(add_data=True, model=models.model_hierarchical, )
    bw.explore_models(parallel=False)
    bw.explore_models(parallel=True)


def test_chirp_data():
    df = pd.read_csv(Path('tests') / 'test_data' / 'chirp_power.csv')
    window = BayesWindow(df, y='Log power',
                         treatment='stim_on',
                         condition='Condition code',
                         group='Subject')
    window.fit_slopes(model=models.model_hierarchical, do_mean_over_trials=True,
                      fold_change_index_cols=['Condition code',
                                              'Brain region', 'Stim phase', 'stim_on', 'Fid', 'Subject', 'Inversion'], )
    window.plot_posteriors_slopes(x='Stim phase', color='Fid', independent_axes=True)


def test_chirp_data1():
    df = pd.read_csv(Path('tests') / ('test_data') / 'chirp_power.csv')
    window = BayesWindow(df, y='Log power',
                         treatment='stim_on',
                         condition='Condition code',
                         group='Subject')
    window.fit_slopes(model=models.model_hierarchical, do_mean_over_trials=True,
                      fold_change_index_cols=[  # 'Condition code',
                          'Brain region', 'Stim phase', 'stim_on', 'Fid', 'Subject', 'Inversion'], )
    window.plot_posteriors_slopes(x='Stim phase', color='Fid', independent_axes=True)


def test_conditions2():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=5,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=7,
                                                                    mouse_response_slope=16)
    df.neuron = df.neuron.astype(int)
    window = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')

    window.fit_conditions(model=models.model_single, )
    assert window.y in window.data_and_posterior
    window.plot_posteriors_no_slope(x='stim:O', independent_axes=False, add_data=True);


def random_tests():
    # TODO make a notebook for this
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
    bw.fit_slopes(add_data=True, model=models.model_hierarchical, )
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
    window.fit_slopes(model=models.model_hierarchical,
                      # plot_index_cols=['Brain region', 'Stim phase', 'stim_on', 'Fid','Subject','Inversion'],
                      )
    c = window.plot_posteriors_slopes(x='neuron', color='i_trial')

    window.plot_posteriors_slopes()  # x='Stim phase', color='Fid')#,independent_axes=True)
    window.facet(column='neuron', row='mouse')

from bayes_window import models
from bayes_window.generative_models import generate_fake_spikes, generate_fake_lfp
from bayes_window.visualization import plot_posterior
from bayes_window.workflow import BayesWindow
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform


def test_estimate_posteriors():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse_code', 'neuron_code'), )
    bw.fit_conditions(model=models.model_single_lognormal)

    chart = bw.plot(x='stim:O', column='neuron', row='mouse', add_data=False)
    chart.display()
    chart = bw.plot(x='stim:O', column='neuron_code', row='mouse_code', add_data=False)
    chart.display()


def test_estimate_posteriors_data_overlay():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse_code', 'neuron_code'))
    bw.fit_conditions(model=models.model_single_lognormal)
    chart = bw.plot(x='stim:O', independent_axes=False, add_data=True,
                    column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_data_overlay_indep_axes():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse_code', 'neuron_code'))
    bw.fit_conditions(model=models.model_single_lognormal)
    chart = bw.plot(x='stim:O', independent_axes=True, add_data=True,
                    column='neuron_code', row='mouse_code')
    chart.display()


def test_plot():
    from bayes_window.workflow import BayesWindow

    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    chart = BayesWindow(df).plot(add_data=True)
    chart.display()


def test_estimate_posteriors_slope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse_code', 'neuron_code'), )
    bw.fit_slopes(models.model_hier_lognormal_stim)

    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse', add_data=False)
    chart.display()
    chart = bw.plot(x='neuron_code', column='neuron_code', row='mouse_code', add_data=False)
    chart.display()


def test_estimate_posteriors_data_overlay_slope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse_code', 'neuron_code'))
    bw.fit_slopes(model=models.model_hier_lognormal_stim)
    chart = bw.plot_posteriors_slopes(independent_axes=False, add_data=True, )
    chart.display()
    bw.facet(column='neuron_code', row='mouse_code')
    chart.display()


def test_estimate_posteriors_data_overlay_indep_axes_slope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse_code', 'neuron_code'))
    bw.fit_slopes(model=models.model_hier_lognormal_stim)
    chart = bw.plot(independent_axes=True, add_data=True, )
    chart.display()
    chart = bw.facet(column='neuron_code', row='mouse_code')
    chart.display()


def test_plot_no_slope_data_only():
    from bayes_window.workflow import BayesWindow

    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    chart = BayesWindow(df).plot_posteriors_no_slope(add_data=True)
    chart.display()


def test_plot_slope_data_only():
    from bayes_window.workflow import BayesWindow

    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    chart = BayesWindow(df).plot_posteriors_no_slope(add_data=True)
    chart.display()


def test_fit_conditions():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )

    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse', 'neuron'))
    bw.fit_conditions(add_data=True)


def test_fit_slopes():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse', 'neuron'))
    bw.fit_slopes(add_data=True, model=models.model_hier_normal_stim, )


def test_plot_slopes():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse', 'neuron'))
    bw.fit_slopes(add_data=True, model=models.model_hier_normal_stim, )
    bw.plot()


def test_plot_posteriors_no_slope():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse', 'neuron'))
    bw.fit_slopes(add_data=True, model=models.model_hier_normal_stim, )
    bw.plot_posteriors_slopes()


def test_plot_generic():
    # Slopes:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse', 'neuron'))
    bw.fit_slopes(add_data=True, model=models.model_hier_normal_stim, )
    bw.plot()
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse', 'neuron'))
    bw.fit_slopes(add_data=True, model=models.model_hier_lognormal_stim, )
    bw.plot()
    # conditions:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse', 'neuron'))
    bw.fit_conditions(model=models.model_single_lognormal)
    bw.plot()


def test_facet():
    # Slopes:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse', 'neuron'))
    bw.fit_slopes(add_data=True, model=models.model_hier_normal_stim, )
    bw.plot().facet('neuron', width=40)

    # conditions:
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    bw = BayesWindow(df, y='isi', levels=('stim', 'mouse', 'neuron'))
    bw.fit_conditions(model=models.model_single_lognormal)
    bw.plot().facet('neuron', width=40)


def test_single_condition_withdata():
    df, df_monster, index_cols, _ = generate_fake_lfp()
    bw = BayesWindow(df, y='Log power', levels=('stim', 'mouse'))
    bw.fit_slopes(add_data=True, model=models.model_hier_stim_one_codition,
                  do_make_change='divide', dist_y='normal')
    plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
    bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()

    # Without data again
    bw = BayesWindow(df, y='Log power', levels=('stim', 'mouse'))
    bw.fit_slopes(add_data=False, model=models.model_hier_stim_one_codition,
                  do_make_change='divide', dist_y='normal')
    plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
    bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()


    # With data again
    bw = BayesWindow(df, y='Log power', levels=('stim', 'mouse'))
    bw.fit_slopes(add_data=True, model=models.model_hier_stim_one_codition,
                  do_make_change='divide', dist_y='normal')
    plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
    bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()


def test_single_condition_nodata():
    df, df_monster, index_cols, _ = generate_fake_lfp()
    bw = BayesWindow(df, y='Log power', levels=('stim', 'mouse'))
    bw.fit_slopes(add_data=False, model=models.model_hier_stim_one_codition,
                  do_make_change='divide', dist_y='normal')
    plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
    bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()

def test_single_condition_nodata_dists():
    df, df_monster, index_cols, _ = generate_fake_lfp()
    for dist in ['normal', 'lognormal', 'student']:
        bw = BayesWindow(df, y='Log power', levels=('stim', 'mouse'))
        bw.fit_slopes(add_data=False, model=models.model_hier_stim_one_codition,
                      do_make_change='divide', dist_y=dist)
        plot_posterior(df=bw.data_and_posterior, title=f'Log power', ).display()
        bw.plot_posteriors_slopes(add_box=True, independent_axes=True).display()

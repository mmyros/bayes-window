import bulwark.checks as ck
import xarray as xr
from bayes_window import models
from bayes_window.fitting import fit_numpyro
from bayes_window.generative_models import generate_fake_spikes
from joblib import delayed, Parallel
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform


# def test_fit_numpyro_2step():
#     df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=20,
#                                                                     n_neurons=3,
#                                                                     n_mice=4,
#                                                                     dur=7, )
#
#     trace = fit_numpyro(y=df_monster['isi'].values,
#                         treatment=trans(df_monster['stim']),
#                         condition=trans(df_monster['neuron_x_mouse']),
#                         group=trans(df_monster['mouse']),
#                         progress_bar=True,
#                         model=models.twostep,
#                         add_group_slope=True,
#                         dist_y='gamma',
#                         n_draws=100, num_chains=1, )
#
# def test_fit_numpyro_2levelslope():
#     df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
#                                                                     n_neurons=3,
#                                                                     n_mice=4,
#                                                                     dur=2, )
#
#     trace = fit_numpyro(y=df['isi'].values,
#                         treatment=trans(df['stim']),
#                         condition=trans(df['neuron_x_mouse']),
#                         group=trans(df['mouse']),
#                         progress_bar=True,
#                         model=models.model_hierarchical,
#                         add_group_slope=True,
#                         dist_y='gamma',
#                         n_draws=100, num_chains=1, )


def test_fit_numpyro_gamma():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )

    trace, mcmc = fit_numpyro(y=df['isi'].values,
                        treatment=trans(df['stim']),
                        condition=trans(df['neuron']),
                        group=trans(df['mouse']),
                        progress_bar=True,
                        model=models.model_hierarchical,
                        dist_y='gamma',
                        n_draws=100, num_chains=1, )


def test_fit_numpyro_serial():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )

    for y in (set(df.columns) - set(index_cols)):
        trace, mcmc = fit_numpyro(y=df[y].values,
                            treatment=trans(df['stim']),
                            condition=trans(df['neuron']),
                            group=trans(df['mouse']),
                            progress_bar=True,
                            model=models.model_hierarchical,
                            n_draws=100, num_chains=1, )
        # chart = visualization.plot_posterior_altair(trace,
        #                                             df,
        #                                             b_name='slope_per_condition',
        #                                             x='stim',
        #                                             group_name='neuron'
        #                                             )
        #
        # assert ((type(chart) == FacetChart) |
        #         (type(chart) == Chart) |
        #         (type(chart) == LayerChart)), print(f'{type(chart)}')
        trace.posterior.to_dataframe().pipe(ck.has_no_nans)


def test_fit_numpyro_parallel():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes()
    # df = df.set_index(list(index_cols))
    meas = (set(df.columns) - set(index_cols))

    traces = Parallel(n_jobs=-1, verbose=2,
                      # backend='multiprocessing'
                      )(
        delayed(fit_numpyro)(y=y,
                             treatment=(df['stim']).astype(int).values,
                             condition=trans(df['neuron']),
                             group=trans(df['mouse']),
                             progress_bar=False,
                             model=models.model_hierarchical,
                             n_draws=10
                             )
        for y in [df[y].values for y in meas])
    assert type(traces[0][0].posterior) == xr.Dataset
    [trace[0].posterior.to_dataframe().pipe(ck.has_no_nans) for trace in traces]

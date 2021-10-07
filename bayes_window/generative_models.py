import warnings

import altair as alt
import numpy as np
import pandas as pd
import xarray as xr
from elephant.spike_train_generation import inhomogeneous_poisson_process as poisson
from neo import analogsignal
from quantities import s
from sklearn.preprocessing import LabelEncoder

from bayes_window import utils

trans = LabelEncoder().fit_transform
warnings.filterwarnings("ignore")

alt.data_transformers.disable_max_rows()


# try:
#     alt.renderers.enable('altair_saver', fmts=['png'])
# except Exception:
#     pass


# fake data generation
def generate_fake_lfp(n_trials=10,
                      n_mice=10,
                      dur=7,
                      mouse_response_slope=1.,
                      fastest_fr=38,
                      **kwargs):
    # neuron becomes mouse to provide different baselines
    df, df_monster, index_cols, _ = generate_fake_spikes(n_trials=n_trials,
                                                         n_neurons=n_mice,
                                                         n_mice=2,
                                                         dur=dur,
                                                         mouse_response_slope=mouse_response_slope,
                                                         fastest_fr=fastest_fr,
                                                         **kwargs)

    df = df[df.mouse_code == 1]
    df = df[df.neuron.astype(int) < n_mice - 2]
    df = df.drop(['mouse_code', 'mouse', 'neuron_code'], axis=1)
    df_monster = df_monster[df_monster.mouse_code == 1]
    df_monster = df_monster[df_monster.neuron.astype(int) < n_mice - 2]
    df_monster = df_monster.drop(['mouse_code', 'mouse', 'neuron_code'], axis=1)
    df = df.rename({'firing_rate': 'Power', 'log_firing_rate': 'Log power', 'neuron': 'mouse'}, axis=1)
    df_monster = df_monster.rename({'firing_rate': 'Power', 'log_firing_rate': 'Log power', 'neuron': 'mouse'}, axis=1)
    return df, df_monster, index_cols, None


def generate_spikes_stim_types(mouse_response_slope=3, **kwargs):
    df1 = generate_fake_spikes(mouse_response_slope=mouse_response_slope, **kwargs)[0]
    df1.insert(0, 'stim_strength', np.ones(df1.shape[0]))
    df2 = generate_fake_spikes(mouse_response_slope=2 * mouse_response_slope, **kwargs)[0]
    df2.insert(0, 'stim_strength', 2 * np.ones(df1.shape[0]))
    return pd.concat([df1, df2]).reset_index(drop=True)


def generate_spikes_stim_strength(overall_stim_response_strengths=range(10), **kwargs):
    df = []
    for overall_stim_response_strength in overall_stim_response_strengths:
        df1 = generate_fake_spikes(overall_stim_response_strength=overall_stim_response_strength, **kwargs)[0]
        df1.insert(0, 'stim_strength', df1['stim'] * overall_stim_response_strength)
        df.append(df1)
    return pd.concat(df).reset_index(drop=True)


def generate_fake_spikes(n_trials=6,
                         n_neurons=8,
                         n_mice=10,
                         dur=7,
                         slowest_fr=7,
                         time_randomness=10,
                         overall_stim_response_strength=0,
                         mouse_response_slope=2.,  # .3 is hard, 3 is easy
                         verbose=False,
                         do_ave_trial=False,
                         trial_baseline_randomness=.2,
                         do_bad_mice=False,
                         fastest_fr=15
                         ):
    """
    # mouse id affects slope of fr
    # Stim affects Firing rates for slow more than fast neurons

    """
    trans = LabelEncoder().fit_transform

    def firing_rate_over_time(rate, rate0=None, dur=10, time_randomness=None):
        time_rate = np.tile(rate, dur)
        time_rate[time_rate < 0] = 0.01  # check that no zeros
        #  add irrelevant blip of fr at end of spiketrain
        # Only 1/4 is relevant. Out of 7 sec, that's 7*0.25=1.75 sec
        if not (rate0 is None):
            time_rate[:int(dur / 4 * 2)] = rate0
            time_rate[-int(dur / 4):] = rate0
        return time_rate + time_randomness * np.random.random(dur)

    # Response strength varies by mouse:
    stim_response_mouse = overall_stim_response_strength + np.arange(n_mice) * mouse_response_slope
    if do_bad_mice:  # Half the mice have no response to stim at all
        stim_response_mouse[:int(n_mice / 2)] = 0
    if verbose:
        print(f'Stim response per mouse (in Hz) is {stim_response_mouse}')

    fast_rates = np.logspace(np.log10(slowest_fr + .1 + stim_response_mouse[range(n_mice)]),
                             np.log10(fastest_fr), n_neurons)
    slow_rates = np.logspace(np.tile(np.log10(slowest_fr), n_mice),
                             np.log10(fastest_fr), n_neurons)
    if verbose:
        print(f'Stim response per mouse (in Hz) is {fast_rates - slow_rates}')
        print(f'Fast rates = {fast_rates},\n slow rates= {slow_rates}')

    # make an Xarray of firing rates:
    slow = xr.DataArray(np.vstack(slow_rates).T,
                        dims=['Mouse', 'Neuron'],
                        coords=dict(Mouse=[f'm{i}bayes' for i in range(n_mice)],
                                    Neuron=range(n_neurons)), name='Firing rate')
    fast = xr.DataArray(np.vstack(fast_rates).T,
                        dims=['Mouse', 'Neuron'],
                        coords=dict(Mouse=[f'm{i}bayes' for i in range(n_mice)],
                                    Neuron=range(n_neurons)), name='Firing rate')
    slow['Stim'] = False
    fast['Stim'] = True
    firing_rates = xr.concat([slow, fast], dim='Stim')

    ddf = []
    for (imouse, i_neuron), firing_rate in firing_rates.to_dataframe().groupby(['Mouse', 'Neuron']):

        firing_rate.reset_index(inplace=True)
        fast_rate = firing_rate[firing_rate['Stim']]['Firing rate'].values
        slow_rate = firing_rate[~firing_rate['Stim']]['Firing rate'].values
        for i_trial in range(n_trials):
            # generate  independent Poisson spike trains that varies with time
            spiketrains_stim = np.array(poisson(
                analogsignal.AnalogSignal(
                    firing_rate_over_time(fast_rate,
                                          slow_rate + abs(np.random.randn() * trial_baseline_randomness),
                                          dur=dur,
                                          time_randomness=time_randomness),
                    1 / s, sampling_rate=1 / s)))
            spiketrains_nostim = np.array(poisson(
                analogsignal.AnalogSignal(
                    firing_rate_over_time(slow_rate, dur=dur, time_randomness=time_randomness),
                    1 / s, sampling_rate=1 / s)))
            for i_spike, spike in enumerate(spiketrains_stim):
                ddf.append(dict(spike_time=spike,
                                neuron=str(i_neuron),
                                stim=True,
                                mouse=firing_rate['Mouse'].values[0],
                                i_trial=i_trial,
                                i_spike=i_spike))

            for i_spike, spike in enumerate(spiketrains_nostim):
                ddf.append(dict(spike_time=spike,
                                neuron=str(i_neuron),
                                stim=False,
                                mouse=firing_rate['Mouse'].values[0],
                                i_trial=i_trial,
                                i_spike=i_spike))

    df = pd.DataFrame.from_records(ddf).sort_values(
        ['neuron', 'stim', 'mouse', 'i_trial', 'spike_time']).reset_index(drop=True)

    # Make index for pymc

    df['neuron_x_mouse'] = df['mouse'] + df['neuron']
    df['mouse_code'] = trans(df['mouse'])
    df['neuron_code'] = trans(df['neuron_x_mouse'])
    df['mouse_code'] = trans(df['mouse'])
    df['stim'] = trans(df['stim'])
    index_cols = set(df.columns) - {'spike_time', 'firing_rate', 'Firing rate', 'log_firing_rate', 'num_spikes', 'isi',
                                    'log_isi', '1/isi', 'log_1/isi'}

    # make full ISI
    df['isi'] = np.diff(df['spike_time'], prepend=0)
    df = df[df['isi'] > 0]  # drop the first spike of every trial, since impossible to estimate its isi
    df = df.drop(0)  # drop the first spike, since impossible to estimate its isi
    df['1/isi'] = df['isi'].apply(lambda x: 1 / x)
    df['log_1/isi'] = np.log10(df['1/isi'])
    df['log_isi'] = np.log10(df['isi'])

    if do_ave_trial:
        df = df.groupby(['neuron', 'neuron_x_mouse', 'mouse', 'stim']).mean().reset_index()

    # Trialwise measures of firing rate:
    dddf = []
    for i, ddf in df.groupby(list(index_cols - {'i_spike'})):
        ddf.insert(0, 'firing_rate', ddf['1/isi'].mean())
        ddf.insert(0, 'Firing rate', ddf['firing_rate'])
        ddf.insert(0, 'log_firing_rate', np.log10(ddf['firing_rate']))
        dddf.append(ddf)

    df = pd.concat(dddf)

    df_monster = df.copy()

    # make reduced dataframe
    # Mean over rounds, drop spike time and other stuff that doesnt make sense for single-trial data
    df = df.set_index(['i_trial']).groupby(list(index_cols - {'i_spike'})).mean().drop(
        ['log_isi', '1/isi', 'log_1/isi', 'spike_time'], axis=1).reset_index()

    if verbose:
        print(f'Difference in firing rates is {fast_rates[0] - slow_rates[0]}')
    return df, df_monster, index_cols, firing_rates


def fake_spikes_explore(df, df_monster, index_cols):
    # mean firing rate per trial per mouse
    width = 50
    fig_trials = alt.Chart(df).mark_line(fill=None, ).encode(
        x=alt.X('stim:O'),
        y=alt.Y('mean(log_firing_rate)', scale=alt.Scale(zero=False)),
        color='neuron:N',
        detail='i_trial:Q',
        opacity=alt.value(1),
        size=alt.value(.9),
        facet='mouse'
    ).properties(
        title='All trials and neurons',
        # columns=5,
        width=width,
        height=300
    )

    # mean firing rate per trial per mouse (select first and last mouse)
    alt.data_transformers.disable_max_rows()
    fig_select = alt.Chart(df[(df['neuron'] == '0') |
                              (df['neuron'] == str(df['neuron'].astype(int).max().astype(int) - 1))]).mark_line(
        fill=None, ).encode(
        x=alt.X('stim:O'),
        y=alt.Y('mean(log_firing_rate)', scale=alt.Scale(zero=False)),
        color='neuron:N',
        detail='i_trial:Q',
        opacity=alt.value(1),
        size=alt.value(2),
        facet='mouse'
    ).properties(
        title='Slow neurons are more responsive',
        # columns=5,
        width=width,
        height=300
    )

    # mean firing rate per mouse
    fig_neurons = alt.Chart(df).mark_line(fill=None, ).encode(
        x=alt.X('stim:O'),
        y=alt.Y('mean(log_firing_rate)', scale=alt.Scale(zero=False)),
        color='neuron:N',
        opacity=alt.value(1),
        size=alt.value(2),
        facet='mouse'
    ).properties(
        title='All neurons',
        # columns=5,
        width=width,
        height=300
    )
    # mean firing rate per mouse
    fig_mice = alt.Chart(df[(df['neuron'] == '0') |
                            (df['neuron'] == str(df['neuron'].astype(int).max()))]).mark_line(fill=None).encode(
        x=alt.X('stim:O'),
        y=alt.Y('mean(log_firing_rate)', scale=alt.Scale(zero=True), title='Log firing rate'),
        opacity=alt.value(1),
        size=alt.value(3),
        color='mouse:N',
        column=alt.Column('neuron', title=['Best (left) and worst (right)', 'neuron from each mouse'])
    ).properties(
        title=['Some mice are more responsive '],
        # columns=5,
        width=width,
        height=300
    )

    # Monster-level ISI
    df_isi = df_monster[
        (
            (df_monster['neuron'] == '0') |
            (df_monster['neuron'] == str(df_monster['neuron'].astype(int).max()))
        ) &
        # (df_monster['mouse']=='m0bayes') |
        (df_monster['mouse'] == f'm{df_monster["mouse_code"].astype(int).max()}bayes')
        ]
    fig_isi = alt.Chart(df_isi).mark_tick(opacity=.2).encode(
        x=alt.X('stim:O'),
        y=alt.Y('mean(log_1/isi)', scale=alt.Scale(zero=False), ),
        color='neuron:N',
        detail='i_spike:Q',  # Crucial: Q!
    ).properties(
        title=['Multiple trials per mouse', 'many spikes'],
        height=500
    )
    fig_overlay = alt.Chart(df_isi).mark_line(fill=None, ).encode(
        x=alt.X('stim:O'),
        y=alt.Y('log_firing_rate', scale=alt.Scale(zero=False)),
        color='neuron:N',
        detail='i_trial:Q',
        size=alt.value(2),
    )

    data_fold_change, y = utils.make_fold_change(df, y='log_firing_rate',
                                                 index_cols=list(set(index_cols) - {'i_spike'}),
                                                 treatment_name='stim',
                                                 treatments=(0, 1))
    box = alt.Chart(data=data_fold_change).mark_boxplot().encode(y=y).encode(
        x=alt.X('neuron:N', ),
        y=alt.Y(y, scale=alt.Scale(zero=True)),
    ).properties(width=width, height=240).facet(
        # row='mouse:N',
        column=alt.Column('mouse'))

    bar = (alt.Chart(data=data_fold_change).mark_bar().encode(y=alt.Y(y, aggregate='mean')) +
           alt.Chart(data=data_fold_change).mark_errorbar().encode(y=alt.Y(y, aggregate='stderr'))).encode(
        x=alt.X('neuron:N', ),
        y=alt.Y(y),
    ).properties(width=width * 2, height=240).facet(
        # row='Inversion:N',
        column=alt.Column('mouse'))

    bar_combined = (alt.Chart(data=data_fold_change).mark_bar().encode(y=alt.Y(y, aggregate='mean')) +
                    alt.Chart(data=data_fold_change).mark_errorbar().encode(y=alt.Y(y, aggregate='stderr'))).encode(
        x=alt.X('neuron:N', ),
        y=alt.Y(y),
    ).properties(width=width, height=240)

    # Monster-level ISI
    df_raster = df_monster[
        # (
        #    (df_monster['neuron']=='0')       |
        #    (df_monster['neuron']==str(n_neurons-1))
        # )
        # &
        # (df_monster['mouse']==f'm{df_monster["mouse_code"].astype(int).max()}bayes') |
        (df_monster['mouse'] == f'm{df_monster["mouse_code"].astype(int).max()}bayes')
    ]
    fig_raster = alt.Chart(df_raster).mark_tick(thickness=.8).encode(
        y=alt.Y('neuron'),
        x=alt.X('spike_time', scale=alt.Scale(zero=False), ),
        # color='neuron:N',
        detail='i_spike:Q',
    ).properties(
        # title=['Multiple trials per mouse','many spikes'],
        width=800,
        height=140
    ).facet(row='stim')

    return fig_mice, fig_select, fig_neurons, fig_trials, fig_isi + fig_overlay, bar, box, fig_raster, bar_combined

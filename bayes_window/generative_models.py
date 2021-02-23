import warnings

import numpy as np
import pandas as pd
import xarray as xr
from elephant.spike_train_generation import inhomogeneous_poisson_process as poisson
from neo import analogsignal
from quantities import s
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform
warnings.filterwarnings("ignore")


# fake data generation
def generate_fake_lfp(n_trials=10,
                      n_mice=10,
                      dur=7,
                      mouse_response_slope=1.):
    # neuron becomes mouse to provide different baselines
    df, df_monster, index_cols, _ = generate_fake_spikes(n_trials=n_trials,
                                                         n_neurons=n_mice,
                                                         n_mice=2,
                                                         dur=dur,

                                                         mouse_response_slope=mouse_response_slope)

    df = df[df.mouse_code == 1]
    df = df[df.neuron.astype(int) < n_mice - 2]
    df = df.drop('mouse', axis=1)
    # df_monster = df_monster.drop('mouse', axis=1)
    df = df.rename({'firing_rate': 'Power', 'log_firing_rate': 'Log power', 'neuron': 'mouse'}, axis=1)
    # df_monster = df_monster.rename({'firing_rate': 'Power', 'log_firing_rate': 'Log power', 'neuron': 'mouse'}, axis=1)
    return df, None, index_cols, None


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
                             np.log10(38), n_neurons)
    slow_rates = np.logspace(np.tile(np.log10(slowest_fr), n_mice),
                             np.log10(38), n_neurons)
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

    df['neuron_x_mouse'] = df['neuron'] + df['mouse']
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

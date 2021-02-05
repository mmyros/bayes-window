import bulwark.checks as ck
import numpy as np
from vis_and_bayes.fake_spikes import generate_fake_spikes


def test_generate_fake_spikes():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes()
    df = df.set_index(list(index_cols))
    assert abs(firing_rates.sel(Stim=1).isel(Neuron=0, Mouse=0) -
               firing_rates.sel(Stim=0).isel(Neuron=0, Mouse=0)
               ) < 1

    assert (firing_rates.sel(Stim=1).isel(Neuron=0, Mouse=-1) -
            firing_rates.sel(Stim=0).isel(Neuron=0, Mouse=-1)
            ) > 1

    assert abs(firing_rates.sel(Stim=1).isel(Neuron=-1, Mouse=-1) -
               firing_rates.sel(Stim=0).isel(Neuron=-1, Mouse=-1)
               ) < 1

    df.pipe(ck.has_no_nans)
    df_monster.pipe(ck.has_no_nans)

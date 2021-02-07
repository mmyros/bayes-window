from altair.vegalite.v4.api import FacetChart, Chart, LayerChart
from sklearn.preprocessing import LabelEncoder

from bayes_window.fake_spikes import generate_fake_spikes
from bayes_window.visualization import fake_spikes_explore

trans = LabelEncoder().fit_transform


def test_fake_spikes_explore():
    df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                    n_neurons=3,
                                                                    n_mice=4,
                                                                    dur=2, )
    charts = fake_spikes_explore(df, df_monster, index_cols)
    for chart in charts:
        assert ((type(chart) == FacetChart) |
                (type(chart) == Chart) |
                (type(chart) == LayerChart)), print(f'{type(chart)}')

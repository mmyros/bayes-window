# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Neurons example: high-level interface
# ## Generate some data

from bayes_window import models
from bayes_window.generative_models import generate_fake_spikes

# +

df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=10,
                                                                n_neurons=7,
                                                                n_mice=6,
                                                                dur=7,
                                                               mouse_response_slope=16)
# -

# ## Exploratory plot without any fitting

# +

from bayes_window.visualization import fake_spikes_explore
charts=fake_spikes_explore(df,df_monster,index_cols)
[chart.display() for chart in charts];
#fig_mice, fig_select, fig_neurons, fig_trials, fig_isi + fig_overlay, bar, box, fig_raster, bar_combined
# -

# ## Estimate with mouse>neuron

# +
from bayes_window import workflow
from bayes_window import visualization
from importlib import reload
reload(workflow)
reload(visualization)

bw=workflow.BayesWindow(df,y='isi', treatment='stim', condition='neuron', group='mouse')
bw.fit_slopes(add_data=True, model=models.model_hier_normal_stim,do_make_change='subtract',
             plot_index_cols=('stim', 'mouse', 'neuron'))

# +
chart=bw.plot_posteriors_slopes(add_box=True, independent_axes=True,x='neuron:O',color='mouse')

chart
# -

chart.resolve_scale(y='independent')

bw.plot(x='neuron',color='mouse',independent_axes=True)


bw.facet(column='neuron')

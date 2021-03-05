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

from bayes_window import models, fake_spikes_explore
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

bw = workflow.BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
bw.fit_slopes(add_data=True, model=models.model_hierarchical, do_make_change='subtract',
              fold_change_index_cols=('stim', 'mouse', 'neuron'))
# -

bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)


bw.plot_posteriors_slopes(add_box=True, independent_axes=False, x='neuron:O', color='mouse')

bw.plot_posteriors_slopes(add_box=False, independent_axes=True, x='neuron:O', color='mouse')

bw.plot_posteriors_slopes(independent_axes=False, x='neuron:O', color='mouse')

# +
chart = bw.plot_posteriors_slopes(add_box=True, independent_axes=True, x='neuron:O', color='mouse')

chart
# -

chart.resolve_scale(y='independent')

bw.facet(column='neuron')

# +
window = workflow.BayesWindow(df, y='isi',
                              treatment='stim',
                              condition='neuron',
                              group='mouse')
window.fit_slopes(model=models.model_hierarchical,
                  # plot_index_cols=['Brain region', 'Stim phase', 'stim_on', 'Fid','Subject','Inversion'],
                  )
c = window.plot_posteriors_slopes(x='neuron', color='i_trial')

window.plot_posteriors_slopes()  # x='Stim phase', color='Fid')#,independent_axes=True)
window.facet(column='neuron', row='mouse')
# -

window.plot_model_quality()

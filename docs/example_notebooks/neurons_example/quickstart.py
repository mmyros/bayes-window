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

# + [markdown] slideshow={"slide_type": "slide"} hideCode=false hidePrompt=false
# # Neurons example: high-level interface
# ## Generate some data

# + slideshow={"slide_type": "skip"} hideCode=false hidePrompt=false
from bayes_window import models, fake_spikes_explore, BayesWindow
from bayes_window.generative_models import generate_fake_spikes
import numpy as np

# + slideshow={"slide_type": "skip"} hideCode=false hidePrompt=false

df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=20,
                                                                n_neurons=6,
                                                                n_mice=3,
                                                                dur=5,
                                                               mouse_response_slope=40,
                                                               overall_stim_response_strength=45)

# + [markdown] slideshow={"slide_type": "slide"} hideCode=false hidePrompt=false
# ## Exploratory plot without any fitting

# + slideshow={"slide_type": "fragment"} hideCode=false hidePrompt=false

charts=fake_spikes_explore(df,df_monster,index_cols)
[chart.display() for chart in charts];
#fig_mice, fig_select, fig_neurons, fig_trials, fig_isi + fig_overlay, bar, box, fig_raster, bar_combined

# + [markdown] slideshow={"slide_type": "slide"} hideCode=false hidePrompt=false
# ## Estimate with neuron as condition
# -

# ### ISI

bw = BayesWindow(df, y='firing_rate', treatment='stim', condition='neuron_x_mouse', group='mouse',)
#bw.fit_anova()
try:
    bw.fit_lme()
    bw.plot_posteriors_slopes(x='neuron_x_mouse:O')
except np.linalg.LinAlgError as e:
    print(e)

# +
bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse')
bw.fit_slopes(add_data=True, model=models.model_hierarchical, do_make_change='subtract',
              progress_bar=False,
              dist_y='student',
              add_group_slope=True, add_group_intercept=False,
              fold_change_index_cols=('stim', 'mouse', 'neuron','neuron_x_mouse'))

bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)
bw.facet(column='mouse',width=200,height=200).display()

# + hidePrompt=true
import altair as alt 
slopes=bw.trace.posterior['b_stim_per_subject'].mean(['chain','draw']).to_dataframe().reset_index()
chart_slopes=alt.Chart(slopes).mark_bar().encode(
    x=alt.X('mouse:O',title='Mouse'),
    y=alt.Y('b_stim_per_subject', title='Slope')
)
chart_slopes
# -

# ### Firing rate

# +
bw = BayesWindow(df, y='firing_rate', treatment='stim', condition='neuron_x_mouse', group='mouse')
bw.fit_slopes(add_data=True, model=models.model_hierarchical, do_make_change='subtract',
              progress_bar=False,
              dist_y='student',
              add_group_slope=True, add_group_intercept=False,
              fold_change_index_cols=('stim', 'mouse', 'neuron','neuron_x_mouse'))

bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)
bw.facet(column='mouse',width=200,height=200).display()

# + [markdown] slideshow={"slide_type": "slide"} hideCode=false hidePrompt=false
# ANOVA may not be appropriate here: It considers every neuron. If we look hard enough, surely we'll find a responsive neuron or two out of hundreds?

# + slideshow={"slide_type": "fragment"} hideCode=false hidePrompt=false
bw.fit_anova()
# -

bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse')
bw.fit_slopes(add_data=True, model=models.model_hierarchical, do_make_change='subtract',
              progress_bar=False,
              dist_y='student',
              add_group_slope=True, add_group_intercept=False,
              fold_change_index_cols=('stim', 'mouse', 'neuron','neuron_x_mouse'))

# + slideshow={"slide_type": "slide"} hideCode=false hidePrompt=false

bw.plot_model_quality()
# -

# ### All data points

# +

for y in ['isi', 'firing_rate']:
    print(y)
    bw = BayesWindow(df_monster, y=y, treatment='stim', condition='neuron_x_mouse', group='mouse')
    bw.fit_slopes(add_data=True, model=models.model_hierarchical, do_make_change='subtract',
                  progress_bar=True,
                  dist_y='student',
                  add_group_slope=True, add_group_intercept=False,
                  fold_change_index_cols=('stim', 'mouse', 'neuron','neuron_x_mouse'))

    bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)
    bw.facet(column='mouse',width=200,height=200).display()

    bw.explore_models()

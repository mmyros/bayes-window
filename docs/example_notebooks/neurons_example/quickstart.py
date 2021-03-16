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

# + slideshow={"slide_type": "skip"} hideCode=false hidePrompt=false

df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=10,
                                                                n_neurons=7,
                                                                n_mice=6,
                                                                dur=7,
                                                               mouse_response_slope=7)

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

# +
bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
bw.fit_slopes(add_data=True, model=models.model_hierarchical_gamma, do_make_change='subtract',
              fold_change_index_cols=('stim', 'mouse', 'neuron'))

bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)
# -

# ### Firing rate

# + slideshow={"slide_type": "fragment"} hideCode=false hidePrompt=false
bw = BayesWindow(df, y='firing_rate', treatment='stim', condition='neuron', group='mouse')
bw.fit_slopes(add_data=True, model=models.model_hierarchical, do_make_change='subtract',
              fold_change_index_cols=('stim', 'mouse', 'neuron'),
             dist_y='lognormal')

bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)

# + [markdown] slideshow={"slide_type": "fragment"} hideCode=false hidePrompt=false
# So firing rate is estimated as expected. Neurons 6 has a higher stim effect than neuron 0 in all mice. However, if we misspecify the prior, it won't work:

# + slideshow={"slide_type": "fragment"} hideCode=false hidePrompt=false
bw = BayesWindow(df, y='firing_rate', treatment='stim', condition='neuron', group='mouse')
bw.fit_slopes(add_data=True, model=models.model_hierarchical, do_make_change='subtract',
              fold_change_index_cols=('stim', 'mouse', 'neuron'),
             dist_y='normal')

bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)

# + [markdown] slideshow={"slide_type": "slide"} hideCode=false hidePrompt=false
# ISI gives a strange result. ISI needs to be gamma or exponential

# + slideshow={"slide_type": "fragment"} hideCode=false hidePrompt=false
bw = BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
bw.fit_slopes(add_data=True, model=models.model_hierarchical, do_make_change='subtract',
              fold_change_index_cols=('stim', 'mouse', 'neuron'),
              dist_y='lognormal')

bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)

# + slideshow={"slide_type": "skip"} hideCode=false hidePrompt=false
import arviz as az
az.plot_forest(bw.trace.posterior,'ridgeplot',rope=[-.1,.1]);

# + [markdown] slideshow={"slide_type": "slide"} hideCode=false hidePrompt=false
# ANOVA may not be appropriate here: It considers every neuron. If we look hard enough, surely we'll find a responsive neuron or two out of hundreds?

# + slideshow={"slide_type": "fragment"} hideCode=false hidePrompt=false
bw.fit_anova()

# + [markdown] slideshow={"slide_type": "slide"} hideCode=false hidePrompt=false
# LME needs to somehow estimate for each neuron. I don't know how to estimate slope for each level of neuron...

# + slideshow={"slide_type": "fragment"} hideCode=false hidePrompt=false
bw.fit_lme().posterior

# + slideshow={"slide_type": "slide"} hideCode=false hidePrompt=false
window.plot_model_quality()

# + hideCode=false hidePrompt=false slideshow={"slide_type": "skip"}
# Monster level with firing rate
# NBVAL_SKIP
bw = BayesWindow(df_monster, y='firing_rate', treatment='stim', condition='neuron', group='mouse')
bw.fit_slopes(add_data=True, model=models.model_hierarchical, do_make_change='subtract',
              dist_y='lognormal',
              fold_change_index_cols=('stim', 'mouse', 'neuron'),progress_bar=True)

bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)

# + slideshow={"slide_type": "slide"} hideCode=false hidePrompt=false
# Monster level ISI
# NBVAL_SKIP
bw = BayesWindow(df_monster, y='isi', treatment='stim', condition='neuron', group='mouse')
bw.fit_slopes(add_data=True, model=models.model_hierarchical, do_make_change='subtract',
              dist_y='lognormal',
              fold_change_index_cols=('stim', 'mouse', 'neuron'),progress_bar=True)

bw.plot(x='neuron', color='mouse', independent_axes=False, finalize=True)

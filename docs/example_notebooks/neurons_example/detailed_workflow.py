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
#     display_name: PyCharm (jup)
#     language: python
#     name: pycharm-d5912792
# ---

# # Neurons example via low-level, flexible interface
# ## Prepare

# +
from bayes_window import models
from bayes_window.fitting import fit_numpyro
from bayes_window.generative_models import generate_fake_spikes
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform


# -

# ## Make some data
#

df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                n_neurons=8,
                                                                n_mice=4,
                                                                dur=7, )

import numpy as np
df['log_isi']=np.log10(df['isi'])

# +
from bayes_window import visualization,utils
from importlib import reload
reload(visualization)
reload(utils)
y='isi'
df['neuron']=df['neuron'].astype(int)
ddf, dy = utils.make_fold_change(df,
                         y=y,
                         index_cols=('stim', 'mouse_code', 'neuron'),
                         condition_name='stim',
                         do_take_mean=True)

visualization.plot_data(x='neuron',y=dy, color='mouse_code',add_box=True,df=ddf)
# -

# ## Estimate model

#y = list(set(df.columns) - set(index_cols))[0]
trace = fit_numpyro(y=df[y].values,
                    stim_on=(df['stim']).astype(int).values,
                    treat=trans(df['neuron']),
                    subject=trans(df['mouse']),
                    progress_bar=True,
                    model=models.model_hier_normal_stim,
                    n_draws=100, num_chains=1, )

# ## Add data back

reload(utils)
df_both = utils.add_data_to_posterior(df,
                                trace=trace,
                                y=y,
                                index_cols=['neuron', 'stim', 'mouse_code', ],
                                condition_name='stim',
                                b_name='b_stim_per_condition',  # for posterior
                                group_name='neuron'  # for posterior
                                )

# ## Plot data and posterior

# +
#BayesWindow.plot_posteriors_slopes(df_both, y=f'{y} diff', x='neuron',color='mouse_code',title=y,hold_for_facet=False,add_box=False)


chart_d = visualization.plot_data(df=df_both,x='neuron', y=f'{y} diff',color='mouse_code')
chart_d
# -

chart_p = visualization.plot_posterior(df=df_both, title=f'd_{y}', x='neuron',)
chart_p

(chart_d+chart_p).resolve_scale(y='independent')

(chart_d+chart_p).facet(column='neuron')

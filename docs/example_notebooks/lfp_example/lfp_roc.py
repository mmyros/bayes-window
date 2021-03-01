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

# + [markdown] slideshow={"slide_type": "slide"}
# # Compare to ANOVA and LMM: ROC curve

# + slideshow={"slide_type": "skip"}
from bayes_window import model_comparison

alt.data_transformers.disable_max_rows()

import numpy as np
from importlib import reload

reload(model_comparison)

# + slideshow={"slide_type": "skip"}
res = model_comparison.run_conditions(true_slopes=np.hstack([np.zeros(10), np.tile(np.linspace(.2, 20, 40), 10)]),
                                      n_trials=range(10, 90, 70),
                                      trial_baseline_randomness=(.2, 10.8),
                                      parallel=True)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Binary
# -

reload(model_comparison)
df = model_comparison.make_roc_auc_old(
    res, binary=True, groups=('method', 'y', 'randomness', 'n_trials'))

bars, roc = model_comparison.plot_roc(df[df['y'] == 'Log power'])
bars.facet(column='n_trials', row='randomness').display()

roc.facet(column='n_trials', row='randomness').display()

# + slideshow={"slide_type": "fragment"}
reload(model_comparison)
bars, roc = model_comparison.plot_roc(df)
bars.facet(column='n_trials', row='y').display()
bars.facet(column='randomness', row='y').display()

# + slideshow={"slide_type": "fragment"}
roc.properties(width=150).facet(column='n_trials', row='y').display()

# + slideshow={"slide_type": "fragment"}
roc.properties(width=150).facet(column='randomness', row='y').display()

# + [markdown] slideshow={"slide_type": "slide"}
# ## Non-Binary
# For models that have CI

# + slideshow={"slide_type": "fragment"}
reload(model_comparison)
dfnb = model_comparison.make_roc_auc(
    res, binary=False, groups=('method', 'y', 'randomness', 'n_trials'))
bars, roc = model_comparison.plot_roc(dfnb)
bars.facet(column='randomness', row='y')
# -

roc.facet(column='randomness', row='y')

roc.facet(column='n_trials', row='y')

# ## CM

# +
# def plot_roc(res, binary=True, groups=('method', 'y', 'randomness', 'n_trials')):
# Make ROC and AUC
reload(model_comparison)

model_comparison.plot_confusion(
    model_comparison.make_confusion_matrix(res, ('method', 'y', 'randomness', 'n_trials')
                                           )).facet(column='method', row='y')

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

# + slideshow={"slide_type": "slide"}
from importlib import reload

import numpy as np

# + slideshow={"slide_type": "skip"}
from bayes_window import model_comparison, models
from bayes_window.generative_models import generate_fake_lfp

reload(model_comparison)

res = model_comparison.run_conditions(true_slopes=np.hstack([np.zeros(15), 
                                                             np.tile(15, 15)]),
#                                                              np.tile(np.linspace(20, 40, 3), 15)]),
                                      n_trials=np.linspace(15, 70, 5).astype(int),
#                                       trial_baseline_randomness=np.linspace(.2, 11, 3),
                                      ys=('Power', 'Log power'),
                                      parallel=True)
# -

# ## Confusion matrix

model_comparison.plot_confusion(
    model_comparison.make_confusion_matrix(res[res['y']=='Power'], ('method', 'y', 'randomness', 'n_trials')
                                           )).properties(width=140).facet(row='method', column='n_trials')

model_comparison.plot_confusion(
    model_comparison.make_confusion_matrix(res[res['y']=='Log power'], ('method', 'y', 'randomness', 'n_trials')
                                           )).properties(width=140).facet(row='method', column='n_trials')

# + [markdown] slideshow={"slide_type": "slide"}
# ## ROC curve

# +
reload(model_comparison)
df = model_comparison.make_roc_auc(res, binary=False, groups=('method', 'y', 'n_trials'))

bars, roc = model_comparison.plot_roc(df)
bars.facet(column='n_trials', row='y').properties().display()
roc.facet(column='n_trials', row='y').properties()

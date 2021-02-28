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
import numpy as np
from importlib import reload
reload(model_comparison)

# + slideshow={"slide_type": "skip"}
res = model_comparison.run_conditions(true_slopes=np.hstack([np.zeros(50), np.linspace(.03, 18, 80)]),
                                      n_trials=range(5, 90, 10),
                                      trial_baseline_randomness=(.2, 5, 10.8),

                                      parallel=True)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Binary

# + slideshow={"slide_type": "fragment"}
reload(model_comparison)
bars, roc = model_comparison.plot_roc(res, binary=True)
bars.facet(column='n_trials', row='y').display()
bars.facet(column='randomness', row='y').display()

# + slideshow={"slide_type": "fragment"}
roc.properties(width=150).facet(column='n_trials', row='y').display()

# + [markdown] slideshow={"slide_type": "slide"}
# ## Non-Binary
# For models that have CI

# + slideshow={"slide_type": "fragment"}
reload(model_comparison)
bars, roc = model_comparison.plot_roc(res, binary=False)
bars.facet(column='randomness', row='y')
# -

roc.facet(column='randomness', row='y')

# ## CM

# +
# def plot_roc(res, binary=True, groups=('method', 'y', 'randomness', 'n_trials')):
# Make ROC and AUC
reload(model_comparison)

model_comparison.plot_confusion(
    model_comparison.make_confusion_matrix(res, ('method', 'y', 'randomness', 'n_trials')
                                           )).facet(column='method', row='y')

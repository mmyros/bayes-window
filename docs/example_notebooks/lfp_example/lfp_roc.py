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
y_scores,true_slopes=model_comparison.run_methods(np.hstack([np.zeros(180), np.linspace(.03, 18, 140)]))

# + [markdown] slideshow={"slide_type": "slide"}
# ## Binary

# + slideshow={"slide_type": "fragment"}
model_comparison.plot_roc(y_scores, true_slopes)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Non-Binary
# For models that have CI

# + slideshow={"slide_type": "fragment"}
model_comparison.plot_roc(y_scores, true_slopes,binary=False)

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
y_scores, true_slopes = model_comparison.run_methods(np.hstack([np.zeros(160), np.linspace(.03, 18, 140)]),
                                                     parallel=True)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Binary

# + slideshow={"slide_type": "fragment"}
reload(model_comparison)
model_comparison.plot_roc(y_scores, true_slopes)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Non-Binary
# For models that have CI

# + slideshow={"slide_type": "fragment"}
model_comparison.plot_roc(y_scores, true_slopes, binary=False)
# -

# ## CM

# +
from itertools import product
import altair as alt
from sklearn.metrics import confusion_matrix
import pandas as pd

columns = ["actual", "predicted", "Occurences", "Method", 'y']

df = pd.DataFrame(columns=columns)

y_true = true_slopes > 0
for col in y_scores.keys():
    y_pred = np.array(y_scores[col]) > 0
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = [y for i in cm for y in i]
    roll = list(product(np.unique(y_true), repeat=2))
    for i in range(len(roll)):
        df = df.append({'actual': roll[i][0],
                        'predicted': roll[i][1],
                        'Occurences': cm[i],
                        'Method': col.split(',')[0],
                        'y': col.split(',')[1],
                        }, ignore_index=True)


# plot figure
def make_example():
    return alt.Chart(df).mark_rect().encode(
        x="predicted",
        y="actual",
        color='Occurences:O'
    ).properties(width=180, height=180)


make_example().facet(column='Method', row='y')

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
from bayes_window import model_comparison, models
from bayes_window.generative_models import generate_fake_lfp

import numpy as np
from importlib import reload
reload(model_comparison)

# + slideshow={"slide_type": "skip"}
res = model_comparison.run_conditions(true_slopes=np.hstack([np.zeros(10), np.tile(np.linspace(.2, 20, 10), 10)]),
                                      n_trials=np.linspace(10, 70, 5).astype(int),
                                      trial_baseline_randomness=np.linspace(.2, 11, 15),
                                      parallel=True)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Binary
# -

reload(model_comparison)
df = model_comparison.make_roc_auc(
    res, binary=True, groups=('method', 'y', 'randomness', 'n_trials'))

# TODO AUC is funky for anova

bars, roc = model_comparison.plot_roc(df)
bars.facet(column='n_trials', row='y').properties().display()
roc.facet(column='n_trials', row='y').properties()

bars, roc = model_comparison.plot_roc(df)
bars.facet(column='y').properties().display()
roc.facet(column='y').properties()

# ## CM

# +
# def plot_roc(res, binary=True, groups=('method', 'y', 'randomness', 'n_trials')):
# Make ROC and AUC
reload(model_comparison)

model_comparison.plot_confusion(
    model_comparison.make_confusion_matrix(res, ('method', 'y', 'randomness', 'n_trials')
                                           )).facet(column='method', row='y')
# -

# ## Model comparison

reload(model_comparison)
df, df_monster, index_cols, _ = generate_fake_lfp(mouse_response_slope=13,
                                                  n_trials=40)
model_comparison.compare_models(df=df,
                                models={
                                    'no_teratment': models.model_hierarchical,
                                    'no_group': models.model_hierarchical,
                                    'full_normal': models.model_hierarchical,
                                    'full_student': models.model_hierarchical,
                                    'full_lognogmal': models.model_hierarchical,

                                },
                                extra_model_args=[
                                    {'treatment': None},
                                    {'group': None},
                                    {'treatment': 'stim'},
                                    {'treatment': 'stim', 'dist_y': 'student'},
                                    {'treatment': 'stim', 'dist_y': 'lognormal'},
                                ],
                                y='isi',
                                condition=None,
                                parallel=False
                                );

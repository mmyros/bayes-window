.. image:: https://img.shields.io/pypi/v/bayes_window.svg
        :target: https://pypi.python.org/pypi/bayes_window

.. image:: https://github.com/mmyros/bayes-window/actions/workflows/pytest.yaml/badge.svg
        :target: https://github.com/mmyros/bayes-window/actions/workflows/pytest.yaml/badge.svg

.. image:: https://readthedocs.org/projects/bayes-window/badge/?version=latest
        :target: https://bayes-window.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/mmyros/bayes-window/branch/master/graph/badge.svg?token=CQMHJRNC9I
      :target: https://codecov.io/gh/mmyros/bayes-window

.. image:: https://img.shields.io/lgtm/grade/python/g/mmyros/bayes-window.svg?logo=lgtm&logoWidth=18
      :target: https://lgtm.com/projects/g/mmyros/bayes-window/context:python
      :alt: Language grade: Python

============
Bayes Window
============

.. image:: bw_logo_drawing.png
   :height: 160
   :align: center


Pretty and easy hierarchical Bayesian or linear mixed effects estimation with data overlay


* Free software: MIT license
* Documentation: https://mmyros.github.io/bayes-book/index.html


INSTALLATION
------------
For gpu:
```pip install --upgrade "jax[cuda102]" -f https://storage.googleapis.com/jax-releases/jax_releases.html```
or cudaXXX from `nvcc --version`


Features
--------

See extensive examples https://mmyros.github.io/bayes-book/index.html



TODO
----
- Fit checks for LME under plot_diagnostics()
- Posterior predictive to mimic data plot
- Less haphazard testing
- Shrinkage layer: Should be as easy as alt.layer with unpooled model
- Prior predictive and posterior predictive plots
- wait for resolution to https://github.com/vega/vega-lite/issues/4373#issuecomment-447726094
- Random-effect plots (eg intercepts): bw.plot_extras?
- Decide on Vega theme

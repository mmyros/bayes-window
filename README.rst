============
Bayes Window
============
.. image:: bw_logo.jpg
   :width: 70
   :height: 40
   :align: center

=================================


.. image:: https://img.shields.io/pypi/v/bayes_window.svg
        :target: https://pypi.python.org/pypi/bayes_window

.. image:: https://github.com/mmyros/bayes-window/actions/workflows/pytest.yaml/badge.svg
        :target: https://github.com/mmyros/bayes-window/actions/workflows/pytest.yaml/badge.svg

.. image:: https://readthedocs.org/projects/bayes-window/badge/?version=latest
        :target: https://bayes-window.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/mmyros/bayes-window/branch/master/graph/badge.svg?token=CQMHJRNC9I
      :target: https://codecov.io/gh/mmyros/bayes-window


Pretty and easy hierarchical Bayesian or linear mixed effects estimation with data overlay


* Free software: MIT license
* Documentation: https://bayes-window.readthedocs.io.

TODO
----
- Less haphazard testing
- Shrinkage layer: Should be as easy as alt.layer with unpooled model
   - Make sure + works well with facet
- how is holoviz with coposite overlays? Maybe the facet method is not necessary then
   - related to https://github.com/vega/vega-lite/issues/4373#issuecomment-447726094
- delete facet attribute of chart if independent_axes=True
- Formalize detailed data plot: trial-slope plus box. Try to add posteriors? see notebook
- Random-effect plots (eg intercepts): bw.plot_extras?
   - see lfp notebook
- Recode multiple conditions for LME
- Decide on Vega theme


Features
--------

* TODO

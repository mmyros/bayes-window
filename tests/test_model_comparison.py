from bayes_window.model_comparison import *


def test_run_methods():
    y_scores, true_slopes = run_methods(
        true_slopes=np.hstack([np.zeros(2), np.linspace(.03, 18, 2)]))
    plot_roc(y_scores, true_slopes).display()

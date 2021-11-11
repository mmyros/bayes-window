"""Top-level package ."""

__author__ = """Maxym Myroshnychenko"""
__email__ = 'mmyros@gmail.com'
__version__ = '0.1.0'

from .workflow import BayesWindow
from .conditions import BayesConditions
from .fitting import fit_numpyro, fit_svi
from .generative_models import (generate_fake_spikes, generate_spikes_stim_types, generate_fake_lfp,
                                generate_spikes_stim_strength, fake_spikes_explore)
# from .model_comparison import (compare_models, make_roc_auc, make_confusion_matrix, confusion_matrix, plot_confusion)
# from .utils import *
# from .visualization import *

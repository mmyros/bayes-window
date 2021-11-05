"""Top-level package ."""

__author__ = """Maxym Myroshnychenko"""
__email__ = 'mmyros@gmail.com'
__version__ = '0.1.0'

from .workflow import BayesWindow
from .generative_models import (generate_fake_lfp, generate_fake_spikes, generate_spikes_stim_types,
                                generate_spikes_stim_strength, fake_spikes_explore
                                )
from .fitting import fit_numpyro, fit_svi
# from .model_comparison import *
# from .utils import *
from .visualization import plot_posterior, facet
from .conditions import BayesConditions
from .lme import LMERegression
from .slopes import BayesRegression

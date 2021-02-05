import warnings
from importlib import reload
from pdb import set_trace
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xarray as xr
from joblib import Parallel, delayed

# import models  # My module contains statistical and poisson_firin_rate


import numpyro
from jax import random
import jax.numpy as jnp
import numpyro.distributions as dist

from numpyro.infer import MCMC, NUTS, Predictive, log_likelihood

import arviz as az


class Models():
    @staticmethod
    def model_single_normal_stim(fr, stim_on, treat):
        n_conditions = np.unique(treat).shape[0]
        a = numpyro.sample('a', dist.Normal(0, 1))
        b = numpyro.sample('b', dist.Normal(0, 1))
        sigma_b = numpyro.sample('sigma_b', dist.HalfNormal(1))
        b_stim_per_condition = numpyro.sample('b_stim', dist.Normal(jnp.tile(b, n_conditions), sigma_b))
        theta = a + b_stim_per_condition[treat] * stim_on

        sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
        numpyro.sample('y', dist.Normal(theta, sigma_obs), obs=fr)

    @staticmethod
    def model_single_exponential(fr, treat):
        n_conditions = np.unique(treat).shape[0]
        a_neuron = numpyro.sample('FR_neuron', dist.Gamma(4, 1))
        sigma_neuron = numpyro.sample('sigma_neuron', dist.Gamma(4, 1))
        a_neuron_per_condition = numpyro.sample('FR_neuron_per_condition',
                                                dist.Gamma(jnp.tile(a_neuron, n_conditions),
                                                           jnp.tile(sigma_neuron, n_conditions)))
        # sample_shape=(n_conditions,))
        theta = a_neuron + a_neuron_per_condition[treat]

        sigma_y_isi = numpyro.sample('sigma_y_isi', dist.Gamma(1, .5))
        numpyro.sample('y_isi', dist.Gamma(theta, sigma_y_isi), obs=fr)
        # numpyro.sample('y_isi', dist.Exponential(theta), obs=fr)

    @staticmethod
    def model_single_normal(data, treat):
        n_conditions = np.unique(treat).shape[0]
        a_neuron = numpyro.sample('mu', dist.Normal(0, 1))
        sigma_neuron = numpyro.sample('sigma', dist.HalfNormal(1))
        a_neuron_per_condition = numpyro.sample('mu_per_condition',
                                                dist.Normal(jnp.tile(a_neuron, n_conditions), sigma_neuron))
        theta = a_neuron + a_neuron_per_condition[treat]
        sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
        numpyro.sample('y', dist.Normal(theta, sigma_obs), obs=data)

    @staticmethod
    def model_single_student(data, treat):
        n_conditions = np.unique(treat).shape[0]
        a_neuron = numpyro.sample('mu', dist.Normal(0, 1))
        sigma_neuron = numpyro.sample('sigma', dist.HalfNormal(1))
        a_neuron_per_condition = numpyro.sample('mu_per_condition',
                                                dist.Normal(jnp.tile(a_neuron, n_conditions), sigma_neuron))
        theta = a_neuron + a_neuron_per_condition[treat]
        sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
        nu_y = numpyro.sample('nu_y', dist.Gamma(2, .1))
        numpyro.sample('y', dist.StudentT(nu_y, theta, sigma_obs), obs=data)

    @staticmethod
    def model_single_lognormal(data, treat):
        n_conditions = np.unique(treat).shape[0]
        a_neuron = numpyro.sample('mu', dist.LogNormal(0, 1))
        sigma_neuron = numpyro.sample('sigma', dist.HalfNormal(1))
        a_neuron_per_condition = numpyro.sample('mu_per_condition',
                                                dist.LogNormal(jnp.tile(a_neuron, n_conditions), sigma_neuron))
        theta = a_neuron + a_neuron_per_condition[treat]
        sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
        numpyro.sample('y', dist.LogNormal(theta, sigma_obs), obs=data)

    @staticmethod
    def model_hier_lognormal_stim(y, stim_on, treat, subject):
        n_conditions = np.unique(treat).shape[0]
        n_subjects = np.unique(subject).shape[0]
        a = numpyro.sample('a', dist.LogNormal(0, 1))
        # b_subject = numpyro.sample('b_subject', dist.Normal(jnp.tile(0, n_subjects), 1))
        # sigma_b_subject = numpyro.sample('sigma_b_subject', dist.HalfNormal(1))
        a_subject = numpyro.sample('a_subject', dist.LogNormal(jnp.tile(0, n_subjects), 1))
        sigma_a_subject = numpyro.sample('sigma_a_subject', dist.HalfNormal(1))
        # b = numpyro.sample('b', dist.Normal(0, 1))
        sigma_b_condition = numpyro.sample('sigma_b_condition', dist.HalfNormal(1))
        b_stim_per_condition = numpyro.sample('b_stim_per_condition', dist.LogNormal(jnp.tile(0, n_conditions), .5))

        theta = (a + a_subject[subject] * sigma_a_subject +
                 (  # b
                     # + b_subject[subject] * sigma_b_subject
                     + b_stim_per_condition[treat] * sigma_b_condition
                 ) * stim_on
                 )

        sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
        numpyro.sample('y', dist.LogNormal(theta, sigma_obs), obs=y)

    @staticmethod
    def model_hier_normal_stim(y, stim_on, treat, subject):
        n_conditions = np.unique(treat).shape[0]
        n_subjects = np.unique(subject).shape[0]
        a = numpyro.sample('a', dist.Normal(0, 1))

        # b_subject = numpyro.sample('b_subject', dist.Normal(jnp.tile(0, n_subjects), 1))
        # sigma_b_subject = numpyro.sample('sigma_b_subject', dist.HalfNormal(1))

        a_subject = numpyro.sample('a_subject', dist.Normal(jnp.tile(0, n_subjects), 1))
        sigma_a_subject = numpyro.sample('sigma_a_subject', dist.HalfNormal(1))

        # b = numpyro.sample('b', dist.Normal(0, 1))
        sigma_b_condition = numpyro.sample('sigma_b_condition', dist.HalfNormal(1))
        b_stim_per_condition = numpyro.sample('b_stim_per_condition', dist.Normal(jnp.tile(0, n_conditions), .5))

        theta = (a + a_subject[subject] * sigma_a_subject +
                 (  # b
                     # + b_subject[subject] * sigma_b_subject
                     + b_stim_per_condition[treat] * sigma_b_condition
                 ) * stim_on
                 )

        sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
        nu_y = numpyro.sample('nu_y', dist.Gamma(1, .1))
        numpyro.sample('y', dist.StudentT(nu_y, theta, sigma_obs), obs=y)


def fit_numpyro(progress_bar=False, model=None, n_draws=1000, num_chains=1, **kwargs):
    if model is None:
        model = Models.model_hier_normal_stim
    numpyro.set_host_device_count(4)
    mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=n_draws, num_chains=num_chains, progress_bar=progress_bar)
    mcmc.run(random.PRNGKey(16), **kwargs)

    # arviz convert
    trace = az.from_numpyro(mcmc)
    # Print diagnostics
    print(f"n(Divergences) = {trace.sample_stats.diverging.sum(['chain', 'draw']).values}")
    return trace.posterior

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam


def reparam_model(model):
    return reparam(model, config={'x': LocScaleReparam(0)})


def model_single_normal_stim(y, treatment, condition):
    n_conditions = np.unique(condition).shape[0]
    a = numpyro.sample('a', dist.Normal(0, 1))
    b = numpyro.sample('b', dist.Normal(0, 1))
    sigma_b = numpyro.sample('sigma_b', dist.HalfNormal(1))
    b_stim_per_condition = numpyro.sample('b_stim', dist.Normal(jnp.tile(b, n_conditions), sigma_b))
    theta = a + b_stim_per_condition[condition] * treatment

    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
    numpyro.sample('y', dist.Normal(theta, sigma_obs), obs=y)


def sample_y(dist_y, theta, y):
    if dist_y == 'gamma':
        sigma_obs = numpyro.sample('sigma_obs', dist.Exponential(1))
    else:
        sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))

    if dist_y == 'student':
        numpyro.sample('y', dist.StudentT(numpyro.sample('nu_y', dist.Gamma(1, .1)), theta, sigma_obs), obs=y)
    elif dist_y == 'normal':
        numpyro.sample('y', dist.Normal(theta, sigma_obs), obs=y)
    elif dist_y == 'lognormal':
        numpyro.sample('y', dist.LogNormal(theta, sigma_obs), obs=y)
    elif dist_y == 'gamma':
        numpyro.sample('y', dist.Gamma(jnp.exp(theta), sigma_obs), obs=y)
    elif dist_y == 'exponential':
        numpyro.sample('y', dist.Exponential(jnp.exp(theta)), obs=y)
    else:
        raise NotImplementedError


def model_single(y, condition, dist_y='normal'):
    n_conditions = np.unique(condition).shape[0]
    a_neuron = numpyro.sample('mu', dist.Normal(0, 1))
    sigma_neuron = numpyro.sample('sigma', dist.HalfNormal(1))
    a_neuron_per_condition = numpyro.sample('mu_per_condition',
                                            dist.Normal(jnp.tile(a_neuron, n_conditions), sigma_neuron))
    theta = a_neuron + a_neuron_per_condition[condition]
    sample_y(dist_y=dist_y, theta=theta, y=y)


def model_hierarchical(y, condition=None, group=None, treatment=None, dist_y='normal', add_group_slope=False,
                       add_group_intercept=True):
    n_subjects = np.unique(group).shape[0]
    # condition = condition.astype(int)
    intercept = numpyro.sample('a', dist.Normal(0, 1))
    # intercept = 0
    if (group is not None) and add_group_intercept:
        sigma_a_subject = numpyro.sample('sigma_a_subject', dist.HalfNormal(1))
        a_subject = numpyro.sample('a_subject', dist.HalfNormal(jnp.tile(1, n_subjects)))
        intercept += a_subject[group] * sigma_a_subject

    if condition is None:
        slope = numpyro.sample('b', dist.Normal(0, 1))
    else:
        n_conditions = np.unique(condition).shape[0]
        # b_stim_per_condition = numpyro.sample('b_stim_per_condition',
        #                                       dist.Normal(jnp.tile(0, n_conditions), 1))
        # Robust slopes:
        b_stim_per_condition = numpyro.sample('b_stim_per_condition',
                                              dist.StudentT(1, jnp.tile(4, n_conditions), 2))

        sigma_b_condition = numpyro.sample('sigma_b_condition', dist.HalfNormal(1))
        slope = b_stim_per_condition[condition] * sigma_b_condition

    if (group is not None) and add_group_slope:
        sigma_b_group = numpyro.sample('sigma_b_group', dist.HalfNormal(1))
        b_stim_per_group = numpyro.sample('b_stim_per_subject', dist.Normal(jnp.tile(0, n_subjects), 1))
        slope += b_stim_per_group[group] * sigma_b_group

    if treatment is not None:
        slope *= treatment

    sample_y(dist_y=dist_y, theta=intercept + slope, y=y)


def model_hier_stim_one_codition(y, treatment=None, group=None, dist_y='normal', **kwargs):
    n_subjects = np.unique(group).shape[0]
    a = numpyro.sample('a', dist.Normal(0, 1))

    # b_subject = numpyro.sample('b_subject', dist.Normal(jnp.tile(0, n_subjects), 1))
    # sigma_b_subject = numpyro.sample('sigma_b_subject', dist.HalfNormal(1))

    a_subject = numpyro.sample('a_subject', dist.Normal(jnp.tile(0, n_subjects), 1))
    sigma_a_subject = numpyro.sample('sigma_a_subject', dist.HalfNormal(1))

    b = numpyro.sample('b_stim_per_condition', dist.Normal(0, 1))

    theta = a + a_subject[group] * sigma_a_subject
    slope = b
    if treatment is not None:
        slope = slope * treatment
    theta += slope

    sample_y(dist_y=dist_y, theta=theta, y=y)

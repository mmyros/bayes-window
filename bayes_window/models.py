# from contextlib import nullcontext

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam


def reparam_model(model):
    return reparam(model, config={'x': LocScaleReparam(0)})


def sample_y(dist_y, theta, y, sigma_obs=None):
    if not sigma_obs:
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
                       add_group_intercept=True, add_condition_slope=True, group2=None, add_group2_slope=False,
                       robust_slopes=True):
    n_subjects = np.unique(group).shape[0]
    if (group is not None) and add_group_intercept:
        sigma_a_group = numpyro.sample('sigma_intercept_per_group', dist.HalfNormal(100))
        a_group = numpyro.sample(f'intercept_per_group', dist.HalfNormal(jnp.tile(100, n_subjects)))
        intercept = a_group[group] * sigma_a_group
    else:
        intercept = numpyro.sample('intercept', dist.Normal(0, 100))

    if (condition is None) or (np.unique(condition).size < 2):
        add_condition_slope = False  # no need for per-condition slope
    if not add_condition_slope:
        slope = numpyro.sample('slope', dist.Normal(0, 100))
    else:
        n_conditions = np.unique(condition).shape[0]
        if robust_slopes:
            # Robust slopes:
            b_per_condition = numpyro.sample('slope_per_condition',
                                             dist.StudentT(1, jnp.tile(0, n_conditions), 100))
        else:
            b_per_condition = numpyro.sample('slope_per_condition',
                                             dist.Normal(jnp.tile(0, n_conditions), 100))

        sigma_b_condition = numpyro.sample('sigma_slope_per_condition', dist.HalfNormal(100))
        slope = b_per_condition[condition] * sigma_b_condition

    if (group is not None) and add_group_slope:
        sigma_b_group = numpyro.sample('sigma_slope_per_group', dist.HalfNormal(100))
        b_per_group = numpyro.sample('slope_per_group', dist.Normal(jnp.tile(0, n_subjects), 100))
        slope = slope + b_per_group[group] * sigma_b_group

    if (group2 is not None) and add_group2_slope:
        sigma_b_group = numpyro.sample('sigma_slope_per_group2', dist.HalfNormal(100))
        b_per_group = numpyro.sample('slope_per_group2', dist.Normal(jnp.tile(0, n_subjects), 100))
        slope = slope + b_per_group[group] * sigma_b_group

    if treatment is not None:
        slope = slope * treatment

    sample_y(dist_y=dist_y, theta=intercept + slope, y=y)


def model_hierarchical_for_render(y, condition=None, group=None, treatment=None, dist_y='normal', add_group_slope=False,
                                  add_group_intercept=True):
    # Hyperpriors:
    a = numpyro.sample('hyper_a', dist.Normal(0., 5))
    sigma_a = numpyro.sample('hyper_sigma_a', dist.HalfNormal(1))

    # with numpyro.plate('n_groups1', np.unique(group).size) if add_group_slope else nullcontext():
    #     sigma_b = numpyro.sample('sigma_b', dist.HalfNormal(1))
    #     b = numpyro.sample('b', dist.Normal(0., 1))
    #     with numpyro.plate('n_conditions', np.unique(condition).size):
    #         # Varying slopes:
    #         b_condition = numpyro.sample('slope', dist.Normal(b, sigma_b))

    sigma_b = numpyro.sample('hyper_sigma_b_condition', dist.HalfNormal(1))
    b = numpyro.sample('hyper_b_condition', dist.Normal(0., 1))
    with (  # numpyro.plate('n_groups1', np.unique(group).size) if add_group_slope else
        numpyro.plate('n_conditions', np.unique(condition).size)):
        # Varying slopes:
        b_condition = numpyro.sample('slope_per_condition', dist.Normal(b, sigma_b))

    sigma_b = numpyro.sample('hyper_sigma_b_group', dist.HalfNormal(1))
    b = numpyro.sample('hyper_b_group', dist.Normal(0., 1))
    if add_group_slope:
        with numpyro.plate('n_groups', np.unique(group).size):
            # Varying slopes:
            b_group = numpyro.sample('b_group', dist.Normal(b, sigma_b))
    else:
        b_group = 0
    if add_group_intercept:
        with numpyro.plate('n_groups', np.unique(group).size):
            # Varying intercepts:
            a_group = numpyro.sample('a_group', dist.Normal(a, sigma_a))
        theta = a_group[group] + (b_condition[condition] + b_group[group]) * treatment
    else:
        theta = a + (b_condition[condition] + b_group[group]) * treatment
    sample_y(dist_y=dist_y, theta=theta, y=y)


def model_hierarchical_next(y, condition=None, group=None, treatment=None, dist_y='normal', add_group_slope=False,
                            add_group_intercept=True):
    # Hyperpriors:
    a = numpyro.sample('a', dist.Normal(0., 5))
    sigma_a = numpyro.sample('sigma_a', dist.HalfNormal(1))
    b = numpyro.sample('b', dist.Normal(0., 1))
    sigma_b = numpyro.sample('sigma_b', dist.HalfNormal(1))

    with numpyro.plate('n_conditions', np.unique(condition).size):# if add_group_slope else nullcontext():
        # Varying slopes:
        b_condition = numpyro.sample('slope_per_group', dist.Normal(b, sigma_b))

    with numpyro.plate('n_groups', np.unique(group).size):# if add_group_intercept else nullcontext():
        # Varying intercepts:
        a_group = numpyro.sample('a_group', dist.Normal(a, sigma_a))
        theta = a_group[group] + b_condition[condition] * treatment
        sample_y(dist_y=dist_y, theta=theta, y=y)


def model_hier_stim_one_codition(y, treatment=None, group=None, dist_y='normal', **kwargs):
    n_subjects = np.unique(group).shape[0]
    a = numpyro.sample('a', dist.Normal(0, 1))

    # b_subject = numpyro.sample('b_subject', dist.Normal(jnp.tile(0, n_subjects), 1))
    # sigma_b_subject = numpyro.sample('sigma_b_subject', dist.HalfNormal(1))

    a_group = numpyro.sample(f'intercept_per_group', dist.Normal(jnp.tile(0, n_subjects), 100))
    sigma_a_group = numpyro.sample('sigma_intercept_per_group', dist.HalfNormal(1))

    b = numpyro.sample('slope', dist.Normal(0, 1))

    theta = a + a_group[group] * sigma_a_group
    slope = b
    if treatment is not None:
        slope = slope * treatment
    theta += slope

    sample_y(dist_y=dist_y, theta=theta, y=y)

# from contextlib import nullcontext
import warnings

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
    elif dist_y == 'gamma_raw':
        numpyro.sample('y', dist.Gamma(theta, sigma_obs), obs=y)
    elif dist_y == 'poisson':
        numpyro.sample('y', dist.Poisson(theta), obs=y)
    elif dist_y == 'exponential':
        numpyro.sample('y', dist.Exponential(jnp.exp(theta)), obs=y)
    elif dist_y == 'exponential_raw':
        numpyro.sample('y', dist.Exponential(theta), obs=y)
    elif dist_y == 'uniform':
        numpyro.sample('y', dist.Uniform(0, 1), obs=y)
    else:
        raise NotImplementedError


def model_single(y, condition, group=None, dist_y='normal', add_group_intercept=True, add_intercept=True, **kwargs):
    n_conditions = np.unique(condition).shape[0]
    sigma = numpyro.sample('sigma', dist.HalfNormal(10))
    mu_per_condition = numpyro.sample('mu_per_condition',
                                     dist.Normal(jnp.tile(0, n_conditions), sigma))
    theta = mu_per_condition[condition]
    if add_intercept:
        mu = numpyro.sample('mu', dist.Normal(0, 10))
        theta += mu
    if group is not None and add_group_intercept:
        if dist_y == 'poisson':
            a_group = numpyro.sample('mu_intercept_per_group', dist.HalfNormal(jnp.tile(10, np.unique(group).shape[0])))
        else:
            sigma_group = numpyro.sample('sigma_intercept_per_group', dist.HalfNormal(10))
            a_group = numpyro.sample('mu_intercept_per_group', dist.Normal(jnp.tile(0, np.unique(group).shape[0]),
                                                                           sigma_group))
        theta += a_group[group]
    sample_y(dist_y=dist_y, theta=theta, y=y)


def model_hierarchical(y, condition=None, group=None, treatment=None, dist_y='normal', add_group_slope=False,
                       add_group_intercept=True,
                       add_condition_slope=True, group2=None, add_group2_slope=False,
                       center_intercept=True, center_slope=False, robust_slopes=False,
                       add_condition_intercept=False,
                       dist_slope=dist.Normal
                       ):
    if group is not None and not add_group_intercept and not add_group_slope:
        warnings.warn(f'No group intercept or group slope requested. What was the point of providing group {group}?')
    n_subjects = np.unique(group).shape[0]

    n_conditions = np.unique(condition).shape[0]
    if (condition is None) or not n_conditions:
        add_condition_slope = False  # Override
        add_condition_intercept = False

    if center_intercept:
        intercept = numpyro.sample('intercept', dist.Normal(0, 100))
    else:
        intercept = 0
    if (group is not None) and add_group_intercept:
        if dist_y == 'poisson':
            print('poisson intercepts')
            a_group = numpyro.sample(f'mu_intercept_per_group', dist.Poisson(jnp.tile(0, n_subjects)))
            intercept += a_group
        else:
            sigma_a_group = numpyro.sample('sigma_intercept_per_group', dist.HalfNormal(100))
            a_group = numpyro.sample(f'mu_intercept_per_group', dist.Normal(jnp.tile(0, n_subjects), 10))
            intercept += (a_group[group] * sigma_a_group)

    if add_condition_intercept:
        intercept_per_condition = numpyro.sample('intercept_per_condition',
                                                 dist.Normal(jnp.tile(0, n_conditions), 100))

        sigma_intercept_per_condition = numpyro.sample('sigma_intercept_per_condition', dist.HalfNormal(100))
        intercept += intercept_per_condition[condition] * sigma_intercept_per_condition

    if not add_condition_slope:
        center_slope = True  # override center_slope

    if center_slope:
        slope = numpyro.sample('slope', dist_slope(0, 1))
    else:
        slope = 0
    if add_condition_slope:
        if robust_slopes:
            # Robust slopes:
            b_per_condition = numpyro.sample('slope_per_condition',
                                             dist.StudentT(1, jnp.tile(0, n_conditions), 100))
        else:
            b_per_condition = numpyro.sample('slope_per_condition',
                                             dist_slope(jnp.tile(0, n_conditions), 100))

        slope = slope + b_per_condition[condition]

    if (group is not None) and add_group_slope:
        b_per_group = numpyro.sample('slope_per_group', dist_slope(jnp.tile(0, n_subjects), 1))
        slope = slope + b_per_group[group]

    if (group2 is not None) and add_group2_slope:
        b_per_group = numpyro.sample('slope_per_group2', dist_slope(jnp.tile(0, n_subjects), 1))
        slope = slope + b_per_group[group]

    if type(intercept) is int:
        print('Caution: No intercept')
    if treatment is not None:
        slope = slope * treatment

    sample_y(dist_y=dist_y, theta=intercept + slope, y=y)


def model_regression_simple(y, condition, treatment, **kwargs):
    n_conditions = np.unique(condition).shape[0]
    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
    intercept = numpyro.sample('intercept', dist.Normal(0, 1))

    #     b_per_condition = numpyro.sample('slope_per_condition',
    #                                      dist.StudentT(4,
    #                                                    jnp.tile(0, n_conditions),
    #                                                    jnp.tile(2, n_conditions),))
    b_per_condition = numpyro.sample('slope_per_condition',
                                     dist.Normal(jnp.tile(0, n_conditions), 10))
    theta = intercept + b_per_condition[condition] * treatment
    numpyro.sample('y', dist.Normal(theta, sigma_obs), obs=y)


def model_hierarchical_for_render(y, condition=None, group=None, treatment=None, dist_y='normal', add_group_slope=False,
                                  add_group_intercept=True, add_condition_slope=True):
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

    with numpyro.plate('n_conditions', np.unique(condition).size):  # if add_group_slope else nullcontext():
        # Varying slopes:
        b_condition = numpyro.sample('slope_per_group', dist.Normal(b, sigma_b))

    with numpyro.plate('n_groups', np.unique(group).size):  # if add_group_intercept else nullcontext():
        # Varying intercepts:
        a_group = numpyro.sample('a_group', dist.Normal(a, sigma_a))
        theta = a_group[group] + b_condition[condition] * treatment
        sample_y(dist_y=dist_y, theta=theta, y=y)

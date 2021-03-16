import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist


def model_single_normal_stim(y, treatment, condition):
    n_conditions = np.unique(condition).shape[0]
    a = numpyro.sample('a', dist.Normal(0, 1))
    b = numpyro.sample('b', dist.Normal(0, 1))
    sigma_b = numpyro.sample('sigma_b', dist.HalfNormal(1))
    b_stim_per_condition = numpyro.sample('b_stim', dist.Normal(jnp.tile(b, n_conditions), sigma_b))
    theta = a + b_stim_per_condition[condition] * treatment

    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
    numpyro.sample('y', dist.Normal(theta, sigma_obs), obs=y)


def sample_y(dist_y, theta, sigma_obs, y):
    if dist_y == 'student':
        nu_y = numpyro.sample('nu_y', dist.Gamma(1, .1))
        numpyro.sample('y', dist.StudentT(nu_y, theta, sigma_obs), obs=y)
    elif dist_y == 'normal':
        numpyro.sample('y', dist.Normal(theta, sigma_obs), obs=y)
    elif dist_y == 'lognormal':
        numpyro.sample('y', dist.LogNormal(theta, sigma_obs), obs=y)
    else:
        raise NotImplementedError


def model_single(y, condition, dist_y='normal'):
    n_conditions = np.unique(condition).shape[0]
    a_neuron = numpyro.sample('mu', dist.Normal(0, 1))
    sigma_neuron = numpyro.sample('sigma', dist.HalfNormal(1))
    a_neuron_per_condition = numpyro.sample('mu_per_condition',
                                            dist.Normal(jnp.tile(a_neuron, n_conditions), sigma_neuron))
    theta = a_neuron + a_neuron_per_condition[condition]
    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
    sample_y(dist_y=dist_y, theta=theta, sigma_obs=sigma_obs, y=y)


def model_hierarchical(y, condition=None, group=None, treatment=None, dist_y='normal'):
    n_conditions = np.unique(condition).shape[0]

    # sigmas
    sigma_a_subject = numpyro.sample('sigma_a_subject', dist.HalfNormal(1))
    sigma_b_condition = numpyro.sample('sigma_b_condition', dist.HalfNormal(1))

    # b
    b_stim_per_condition = numpyro.sample('b_stim_per_condition', dist.Normal(jnp.tile(0, n_conditions), .5))
    intercept = numpyro.sample('a', dist.Normal(0, 1))

    if group is not None:
        n_subjects = np.unique(group).shape[0]
        a_subject = numpyro.sample('a_subject', dist.Normal(jnp.tile(0, n_subjects), 1))
        intercept += a_subject[group] * sigma_a_subject

    if condition is None:
        slope = numpyro.sample('b', dist.Normal(0, 1))
    else:
        slope = b_stim_per_condition[condition] * sigma_b_condition
    if treatment is not None:
        slope *= treatment
    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
    sample_y(dist_y=dist_y, theta=intercept+slope, sigma_obs=sigma_obs, y=y)


def model_hierarchical_gamma(y, condition=None, group=None, treatment=None, dist_y='normal'):
    n_conditions = np.unique(condition).shape[0]

    # sigmas
    sigma_a_subject = numpyro.sample('sigma_a_subject', dist.Gamma(4, 1))
    sigma_b_condition = numpyro.sample('sigma_b_condition', dist.Gamma(4, 1))

    # b
    b_stim_per_condition = numpyro.sample('b_stim_per_condition', dist.Gamma(jnp.tile(4, n_conditions), 1))
    intercept = numpyro.sample('a', dist.Gamma(4, 1))

    if group is not None:
        n_subjects = np.unique(group).shape[0]
        a_subject = numpyro.sample('a_subject', dist.Gamma(jnp.tile(4, n_subjects), 1))
        intercept += a_subject[group] * sigma_a_subject

    if condition is None:
        slope = numpyro.sample('b', dist.Gamma(4, 1))
    else:
        slope = b_stim_per_condition[condition] * sigma_b_condition
    if treatment is not None:
        slope *= treatment
    sigma_obs = numpyro.sample('sigma_obs', dist.Gamma(1, .5))
    numpyro.sample('y', dist.Gamma(intercept+slope, sigma_obs), obs=y)


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

    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
    sample_y(dist_y=dist_y, theta=theta, sigma_obs=sigma_obs, y=y)

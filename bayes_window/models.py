import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist


def model_single_normal_stim(y, stim, treat):
    n_conditions = np.unique(treat).shape[0]
    a = numpyro.sample('a', dist.Normal(0, 1))
    b = numpyro.sample('b', dist.Normal(0, 1))
    sigma_b = numpyro.sample('sigma_b', dist.HalfNormal(1))
    b_stim_per_condition = numpyro.sample('b_stim', dist.Normal(jnp.tile(b, n_conditions), sigma_b))
    theta = a + b_stim_per_condition[treat] * stim

    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
    numpyro.sample('y', dist.Normal(theta, sigma_obs), obs=y)


def model_single_exponential(y, treat):
    n_conditions = np.unique(treat).shape[0]
    a_neuron = numpyro.sample('FR_neuron', dist.Gamma(4, 1))
    sigma_neuron = numpyro.sample('sigma_neuron', dist.Gamma(4, 1))
    a_neuron_per_condition = numpyro.sample('FR_neuron_per_condition',
                                            dist.Gamma(jnp.tile(a_neuron, n_conditions),
                                                       jnp.tile(sigma_neuron, n_conditions)))
    # sample_shape=(n_conditions,))
    theta = a_neuron + a_neuron_per_condition[treat]

    sigma_y_isi = numpyro.sample('sigma_y_isi', dist.Gamma(1, .5))
    numpyro.sample('y_isi', dist.Gamma(theta, sigma_y_isi), obs=y)
    # numpyro.sample('y_isi', dist.Exponential(theta), obs=fr)


def model_single_normal(y, treat):
    n_conditions = np.unique(treat).shape[0]
    a_neuron = numpyro.sample('mu', dist.Normal(0, 1))
    sigma_neuron = numpyro.sample('sigma', dist.HalfNormal(1))
    a_neuron_per_condition = numpyro.sample('mu_per_condition',
                                            dist.Normal(jnp.tile(a_neuron, n_conditions), sigma_neuron))
    theta = a_neuron + a_neuron_per_condition[treat]
    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
    numpyro.sample('y', dist.Normal(theta, sigma_obs), obs=y)


def model_single_student(y, treat):
    n_conditions = np.unique(treat).shape[0]
    a_neuron = numpyro.sample('mu', dist.Normal(0, 1))
    sigma_neuron = numpyro.sample('sigma', dist.HalfNormal(1))
    a_neuron_per_condition = numpyro.sample('mu_per_condition',
                                            dist.Normal(jnp.tile(a_neuron, n_conditions), sigma_neuron))
    theta = a_neuron + a_neuron_per_condition[treat]
    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
    nu_y = numpyro.sample('nu_y', dist.Gamma(2, .1))
    numpyro.sample('y', dist.StudentT(nu_y, theta, sigma_obs), obs=y)


def model_single_lognormal(y, treat):
    n_conditions = np.unique(treat).shape[0]
    a_neuron = numpyro.sample('mu', dist.LogNormal(0, 1))
    sigma_neuron = numpyro.sample('sigma', dist.HalfNormal(1))
    a_neuron_per_condition = numpyro.sample('mu_per_condition',
                                            dist.LogNormal(jnp.tile(a_neuron, n_conditions), sigma_neuron))
    theta = a_neuron + a_neuron_per_condition[treat]
    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
    numpyro.sample('y', dist.LogNormal(theta, sigma_obs), obs=y)


def model_hier_lognormal_stim(y, stim, treat, subject):
    # Subject = intercept
    # Treat=slope
    # stim=overall slope
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
             ) * stim
             )

    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
    numpyro.sample('y', dist.LogNormal(theta, sigma_obs), obs=y)


def model_hier_normal_stim(y, stim, treat, subject):
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
             ) * stim
             )

    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
    nu_y = numpyro.sample('nu_y', dist.Gamma(1, .1))
    numpyro.sample('y', dist.StudentT(nu_y, theta, sigma_obs), obs=y)


def model_hier_stim_one_codition(y, stim, subject, dist_y='student', **kwargs):
    n_subjects = np.unique(subject).shape[0]
    a = numpyro.sample('a', dist.Normal(0, 1))

    # b_subject = numpyro.sample('b_subject', dist.Normal(jnp.tile(0, n_subjects), 1))
    # sigma_b_subject = numpyro.sample('sigma_b_subject', dist.HalfNormal(1))

    a_subject = numpyro.sample('a_subject', dist.Normal(jnp.tile(0, n_subjects), 1))
    sigma_a_subject = numpyro.sample('sigma_a_subject', dist.HalfNormal(1))

    b = numpyro.sample('b_stim_per_condition', dist.Normal(0, 1))

    theta = a + a_subject[subject] * sigma_a_subject + b * stim

    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1))
    nu_y = numpyro.sample('nu_y', dist.Gamma(1, .1))
    if dist_y == 'student':
        numpyro.sample('y', dist.StudentT(nu_y, theta, sigma_obs), obs=y)
    elif dist_y == 'normal':
        numpyro.sample('y', dist.Normal(theta, sigma_obs), obs=y)
    elif dist_y == 'lognormal':
        numpyro.sample('y', dist.LogNormal(theta, sigma_obs), obs=y)
    else:
        raise ValueError

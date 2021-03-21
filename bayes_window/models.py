import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam


def reparam_model(model):
    return reparam(model, config={'x': LocScaleReparam(0)})


def twostep(y, treatment, condition, group, dist_y='Exponential' or dist.Exponential, **kwargs):

    mouse_index_of_condition = np.array([group[condition == ineuron][0] for ineuron in np.unique(condition)])

    num_conditions = np.unique(condition).shape[0]
    num_treatments = np.unique(treatment).shape[0]

    # ========= Step 1: fit individual firing rates ==============
    if 'gamma' in dist_y:
        a_neuron_per_condition = numpyro.sample('FR_neuron_per_condition',
                                                dist.Gamma(jnp.ones([num_conditions, num_treatments]), 1))
        theta = a_neuron_per_condition[condition, treatment]  # + a_neuron[neuron]

        sigma_y_isi = numpyro.sample('sigma_y_isi', dist.Gamma(1, 1))
        numpyro.sample('y_isi', dist.Gamma(theta, sigma_y_isi), obs=y)

    elif 'normal' in dist_y:
        a_neuron_per_condition = numpyro.sample('FR_neuron_per_condition',
                                                dist.HalfNormal(jnp.ones([num_conditions, num_treatments])))
        theta = a_neuron_per_condition[condition, treatment]  # + a_neuron[neuron]

        sigma_y_isi = numpyro.sample('sigma_y_isi', dist.HalfNormal(1))
        numpyro.sample('y_isi', dist.Normal(theta, sigma_y_isi), obs=y)


    elif 'exponential' in dist_y:

        a_neuron_per_condition = numpyro.sample('FR_neuron_per_condition',
                                                dist.Exponential(jnp.ones([num_conditions, num_treatments])))
        theta = a_neuron_per_condition[condition, treatment]  # + a_neuron[neuron]
        numpyro.sample('y_isi', dist.Exponential(theta), obs=y)

    else:
        a_neuron_per_condition = numpyro.sample('FR_neuron_per_condition',
                                                dist.Exponential(jnp.ones([num_conditions, num_treatments])))
        theta = a_neuron_per_condition[condition, treatment]  # + a_neuron[neuron]
        numpyro.sample('y_isi', dist_y(theta), obs=y)

    # How to go from this to 1 trial, 1d index?
    # 1. flatten() will stack each row horizontally

    y = a_neuron_per_condition.flatten()
    # fr is 1d, dim=ntreat*ncondition
    # corresponding indices:
    # y is 36, and df is 720
    # treatment = np.stack([np.tile(np.unique(treatment)[itreat], num_treatments)
    #                       for itreat in set(treatment)]).flatten()
    treatment = np.tile(np.unique(treatment), num_conditions)
    condition = np.tile(np.unique(condition), num_treatments)
    group = mouse_index_of_condition[condition.astype(int)]
    # print(num_conditions,num_treatments)
    # print(f'y {y.shape},treat {treatment.shape}, cond{condition.shape}, group {group.shape}')

    # ========= Step 2: hier  ==============

    #TODO call model_hierarchical
    hier(y, treatment, condition, group, num_conditions)
    # model_hierarchical(y=y, condition=condition, group=group, treatment=treatment, **kwargs)


def hier(fr,treat, neuron, mouse, num_neurons):
    num_mice = np.unique(mouse).shape[0]
    a = numpyro.sample('a', dist.Normal(0, 1))
    sigma_a = numpyro.sample('sigma_a', dist.HalfNormal(1))
    a_neuron = numpyro.sample('a_neuron', dist.Normal(jnp.zeros(num_neurons), 1))
    sigma_a_neuron = numpyro.sample('sigma_a_neuron', dist.HalfNormal(1))

    b = numpyro.sample('b', dist.Normal(0, 1))
    sigma_b = numpyro.sample('sigma_b', dist.HalfNormal(1))

    b_neuron = numpyro.sample('b_neuron', dist.Normal(jnp.zeros(num_neurons), 1))
    sigma_b_neuron = numpyro.sample('sigma_b_neuron', dist.HalfNormal(1))

    # Mouse level:
    sigma_b_mouse = numpyro.sample('sigma_b_mouse', dist.HalfNormal(1))

    b_mouse = numpyro.sample('b_mouse', dist.Normal(jnp.zeros(num_mice), 1))

    mu = a * sigma_a + a_neuron[neuron] * sigma_a_neuron + (b * sigma_b
                                                            + b_neuron[neuron] * sigma_b_neuron
                                                            + b_mouse[mouse] * sigma_b_mouse) * treat

    sigma_y = numpyro.sample('sigma_y', dist.HalfNormal(1))
    # nu_y = numpyro.sample('nu_y', dist.HalfNormal(1))
    nu_y = numpyro.sample('nu_y', dist.Gamma(2, .1))

    numpyro.sample('y', dist.StudentT(nu_y, mu, sigma_y), obs=fr)


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
        nu_y = numpyro.sample('nu_y', dist.Gamma(1, .1))
        numpyro.sample('y', dist.StudentT(nu_y, theta, sigma_obs), obs=y)
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

from pdb import set_trace
def model_hierarchical(y, condition=None, group=None, treatment=None, dist_y='normal', add_group_slope=False,
                       add_group_intercept=True):
    n_subjects = np.unique(group).shape[0]
    # condition = condition.astype(int)
    intercept = numpyro.sample('a', dist.Normal(0, 1))
    # intercept = 0
    if (group is not None) and add_group_intercept:
        sigma_a_subject = numpyro.sample('sigma_a_subject', dist.HalfNormal(10))
        a_subject = numpyro.sample('a_subject', dist.HalfNormal(jnp.tile(10, n_subjects)))
        intercept += a_subject[group] * sigma_a_subject

    if condition is None:
        slope = numpyro.sample('b', dist.Normal(0, 10))
    else:
        n_conditions = np.unique(condition).shape[0]
        b_stim_per_condition = numpyro.sample('b_stim_per_condition', dist.Normal(jnp.tile(0, n_conditions), 10))
        sigma_b_condition = numpyro.sample('sigma_b_condition', dist.HalfNormal(10))
        slope = b_stim_per_condition[condition] * sigma_b_condition

    if (group is not None) and add_group_slope:
        sigma_b_group = numpyro.sample('sigma_b_group', dist.HalfNormal(10))
        b_stim_per_group = numpyro.sample('b_stim_per_subject', dist.Normal(jnp.tile(0, n_subjects), 10))
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

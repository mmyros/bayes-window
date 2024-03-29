# import pdb
# import jax.numpy as jnp
# import numpyro
# import numpyro.optim as optim
# from jax import lax
import os
import warnings

import arviz as az
import jax
import numpyro
import numpyro.optim as optim
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation

from . import models


def select_device(use_gpu, num_chains):
    if use_gpu:
        try:
            numpyro.set_platform('gpu')
            numpyro.set_host_device_count(1)
        except RuntimeError as e:
            warnings.warn(f'No GPU found: {e}')
            numpyro.set_platform('cpu')
    else:
        numpyro.set_platform('cpu')
        numpyro.set_host_device_count(min((num_chains, os.cpu_count())))
    # Sanity check
    jax.lib.xla_bridge.get_backend().platform


def fit_numpyro(progress_bar=False, model=None, num_warmup=1000,
                n_draws=200, num_chains=4, sampler=NUTS, use_gpu=False,
                **kwargs):
    if 'bayes_window_test_mode' in os.environ:
        # Override settings with minimal
        use_gpu = False
        num_warmup = 5
        n_draws = 5
        num_chains = 1
    select_device(use_gpu, num_chains)
    model = model or models.model_hierarchical
    mcmc = MCMC(sampler(model=model,
                        find_heuristic_step_size=True,
                        target_accept_prob=0.99,
                        # init_strategy=numpyro.infer.init_to_uniform
                        ),
                num_warmup=num_warmup, num_samples=n_draws, num_chains=num_chains, progress_bar=progress_bar,
                chain_method='parallel'
                )
    mcmc.run(jax.random.PRNGKey(16), **kwargs)

    # arviz convert
    try:
        trace = az.from_numpyro(mcmc)
    except AttributeError:
        trace = az.from_dict(mcmc.get_samples())
        print(trace.posterior)

    # Print diagnostics
    if 'sample_stats' in trace:
        if trace.sample_stats.diverging.sum(['chain', 'draw']).values > 0:
            print(f"n(Divergences) = {trace.sample_stats.diverging.sum(['chain', 'draw']).values}")
    return trace, mcmc


def fit_svi(model, n_draws=1000,
            autoguide=AutoLaplaceApproximation,
            loss=Trace_ELBO(),
            optim=optim.Adam(step_size=.00001),
            num_warmup=2000,
            use_gpu=False,
            num_chains=1,
            progress_bar=False,
            sampler=None,
            **kwargs):
    select_device(use_gpu, num_chains)
    guide = autoguide(model)
    svi = SVI(
        model=model,
        guide=guide,
        loss=loss,
        optim=optim,
        **kwargs
    )
    # Experimental interface:
    svi_result = svi.run(jax.random.PRNGKey(0), num_steps=num_warmup, stable_update=True,
                          progress_bar=progress_bar)
    # Old:
    post = guide.sample_posterior(jax.random.PRNGKey(1), params=svi_result.params,
                                  sample_shape=(1, n_draws))
    # New:
    #predictive = Predictive(guide,  params=svi_result.params, num_samples=n_draws)
    #post = predictive(jax.random.PRNGKey(1), **kwargs)

    # Old interface:
    # init_state = svi.init(jax.random.PRNGKey(0))
    # state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(n_draws))#, length=num_warmup)
    # svi_params = svi.get_params(state)
    # post = guide.sample_posterior(jax.random.PRNGKey(1), svi_params, (1, n_draws))

    trace = az.from_dict(post)
    return trace, post

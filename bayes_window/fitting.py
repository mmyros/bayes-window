import arviz as az
import numpyro
from jax import random
from numpyro.infer import MCMC, NUTS

from . import models


def fit_numpyro(progress_bar=False, model=None, num_warmup=1000,
                n_draws=1000, num_chains=5, convert_to_arviz=True, sampler=NUTS, use_gpu=False,
                **kwargs):
    if use_gpu:
        numpyro.set_platform('gpu')
    else:
        numpyro.set_platform('cpu')
    numpyro.set_host_device_count(num_chains)
    model = model or models.model_hierarchical
    mcmc = MCMC(sampler(model=model,
                        find_heuristic_step_size=True),
                num_warmup=num_warmup, num_samples=n_draws, num_chains=num_chains, progress_bar=progress_bar,
                chain_method='parallel'
                )
    mcmc.run(random.PRNGKey(16), **kwargs)
    # arviz convert
    trace = az.from_numpyro(mcmc)
    # Print diagnostics
    if trace.sample_stats.diverging.sum(['chain', 'draw']).values > 0:
        print(f"n(Divergences) = {trace.sample_stats.diverging.sum(['chain', 'draw']).values}")
    if convert_to_arviz:
        return trace
    else:
        return mcmc


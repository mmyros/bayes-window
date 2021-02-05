import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import numpyro
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive, log_likelihood
import arviz as az

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform


def split_train_predict1(model, df, data_cols, fitting_args=None, index_cols=('mouse_code', 'neuron_code', 'stim'),
                         draws=1000, warmup=500, num_chains=1, progress_bar=True):
    """

    :param data_cols: ['column_name1' 'column_name2']
    :param model:  model function
    :param df:
    :param fitting_args:
    :param index_cols:
    :param draws:
    :param warmup:
    :param num_chains:
    :param progress_bar:
    :param kwargs: eg:
        y=df['con_coherence_magnitude_near'].values,
        stim_on=trans(df['event'].values),
        treat=trans(df['condition_code'].values),
        subject=trans(df['subject'].values),
    :return:
    """
    if fitting_args is None:
        fitting_args = {}
    # split into training and test
    df_train, df_test = train_test_split(df, train_size=.5, test_size=.5, stratify=df[list(index_cols)])

    # fit
    mcmc = MCMC(NUTS(model), warmup, draws, num_chains=num_chains, progress_bar=progress_bar)
    data_args = {col: df_train[col].values for col in data_cols}
    mcmc.run(random.PRNGKey(16), **data_args, **fitting_args)

    # Test
    data_args = {col: df_test[col].values for col in data_cols}
    ppc = Predictive(model, num_samples=draws, parallel=False)
    ppc = ppc(
        random.PRNGKey(17),
        **data_args,
        **fitting_args
    )

    trace = az.from_numpyro(mcmc,
                            posterior_predictive=ppc,
                            # coords={"mouse": df_test['mouse_code']},
                            # dims={"y": ["mouse"]},
                            )

    return trace


def compare_models1(models: dict, df, data_cols: list, extra_args=None,
                    index_cols=('mouse_code', 'neuron_code', 'stim'),
                    draws=1000, warmup=500, num_chains=1, do_parallel=False, ):
    """
    compare_models(models={'Hier':bayes.Numpyro.model_hier,
                           'Hier+covariance':bayes.Numpyro.model_hier_covar,
                           'Twostep Exponential':bayes.TwoStep.model_twostep,
                           'Twostep Gamma':bayes.TwoStep.model_twostep,
                          },
                   data=[df,df,df_monster,df_monster],
                   extra_args=[{}, {}, {'prior':'Exponential'}, {'prior':'Gamma'}])
    :param models:
    :param data:
    :param extra_args:
    :param index_cols:
    :param draws:
    :param warmup:
    :param num_chains:
    :return:
    """

    def r2(trace):
        """
        R squared
        :param trace:
        """
        y_true = trace.observed_data["y"].values
        y_pred = trace.posterior_predictive.stack(sample=("chain", "draw"))["y"].values.T
        try:
            print(az.r2_score(y_true, y_pred))
        except (TypeError, ValueError) as e:
            print(e)

    if extra_args is None:
        extra_args = np.tile({}, len(models))
    numpyro.set_host_device_count(10)
    # Run
    if do_parallel:
        # cant use utils because zip
        traces = Parallel(n_jobs=14, verbose=3)(delayed(split_train_predict1)
                                                (model, df, data_cols, fitting_args, index_cols, draws, warmup,
                                                 num_chains,
                                                 progress_bar=False)
                                                for model, fitting_args in
                                                zip(models.values(), extra_args))
    else:
        traces = [split_train_predict1(model, df, data_cols, fitting_args, index_cols, draws, warmup, num_chains)
                  for model, fitting_args in zip(models.values(), extra_args)]

    # save to results
    traces_dict = {}  # initialize results
    for key, trace in zip(models.keys(), traces):
        traces_dict[key] = trace

    for trace_name in traces_dict.keys():

        trace = traces_dict[trace_name]
        # Print diagnostics and effect size
        print(f"n(Divergences) = {trace.sample_stats.diverging.sum(['chain', 'draw']).values}")
        try:
            try:
                slope = trace.posterior['v_mu'].sel({'v_mu_dim_0': 1}).mean(['chain']).values
            except Exception:
                slope = trace.posterior['b'].mean(['chain']).values
            print(f'Effect size={(slope.mean() / slope.std()).round(2)}  == {trace_name}')
        except Exception:
            pass

        # Plot PPC
        az.plot_ppc(trace,
                    flatten=[data_cols[2]],
                    # flatten_pp=data_cols[2],
                    mean=False,
                    # num_pp_samples=1000,
                    # kind='cumulative'
                    )
        plt.title(trace_name)
        plt.show()
        r2(trace)
        # Weird that r2=1
        # Waic
        try:
            print('======= WAIC (higher is better): =========')
            print(az.waic(trace, pointwise=True))
            print(az.waic(trace, var_name='y'))
        except TypeError:
            pass

    try:
        for trace_name in traces_dict.keys():
            trace = traces_dict[trace_name]
            # Print diagnostics and effect size
            print(f"n(Divergences) = {trace.sample_stats.diverging.sum(['chain', 'draw']).values}")
            try:
                slope = trace.posterior['v_mu'].sel({'v_mu_dim_0': 1}).mean(['chain']).values
            except Exception:
                slope = trace.posterior['b'].mean(['chain']).values
            print(f'Effect size={(slope.mean() / slope.std()).round(2)}  == {trace_name}')
    except Exception:
        pass

    try:
        model_compare = az.compare(traces_dict)  # , var_name='y')
        az.plot_compare(model_compare, textsize=12)
        print(model_compare)
        model_compare = az.compare(traces_dict)  # , var_name='y_isi')
        az.plot_compare(model_compare, textsize=12)
        print(model_compare)
    except Exception as e:
        print(e)
        model_compare = []

    return traces_dict, model_compare

import os
from itertools import product

import altair as alt
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bayes_window import BayesRegression, LMERegression
from bayes_window import BayesWindow
from bayes_window import models
from bayes_window import utils
from bayes_window.fitting import fit_numpyro
from bayes_window.generative_models import generate_fake_lfp
from jax import random
from joblib import Parallel, delayed
from numpyro.infer import Predictive
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

trans = LabelEncoder().fit_transform


def make_confusion_matrix(res, groups):
    df = []
    for _, this_res in res.groupby(list(groups)):
        this_res['score'] = this_res['score'].replace({'': None}).astype(float)
        this_res['true_slope'] = this_res['true_slope'] > 0
        this_res['score'] = this_res['score'] > 0

        cm = confusion_matrix(this_res['true_slope'],
                              this_res['score'],
                              labels=this_res['true_slope'].unique())
        cm = [y for i in cm for y in i]
        roll = list(product(np.unique(this_res['true_slope']), repeat=2))
        this_res = this_res.drop(['true_slope', 'score'], axis=1)
        for i in range(len(roll)):
            rez = {'actual': roll[i][0],
                   'predicted': roll[i][1],
                   'Occurences': cm[i],
                   }
            rez.update(this_res.iloc[0].to_dict())
            df.append(rez)
        # Remove raw scores to reduce confusion
    return pd.DataFrame.from_records(df)


def plot_confusion(df):
    # plot
    base = alt.Chart(df)
    heat = base.mark_rect().encode(
        x="predicted",
        y="actual",
        color='Occurences:O'
    ).properties(width=180, height=180)
    # Configure text
    # Doesnt work without mean; mean is meaningless without groupby
    # text = base.mark_text(baseline='middle').encode(
    #     text=alt.Text('mean(Occurences)', format=",.1f", ),
    #     x="predicted",
    #     y="actual",
    #     # color=alt.condition(
    #     #    alt.datum.Occurences > df['Occurences'].mean(),
    #     #    alt.value('black'),
    #     #    alt.value('white')
    #     # )
    # )
    return heat


def make_roc_auc(res, binary=True, groups=('method', 'y', 'randomness', 'n_trials')):
    """ Vary as function of true_slope """
    df = []
    for head, this_res in res.groupby(list(groups)):
        this_res['score'] = this_res['score'].replace({'': None}).astype(float)
        if binary:
            this_res['score'] = this_res['score'] > 0
        else:
            this_res = this_res.dropna(subset=['score'])  # drop nans to prevent errors
        fprs = []
        tprs = []
        slopes = []
        # Loop over true_slopes

        for ts in this_res[this_res['true_slope'] > 0]['true_slope'].unique():
            # Select all no-slopes and this slope:
            x = this_res[(this_res['true_slope'] == 0) |
                         (this_res['true_slope'] == ts)]
            # Binarize:
            x['true_slope'] = x['true_slope'] > 0
            fpr, tpr, _ = roc_curve(x['true_slope'], x['score'])
            # print(f"Yes for {head} {round(ts, 2)}: {fpr, tpr}")
            # print(f"{x['true_slope'].values},\n {x['score'].values}")
            fprs.extend(fpr)
            tprs.extend(tpr)
            slopes.extend(np.repeat(ts, len(fpr)))
        rocs = {'False positive rate': fprs,
                'True positive rate': tprs,
                'AUC': round(auc(fpr, tpr), 5),
                'true_slope': slopes
                }

        rocs = pd.DataFrame(rocs)

        # Remove raw scores to reduce confusion
        this_res = this_res.drop(['true_slope', 'score'], axis=1)
        for col in this_res.columns:
            rocs[col] = this_res[col].iloc[0]
        df.append(rocs)
    return pd.concat(df)


def plot_roc(df):
    roc = (alt.Chart(df).mark_line(size=2.6, opacity=.7).encode(
        x='False positive rate',
        y='mean(True positive rate)',
        color='method'
    ) + alt.Chart(df).mark_point(size=22.6, opacity=.7).encode(
        x='False positive rate',
        y='mean(True positive rate)',
        color='method'
    )).interactive().properties(width=150)
    bars = alt.Chart(df).mark_bar().encode(
        x='method',
        y='AUC',
        color='method'
    ).interactive()
    return bars, roc


def run_method(df, method='bw_student', y='Log power'):
    bw = BayesWindow(df, y=y, treatment='stim', group='mouse', add_data=True)
    if method[:2] == 'bw':
        BayesRegression(bw).fit(model=models.model_hierarchical, dist_y=method[3:], num_chains=1)
        return bw.data_and_posterior['lower interval'].iloc[0]
    elif method[:5] == 'anova':
        return LMERegression(bw).fit()  # Returns p-value

    elif method == 'mlm':
        posterior = LMERegression(bw).fit().data_and_posterior
        try:
            return posterior['lower interval'].iloc[0]
        except AttributeError:
            return posterior['lower interval']


def run_methods(methods, ys, true_slope, n_trials, randomness, parallel=False):
    df, df_monster, index_cols, _ = generate_fake_lfp(mouse_response_slope=true_slope,
                                                      n_mice=6,
                                                      n_trials=n_trials,
                                                      trial_baseline_randomness=randomness
                                                      )
    # df = df.rename({'Log power': 'Log_power'}, axis=1)
    # df_monster = df_monster.rename({'Log power': 'Log_power'}, axis=1)

    if parallel:
        res = Parallel(n_jobs=len(methods) * len(ys), verbose=0)(delayed(run_method)(df=df, method=method, y=y)
                                                                 for y, method in product(ys, methods))
    else:
        res = [run_method(df=df, method=method, y=y) for y, method in tqdm(product(ys, methods))]
    # Save result in dict
    return pd.DataFrame.from_records([{'method': method, 'y': y, 'score': mres, 'true_slope': true_slope,
                                       'n_trials': n_trials, 'randomness': randomness}
                                      for (method, y), mres in zip(product(methods, ys), res)])


def run_conditions(true_slopes=np.hstack([np.zeros(180), np.linspace(.03, 18, 140)]),
                   n_trials=range(8, 30, 7),
                   trial_baseline_randomness=(.2, .4, 1.8),
                   parallel=False,
                   methods=('anova', 'mlm', 'bw_gamma', 'bw_normal'),  # 'bw_student'
                   ys=('Log power',)):
    conditions = list(product(true_slopes, n_trials, trial_baseline_randomness))
    if parallel:
        res = Parallel(n_jobs=12)(delayed(run_methods)(methods, ys, true_slope, n_trials, randomness, parallel=False)
                                  for true_slope, n_trials, randomness in tqdm(conditions))
    else:
        res = [run_methods(methods, ys, true_slope, n_trials, randomness, parallel=False)
               for true_slope, n_trials, randomness in tqdm(conditions)]

    return pd.concat(res)


def split_train_predict(df, model, y, **kwargs):
    """

        y=df['con_coherence_magnitude_near'].values,
        treatment=trans(df['event'].values),
        condition=trans(df['condition_code'].values),
        group=trans(df['subject'].values),
    :return:
    """

    level_names = ['treatment', 'group']
    # Drop levels that were not requested
    level_names = [level_name for level_name in level_names if (level_name in kwargs.keys())]
    # Drop levels not present in data
    level_names = [level_name for level_name in level_names if kwargs[level_name] not in [None, [None]]]
    # Corresponding df labels:
    df_cols = [utils.level_to_data_column(level_name, kwargs) for level_name in level_names]

    if 'combined_condition' in df.columns:
        df_cols.extend(['combined_condition'])

    # split into training and test
    if len(df_cols) > 0:
        df_train, df_test = train_test_split(df, train_size=.5, test_size=.5, stratify=df[df_cols])
    else:
        df_train, df_test = train_test_split(df, train_size=.5, test_size=.5)
    model_args = {'y': df_test[y].values}
    model_args.update({level: trans(df_test[kwargs[level]]) for level in level_names})

    mcmc = fit_numpyro(model=model, **model_args, convert_to_arviz=False, num_chains=1, n_draws=2000)

    if mcmc.num_samples < 50:
        raise IndexError(f"Bug in numpyro? :{mcmc.num_samples, model_args}")

    ppc = Predictive(model, parallel=False, num_samples=2000, batch_ndims=2)
    ppc = ppc(random.PRNGKey(17), **model_args)

    if not ((ppc['y'].shape[0] == mcmc.num_chains and ppc['y'].shape[1] == mcmc.num_samples) |
            (ppc['y'].shape[0] == mcmc.num_chains * mcmc.num_samples)):
        raise IndexError(
            f"Bug in numpyro? :{ppc['y'].shape[0], mcmc.num_chains, ppc['y'].shape[1], mcmc.num_samples, ppc['y'].shape[0], mcmc.num_chains * mcmc.num_samples}")
    predictive = az.from_numpyro(mcmc,
                                 posterior_predictive=ppc,
                                 # coords={"mouse": df_test['mouse_code']},
                                 # dims={"y": ["mouse"]},
                                 )

    return predictive


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


def compare_models(df, models: dict,
                   extra_model_args: list = None,
                   parallel=False, plotose=False, **kwargs):
    """
    kwargs are forwarded to split_train_predict->fit_numpyro
    compare_models(models={'Hier':bayes.Numpyro.model_hier,
                           'Hier+covariance':bayes.Numpyro.model_hier_covar,
                           'Twostep Exponential':bayes.TwoStep.model_twostep,
                           'Twostep Gamma':bayes.TwoStep.model_twostep,
                          },
                   data=[df,df,df_monster,df_monster],
                   extra_args=[{}, {}, {'prior':'Exponential'}, {'prior':'Gamma'}])
    """
    # TODO save all model args in BayesWindow in self
    # Calculate
    extra_model_args = extra_model_args or np.tile({}, len(models))
    if parallel:
        traces = Parallel(n_jobs=min(os.cpu_count(),
                                     len(models)))(delayed(split_train_predict)
                                                   (df, model, num_chains=1, **kwargs, **extra_model_arg)
                                                   for model, extra_model_arg in zip(models.values(), extra_model_args))
    else:
        traces = [split_train_predict(df, model, **kwargs, **extra_model_arg)
                  for model, extra_model_arg in zip(models.values(), extra_model_args)]

    # save tp dict
    traces_dict = {}  # initialize results
    for key, trace in zip(models.keys(), traces):
        traces_dict[key] = trace

    # Plot
    if plotose:
        for trace_name, trace in traces_dict.items():

            # Plot PPC
            az.plot_ppc(trace,
                        # flatten=[treatment],
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

    model_compare = az.compare(traces_dict)  # , var_name='y')
    az.plot_compare(model_compare, textsize=12, show=True)
    return model_compare

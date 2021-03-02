from itertools import product

import altair as alt
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bayes_window import workflow, models
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
    text = base.mark_text(baseline='middle').encode(
        text=alt.Text('mean(Occurences)', format=",.1f", ),
        x="predicted",
        y="actual",
        # color=alt.condition(
        #    alt.datum.Occurences > df['Occurences'].mean(),
        #    alt.value('black'),
        #    alt.value('white')
        # )
    )
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


def make_roc_auc_old(res, binary=True, groups=('method', 'y', 'randomness', 'n_trials')):
    # Make ROC and AUC
    df = []
    for _, this_res in res.groupby(list(groups)):
        this_res['score'] = this_res['score'].replace({'': None}).astype(float)
        this_res['true_slope'] = this_res['true_slope'] > 0
        if binary:
            this_res['score'] = this_res['score'] > 0
        else:
            this_res = this_res.dropna(subset=['score'])  # drop nans to prevent errors
        fpr, tpr, _ = roc_curve(this_res['true_slope'], this_res['score'])

        # Remove raw scores to reduce confusion
        this_res = this_res.drop(['true_slope', 'score'], axis=1)

        # Only keep the number of rows that will be useful for keeping ROC
        this_res = this_res.reset_index(drop=True).iloc[:len(fpr)]

        this_res['False positive rate'] = fpr
        this_res['True positive rate'] = tpr
        this_res['AUC'] = round(auc(fpr, tpr), 5)

        df.append(this_res)
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
    bw = workflow.BayesWindow(df, y=y, treatment='stim', group='mouse')
    if method[:2] == 'bw':
        bw.fit_slopes(model=models.model_hier_stim_one_codition,
                      dist_y=method[3:],
                      add_data=False)
        return bw.data_and_posterior['lower interval'].iloc[0]
    elif method[:5] == 'anova':
        return bw.fit_anova()  # Returns p-value

    elif method == 'mlm':
        posterior = bw.fit_lme(add_data=False).posterior
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
                   methods=('bw_lognormal', 'bw_normal', 'mlm', 'anova',),  # 'bw_student'
                   ys=('Log power', 'Power',)):
    conditions = list(product(true_slopes, n_trials, trial_baseline_randomness))
    if parallel:
        res = Parallel(n_jobs=12)(delayed(run_methods)(methods, ys, true_slope, n_trials, randomness, parallel=False)
                                  for true_slope, n_trials, randomness in tqdm(conditions))
    else:
        res = [run_methods(methods, ys, true_slope, n_trials, randomness, parallel=False)
               for true_slope, n_trials, randomness in tqdm(conditions)]

    return pd.concat(res)


# def split_train_predict(model, df, data_cols, fitting_args=None, index_cols=('mouse_code', 'neuron_code', 'stim'),
#                         draws=1000, warmup=500, num_chains=1, progress_bar=True):
def split_train_predict(df, model, fit_method, y, treatment, condition, group):
    """

        y=df['con_coherence_magnitude_near'].values,
        treatment=trans(df['event'].values),
        condition=trans(df['condition_code'].values),
        subject=trans(df['subject'].values),
    :return:
    """
    from bayes_window import utils
    from bayes_window import BayesWindow
    assert hasattr(BayesWindow, fit_method)
    fitting_args = {}
    condition = condition if type(condition) == list else [condition]
    if condition[0]:
        assert condition[0] in df.columns
    levels = utils.parse_levels(treatment, condition, group)

    # split into training and test
    df_train, df_test = train_test_split(df, train_size=.5, test_size=.5,
                                         stratify=df[levels])

    window = BayesWindow(df_train, y, treatment=treatment, condition=condition, group=group)
    window = getattr(window, fit_method)(add_data=False, model=model)
    # eg window.fit_slopes(add_data=False, model=model)

    data_args = {col: df_test[col].values for col in levels}
    ppc = Predictive(model, parallel=False)
    ppc = ppc(
        random.PRNGKey(17),
        **data_args,
        **fitting_args
    )

    trace = az.from_numpyro(window.trace,
                            posterior_predictive=ppc,
                            # coords={"mouse": df_test['mouse_code']},
                            # dims={"y": ["mouse"]},
                            )

    return trace
    # # fit
    #
    # fitting_args = fitting_args or {}
    # mcmc = MCMC(NUTS(model), warmup, draws, num_chains=num_chains, progress_bar=progress_bar)
    # data_args = {col: df_train[col].values for col in data_cols}
    # mcmc.run(random.PRNGKey(16), **data_args, **fitting_args)
    #
    # # Test
    # data_args = {col: df_test[col].values for col in data_cols}
    # ppc = Predictive(model, num_samples=draws, parallel=False)
    # ppc = ppc(
    #     random.PRNGKey(17),
    #     **data_args,
    #     **fitting_args
    # )
    #
    # trace = az.from_numpyro(mcmc,
    #                         posterior_predictive=ppc,
    #                         # coords={"mouse": df_test['mouse_code']},
    #                         # dims={"y": ["mouse"]},
    #                         )
    #
    # return trace


# def compare_models(models_to_compare: dict, df, data_cols: list, extra_args=None,
#                    index_cols=('mouse_code', 'neuron_code', 'stim'),
#                    draws=1000, warmup=500, num_chains=1, do_parallel=False, ):
def compare_models(df, models: dict, fit_method, y, treatment, condition=None, group=None, parallel=False):
    """
    compare_models(models={'Hier':bayes.Numpyro.model_hier,
                           'Hier+covariance':bayes.Numpyro.model_hier_covar,
                           'Twostep Exponential':bayes.TwoStep.model_twostep,
                           'Twostep Gamma':bayes.TwoStep.model_twostep,
                          },
                   data=[df,df,df_monster,df_monster],
                   extra_args=[{}, {}, {'prior':'Exponential'}, {'prior':'Gamma'}])
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

    # if extra_args is None:
    #     extra_args = np.tile({}, len(models))
    # numpyro.set_host_device_count(10)
    # Run
    if parallel:
        # cant use utils because zip
        pass
        # traces = Parallel(n_jobs=14, verbose=3)(delayed(split_train_predict)
        #                                         (model, df, data_cols, fitting_args, index_cols, draws, warmup,
        #                                          num_chains,
        #                                          progress_bar=False)
        #                                         for model, fitting_args in
        #                                         zip(models.values(), extra_args))
    else:
        # traces = [split_train_predict(model, df, data_cols, fitting_args, index_cols, draws, warmup, num_chains)
        #           for model, fitting_args in zip(models_to_compare.values(), extra_args)]
        traces = [split_train_predict(df, model, fit_method, y, treatment, condition, group)
                  for model in models.values()]

    # save to results
    traces_dict = {}  # initialize results
    for key, trace in zip(models.keys(), traces):
        traces_dict[key] = trace

    for trace_name in traces_dict.keys():

        # Plot PPC
        az.plot_ppc(trace,
                    flatten=[treatment],
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

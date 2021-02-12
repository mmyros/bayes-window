from sklearn.preprocessing import LabelEncoder

from bayes_window import models
from bayes_window.fake_spikes import generate_fake_spikes
from bayes_window.fitting import fit_numpyro
from bayes_window.utils import add_data_to_posterior
from bayes_window.visualization import plot_data_and_posterior, plot_posterior

le = LabelEncoder()


def estimate_slope(df, y='isi', levels=('stim', 'mouse', 'neuron'), hold_for_facet=True):
    # df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
    #                                                                 n_neurons=8,
    #                                                                 n_mice=4,
    #                                                                 dur=7, )
    # By convention, top condition is first in list of levels:
    top_condition = levels[0]
    # Transform conditions to integers as required by numpyro:
    key = dict()
    for level in levels:
        df[level] = le.fit_transform(df[level])
        # Keep key for later use
        key[level] = dict(zip(range(len(le.classes_)), le.classes_))

    # Estimate model
    trace = fit_numpyro(y=df[y].values,
                        stim_on=df[top_condition].values,
                        treat=df[levels[2]].values,
                        subject=df[levels[1]].values,
                        progress_bar=True,
                        model=models.model_hier_normal_stim,
                        n_draws=100, num_chains=1)

    # Add data back
    df_both = add_data_to_posterior(df,
                                    trace=trace,
                                    y=y,
                                    index_cols=levels[:3],
                                    condition_name=top_condition,
                                    b_name='b_stim_per_condition',  # for posterior
                                    group_name=levels[2]  # for posterior
                                    )

    [df_both[col].replace(key[col], inplace=True) for col in key.keys() if not col == top_condition]
    # Plot data and posterior
    chart = plot_data_and_posterior(df_both,
                                    y=f'{y} diff',
                                    x=levels[2],
                                    color=levels[1],
                                    title=y,
                                    hold_for_facet=hold_for_facet)

    return chart


def estimate_posteriors(df, y='isi', levels=('stim', 'mouse', 'neuron'),model=models.model_single_lognormal,
                        add_data=True):
    if 0:
        df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                        n_neurons=8,
                                                                        n_mice=4,
                                                                        dur=7, )
    df['combined_condition'] = df[levels[0]].astype('str')
    for level in levels[1:]:
        df['combined_condition'] += df[level].astype('str')

    # Transform conditions to integers as required by numpyro:
    df['combined_condition'] = le.fit_transform(df['combined_condition'])
    # Keep key for later use
    # key = dict(zip(range(len(le.classes_)), le.classes_))

    # Estimate model
    trace = fit_numpyro(y=df[y].values,
                        treat=df['combined_condition'].values,
                        progress_bar=True,
                        model=model,
                        n_draws=100, num_chains=1)
    # Add data back
    df_both = add_data_to_posterior(df,
                                    trace=trace,
                                    y=y,
                                    index_cols=levels[:3],
                                    condition_name=levels[0],
                                    b_name='mu_per_condition',  # for posterior
                                    group_name='combined_condition',
                                    do_mean_over_trials=False,
                                    do_make_change=False
                                    )
    # Plot data and posterior
    chart = plot_posterior(df=df_both,
                           x=levels[0],
                           do_make_change=False,
                           add_data=add_data,
                           ).properties(width=20)#.facet(row=levels[1],
                                                                     #       column=levels[2])
    # chart = plot_data_and_posterior(df_both,
    #                                 y=f'{y} diff',
    #                                 x=levels[2],
    #                                 color=levels[1],
    #                                 title=y,
    #                                 hold_for_facet=hold_for_facet)

    return chart,df_both

import altair as alt
from sklearn.preprocessing import LabelEncoder

from . import utils

trans = LabelEncoder().fit_transform


def plot_data_and_posterior(df, y='Coherence diff', title='coherence', x='Stim phase', color='Subject',
                            hold_for_facet=True):
    assert (x in df) | (x[:-2] in df)
    assert color in df
    assert y in df

    # Plot data:
    c1 = alt.Chart().mark_line(fill=None, opacity=.5, size=6).encode(
        x=x,
        color=f'{color}:N',
        y=y
    )
    c2 = plot_posterior(df, title=title, x=x, add_data=False)
    chart = alt.layer(c1, c2, data=df)
    if not hold_for_facet:
        chart = chart.resolve_scale(y='independent')  # only works if no facet
    return chart


def plot_posterior(df, title='', x='Stim phase', add_data=True, do_make_change=True):
    assert 'Bayes condition CI0' in df
    assert 'Bayes condition CI1' in df
    assert (x in df.columns) | (x[:-2] in df.columns), print(x, df.columns)
    alt.themes.enable('default')
    if do_make_change:
        # Make the zero line
        df['zero'] = 0
        rule = alt.Chart().mark_rule(color='black', size=.1).encode(y='zero')
        title = 'Δ ' + title
    else:
        rule = None

    # Make bayes
    points = alt.Chart().mark_point(filled=True, color='black').encode(
        y=alt.Y('Bayes condition mean:Q', scale=alt.Scale(zero=False)),
        x=x
    )

    line = alt.Chart().mark_line(color='black').encode(
        y=alt.Y('Bayes condition mean:Q', scale=alt.Scale(zero=False)),
        x=x
    )

    error_bars = alt.Chart().mark_rule().encode(
        x=x,
        y=alt.Y('Bayes condition CI0:Q', title=title, scale=alt.Scale(zero=False)),
        y2='Bayes condition CI1:Q',
    )
    if do_make_change:
        c2 = rule + points + line + error_bars
    else:
        c2 = points + line + error_bars
    if add_data:
        c2 = alt.layer(c2, data=df)
    return c2


def plot_posterior_altair(trace, df,
                          b_name='b_stim_per_condition', x='Stim phase',
                          title='', group_name='condition_code'):
    # Convert to dataframe and fill in original conditions:
    df = utils.trace2df(trace, df, b_name, group_name=group_name)

    chart = plot_posterior(df, title=title, x=x, add_data=True)

    return chart


def fake_spikes_explore(df, df_monster, index_cols):
    import altair as alt

    # mean firing rate per trial per mouse
    width = 50
    fig_trials = alt.Chart(df).mark_line(fill=None, ).encode(
        x=alt.X('stim'),
        y=alt.Y('log_firing_rate', scale=alt.Scale(zero=False)),
        color='neuron:N',
        detail='i_trial:Q',
        opacity=alt.value(1),
        size=alt.value(.9),
        facet='mouse'
    ).properties(
        title='All trials and neurons',
        # columns=5,
        width=width,
        height=300
    )

    # mean firing rate per trial per mouse (select first and last mouse)
    alt.data_transformers.disable_max_rows()
    fig_select = alt.Chart(df[(df['neuron'] == '0') |
                              (df['neuron'] == str(df['neuron'].astype(int).max().astype(int) - 1))]).mark_line(
        fill=None, ).encode(
        x=alt.X('stim'),
        y=alt.Y('log_firing_rate', scale=alt.Scale(zero=False)),
        color='neuron:N',
        detail='i_trial:Q',
        opacity=alt.value(1),
        size=alt.value(2),
        facet='mouse'
    ).properties(
        title='Slow neurons are more responsive',
        # columns=5,
        width=width,
        height=300
    )

    # mean firing rate per mouse
    fig_neurons = alt.Chart(df).mark_line(fill=None, ).encode(
        x=alt.X('stim'),
        y=alt.Y('log_firing_rate', scale=alt.Scale(zero=False)),
        color='neuron:N',
        opacity=alt.value(1),
        size=alt.value(2),
        facet='mouse'
    ).properties(
        title='All neurons',
        # columns=5,
        width=width,
        height=300
    )
    # mean firing rate per mouse
    fig_mice = alt.Chart(df[df['neuron'] == '0']).mark_line(fill=None, ).encode(
        x=alt.X('stim'),
        y=alt.Y('log_firing_rate', scale=alt.Scale(zero=False)),
        opacity=alt.value(1),
        size=alt.value(3),
        facet='mouse:N'
    ).properties(
        title='Mice sorted by response',
        # columns=5,
        width=width,
        height=300
    )

    # Monster-level ISI
    df_isi = df_monster[
        (
            (df_monster['neuron'] == 0) |
            (df_monster['neuron'] == str(df_monster['neuron'].astype(int).max()))
        ) &
        # (df_monster['mouse']=='m0bayes') |
        (df_monster['mouse'] == 'm9bayes')
        ]
    fig_isi = alt.Chart(df_isi).mark_tick(opacity=.2).encode(
        x=alt.Y('stim'),
        y=alt.X('log_1/isi', scale=alt.Scale(zero=False), ),
        color='neuron:N',
        detail='i_spike:Q',  # Crucial: Q!
    ).properties(
        title=['Multiple trials per mouse', 'many spikes'],
        width=width,
        height=500
    )
    fig_overlay = alt.Chart(df_isi).mark_line(fill=None, ).encode(
        x=alt.X('stim'),
        y=alt.Y('log_firing_rate', scale=alt.Scale(zero=False)),
        color='neuron:N',
        detail='i_trial:Q',
        size=alt.value(2),
    )

    data_fold_change, y = utils.make_fold_change(df, y='log_firing_rate', index_cols=index_cols, condition_name='stim',
                                                 conditions=(0, 1))
    box = alt.Chart(data=data_fold_change).mark_boxplot().encode(y=y).encode(
        x=alt.X('neuron:N', ),
        y=alt.Y(y, scale=alt.Scale(zero=True)),
    ).properties(width=width, height=240).facet(
        # row='mouse:N',
        column=alt.Column('mouse'))  # .resolve_scale(y='independent')

    bar = (alt.Chart(data=data_fold_change).mark_bar().encode(y=alt.Y(y, aggregate='mean')) +
           alt.Chart(data=data_fold_change).mark_errorbar().encode(y=alt.Y(y, aggregate='stderr'))).encode(
        x=alt.X('neuron:N', ),
        y=alt.Y(y),
    ).properties(width=width * 2, height=240).facet(
        # row='Inversion:N',
        column=alt.Column('mouse'))  # .resolve_scale(y='independent')

    bar_combined = (alt.Chart(data=data_fold_change).mark_bar().encode(y=alt.Y(y, aggregate='mean')) +
                    alt.Chart(data=data_fold_change).mark_errorbar().encode(y=alt.Y(y, aggregate='stderr'))).encode(
        x=alt.X('neuron:N', ),
        y=alt.Y(y),
    ).properties(width=width, height=240)  # .facet(
    # row='Inversion:N',
    # column=alt.Column('mouse'))#.resolve_scale(y='independent')

    # Monster-level ISI
    df_raster = df_monster[
        # (
        #    (df_monster['neuron']=='0')       |
        #    (df_monster['neuron']==str(n_neurons-1))
        # )
        # &
        # (df_monster['mouse']=='m0bayes') |
        (df_monster['mouse'] == 'm9bayes')
    ]
    fig_raster = alt.Chart(df_raster).mark_tick(thickness=.8).encode(
        y=alt.Y('neuron'),
        x=alt.X('spike_time', scale=alt.Scale(zero=False), ),
        # color='neuron:N',
        detail='i_spike:Q',
    ).properties(
        # title=['Multiple trials per mouse','many spikes'],
        width=800,
        height=140
    ).facet(row='stim')

    return fig_mice, fig_select, fig_neurons, fig_trials, fig_isi + fig_overlay, bar, box, fig_raster, bar_combined

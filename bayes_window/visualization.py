from pdb import set_trace
import altair as alt
from sklearn.preprocessing import LabelEncoder
from . import utils
trans = LabelEncoder().fit_transform
# reload(utils)

def plot_posterior_altair(trace, df, b_name='b_stim_per_condition', plot_x='Stim phase:N',
                          row=None, column=None, title='', width=300, group_name='condition_code'):
    # Convert to dataframe and fill in original conditions:
    df_bayes = utils.trace2df(trace, df, b_name, group_name=group_name)

    # Keep only theta:
    # df_bayes = df_bayes[df_bayes['Stim frequency'] < 30]
    alt.themes.enable('default')

    # Make the zero line
    df_bayes['zero'] = 0
    rule = alt.Chart(df_bayes).mark_rule(color='black', size=.1).encode(y='zero')

    # Make plots
    points = alt.Chart(df_bayes).mark_point(filled=True, color='black').encode(
        y=alt.Y('Bayes condition mean:Q', scale=alt.Scale(zero=False)),
        x=plot_x
    )

    error_bars = points.mark_rule().encode(
        x=plot_x,
        y=alt.Y('Bayes condition ci0:Q', title='Δ ' + title, scale=alt.Scale(zero=False)),
        y2='Bayes condition ci1:Q',
    )

    chart = (rule + points + error_bars).properties(
        width=width,
        # height=250
    )
    if (column is not None) and (row is not None):
        chart = chart.facet(row=row, column=column)
    elif column is not None:
        chart = chart.facet(column=column)
    elif row is not None:
        chart = chart.facet(row=row)
    chart.title = title

    return chart

    # Can add data on same plot, though would need to make slopes:
    # df.rename({'Change coherence mean near':ycoh},axis=1,inplace=True)
    # alt.Chart(utils.DataJoint.humanize(df)).mark_boxplot(opacity=.95,size=10,extent=999).encode(y=ycoh,x='Stim phase:N')



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
        detail='i_spike:Q', # Crucial: Q!
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

    def make_fold_change(df, y='log_firing_rate'):
        # Make multiindex
        mdf = df.set_index(list(index_cols - {'i_spike'})).copy()
        mdf.xs(0, level='stim') - mdf.xs(1, level='stim')

        # Subtract/divide
        data = (mdf.xs(0, level='stim') - mdf.xs(1, level='stim')).reset_index()
        import copy
        y0 = copy.copy(y)
        y1 = f'{y.split(" ")[0]} diff'
        data.rename({y: y1}, axis=1, inplace=True)
        y = y1
        # if y0=='log_firing_rate':
        #     # Transform back to non-log
        #    data[y] = 10** data[y]
        #    y2='fold_change_in_log_space'
        #    data.rename({y:y2},axis=1,inplace=True)
        #    y=y2

        return data, y

    data_fold_change, y = make_fold_change(df  # [
                                           #    (
                                           #        (df['neuron']=='0')       |
                                           #        (df['neuron']==str(n_neurons-1))
                                           #    ) &
                                           #    (
                                           #        (df['mouse']=='m0bayes') |
                                           #        (df['mouse']=='m9bayes')
                                           #    )]
                                           ,
                                           y='log_firing_rate')
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

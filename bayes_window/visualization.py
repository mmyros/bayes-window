import altair as alt
import numpy as np
from sklearn.preprocessing import LabelEncoder

from . import utils

trans = LabelEncoder().fit_transform


# class AltairHack(alt.Chart):
#     def __init__(self):
#         super().__init__()
#         print(self.data)


def facet(base_chart,
          column=None,
          row=None,
          width=80,
          height=150,
          ):
    alt.themes.enable('vox')
    if column is None and row is None:
        raise RuntimeError('Need either column, or row, or both!')

    if column:
        assert column in base_chart.data.columns
    if row:
        assert row in base_chart.data.columns

    def concat_charts(subdata, groupby_name, row_name='', row_val='', how='hconcat'):
        charts = []
        # print(subdata)
        # print(column)
        # TODO funky logic with column and row
        for icol, column_val in enumerate(subdata[groupby_name].unique()):
            title = f"{groupby_name} {column_val} {row_name} {row_val}"  # if icol == 0 else ''
            charts.append(alt.layer(
                base_chart, title=title, data=base_chart.data
            ).transform_filter(
                alt.datum[groupby_name] == column_val
            ).resolve_scale(
                y='independent'
            ).properties(
                width=width, height=height))
        if how == 'hconcat':
            return alt.hconcat(*charts)
        else:
            return alt.vconcat(*charts)

    if row is None:
        chart = concat_charts(base_chart.data, groupby_name=column,
                              how='hconcat')
    elif column is None:
        chart = concat_charts(base_chart.data, groupby_name=row,
                              how='vconcat')
    else:
        chart = alt.vconcat(*[concat_charts(subdata, groupby_name=column,
                                            row_name=row, row_val=val, how='hconcat')
                              for val, subdata in base_chart.data.groupby(row)])
    chart = chart.configure_view(
        stroke=None
    )
    return chart


def plot_data(df=None, x=None, y=None, color=None, add_box=True, base_chart=None):
    assert (df is not None) or (base_chart is not None)
    if x[-2] != ':':
        x = f'{x}:O'
    if color != ':':
        color = f'{color}:N'
    # Plot data:
    base = alt.Chart(df) or base_chart
    chart = base.mark_line(fill=None, opacity=.5, size=3).encode(
        x=x,
        color=f'{color}',
        y=f'{y}:Q'
    )
    if add_box:
        # Shift x axis for box so that it doesnt overlap:
        # df['x_box'] = df[x[:-2]] + .01
        chart += base.mark_boxplot(opacity=.3, size=12, color='black').encode(
            x=x,
            y=f'{y}:Q'
        )
    return chart


def plot_data_and_posterior(df, y='Coherence diff', title='coherence', x='Stim phase', color='Subject',
                            hold_for_facet=True, add_box=True, **kwargs):
    # Keep kwargs!
    assert (x in df) | (x[:-2] in df)
    assert color in df
    assert y in df.columns, f'{y} is not in {df.columns}'

    chart_d = plot_data(df=df, x=x, y=y, color=color, add_box=add_box, base_chart=alt.Chart(df))
    chart_p = plot_posterior(df, title=title, x=x, base_chart=alt.Chart(df))
    chart = chart_d + chart_p

    if not hold_for_facet:
        chart = chart.resolve_scale(y='independent')  # only works if no facet
    return chart


def plot_posterior(df=None, title='', x='Stim phase', do_make_change=True, base_chart=None, **kwargs):
    if df is not None and not base_chart:
        assert 'Bayes condition CI0' in df
        assert 'Bayes condition CI1' in df
    assert (x in df.columns) | (x[:-2] in df.columns), print(x, df.columns)
    assert (df is not None) or (base_chart is not None)
    if x[-2] != ':':
        x = f'{x}:O'  # Ordinal
    # alt.themes.enable('vox')
    alt.themes.enable('default')
    base_chart = alt.Chart(data=df) or base_chart

    # line
    chart = base_chart.mark_line(point=True, color='black').encode(
        y=alt.Y('Bayes condition mean:Q', impute=alt.ImputeParams(value='value')),
        x=x
    )

    # Axis limits
    scale = alt.Scale(zero=False,
                      domain=[float(base_chart.data['Bayes condition CI0'].min()),
                              float(base_chart.data['Bayes condition CI1'].max())])

    # Make the zero line
    if do_make_change:
        df['zero'] = 0
        chart += base_chart.mark_rule(color='black', size=.1, opacity=.4).encode(y='zero')
        title = f'Δ {title}'

    # error_bars
    chart += base_chart.mark_rule().encode(
        x=x,
        y=alt.Y('Bayes condition CI0:Q',
                title=title, scale=scale),
        y2='Bayes condition CI1:Q',
    )

    return chart


def plot_posterior_altair(trace, df,
                          b_name='b_stim_per_condition', x='Stim phase',
                          title='', group_name='condition_code'):
    # Convert to dataframe and fill in original conditions:
    df = utils.trace2df(trace, df, b_name, group_name=group_name)

    chart = plot_posterior(df, title=title, x=x)

    return chart


def fake_spikes_explore(df, df_monster, index_cols):
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
            (df_monster['neuron'] == '0') |
            (df_monster['neuron'] == str(df_monster['neuron'].astype(int).max()))
        ) &
        # (df_monster['mouse']=='m0bayes') |
        (df_monster['mouse'] == f'm{df_monster["mouse_code"].astype(int).max()}bayes')
        ]
    fig_isi = alt.Chart(df_isi).mark_tick(opacity=.2).encode(
        x=alt.Y('stim'),
        y=alt.X('log_1/isi', scale=alt.Scale(zero=False), ),
        color='neuron:N',
        detail='i_spike:Q',  # Crucial: Q!
    ).properties(
        title=['Multiple trials per mouse', 'many spikes'],
        height=500
    )
    fig_overlay = alt.Chart(df_isi).mark_line(fill=None, ).encode(
        x=alt.X('stim'),
        y=alt.Y('log_firing_rate', scale=alt.Scale(zero=False)),
        color='neuron:N',
        detail='i_trial:Q',
        size=alt.value(2),
    )

    data_fold_change, y = utils.make_fold_change(df, y='log_firing_rate', index_cols=index_cols,
                                                 condition_name='stim',
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


def plot_data_slope_trials(x,
                           y,
                           color,
                           detail,
                           base_chart=None,
                           df=None,
                           **kwargs):
    assert (df is not None) or (base_chart is not None)
    if x[-2] != ':':
        x = f'{x}:O'  # Ordinal
    base_chart = alt.Chart(df) or base_chart

    # mean firing rate per trial per mouse
    fig_trials = base_chart.mark_line(fill=None).encode(
        x=alt.X(x),
        y=alt.Y(y, scale=alt.Scale(zero=False,
                                   # domain=[df[y].min(), df[y].max()])),
                                   domain=list(np.quantile(base_chart.data[y], [.05, .95])),
                                   clamp=True)),
        color=color,
        detail=detail,
        opacity=alt.value(.2),
        size=alt.value(.9),
        # **kwargs
    )
    return fig_trials

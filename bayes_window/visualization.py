import altair as alt
import numpy as np
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform


def facet(base_chart,
          column=None,
          row=None,
          width=80,
          height=150,
          ):
    alt.themes.enable('vox')
    if column is None and row is None:
        raise RuntimeError('Need either column, or row, or both!')
    assert base_chart.data is not None
    if column:
        assert column in base_chart.data.columns, f'{column} is not in {base_chart.data.columns}'
    if row:
        assert row in base_chart.data.columns, f'{row} is not in {base_chart.data.columns}'

    def concat_charts(subdata, groupby_name, row_name='', row_val='', how='hconcat'):
        charts = []
        for i_group, group_val in enumerate(subdata[groupby_name].unique()):
            title = f"{groupby_name} {group_val} {row_name} {row_val}"  # if i_group == 0 else ''
            charts.append(alt.layer(
                base_chart, title=title, data=base_chart.data
            ).transform_filter(
                alt.datum[groupby_name] == group_val
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


def plot_data(df=None, x=None, y=None, color=None, add_box=True, base_chart=None, **kwargs):
    color = color or ':O'
    assert (df is not None) or (base_chart is not None)
    if (x == '') or (x[-2] != ':'):
        x = f'{x}:O'
    # Plot data:
    base = base_chart or alt.Chart(df)
    if (color[-2] != ':'):
        color = f'{color}:N'
    chart = base.mark_line(fill=None, opacity=.5, size=3).encode(
        x=x,
        color=f'{color}',
        y=alt.Y(f'mean({y})', scale=alt.Scale(zero=False))
    )
    if add_box:
        # Shift x axis for box so that it doesnt overlap:
        # df['x_box'] = df[x[:-2]] + .01
        chart += base.mark_boxplot(opacity=.3, size=12, color='black').encode(
            x=x,
            y=alt.Y(f'{y}:Q', scale=alt.Scale(zero=False))
        )
    return chart


# from altair.vegalite.v4.api import Undefined
def plot_posterior(df=None, title='', x=':O', do_make_change=True, base_chart=None, **kwargs):
    assert (df is not None) or (base_chart is not None)
    data = base_chart.data if df is None else df
    if x[-2] != ':':
        x = f'{x}:O'  # Ordinal
    assert 'higher interval' in data.columns
    assert 'lower interval' in data.columns
    assert 'mean interval' in data.columns
    # alt.themes.enable('vox')
    alt.themes.enable('default')
    base_chart = base_chart or alt.Chart(data=data)

    # line
    chart = base_chart.mark_line(clip=True, point=True, color='black').encode(
        y=alt.Y('mean interval:Q', impute=alt.ImputeParams(value='value')),
        x=x,
    )
    do_make_change = do_make_change!=False

    # Axis limits
    minmax=[float(data['lower interval'].min()), 0,
                              float(data['higher interval'].max())]
    scale = alt.Scale(zero=do_make_change,  # Any string or True
                      domain=[min(minmax), max(minmax)])

    # Make the zero line
    if do_make_change:
        base_chart.data['zero'] = 0
        chart += base_chart.mark_rule(color='black', size=.1, opacity=.6).encode(y='zero')
        title = f'Δ {title}'

    # error_bars
    chart += base_chart.mark_rule().encode(
        x=x,
        y=alt.Y('lower interval:Q',
                title=title, scale=scale),
        y2='higher interval:Q',
    )

    return chart


def plot_data_slope_trials(x,
                           y,
                           detail,
                           color=None,
                           base_chart=None,
                           df=None,
                           **kwargs):
    assert (df is not None) or (base_chart is not None)
    color = color or ':O'
    if x[-2] != ':':
        x = f'{x}:O'  # Ordinal
    base_chart = base_chart or alt.Chart(df)

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

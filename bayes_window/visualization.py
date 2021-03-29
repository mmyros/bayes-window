import altair as alt
import numpy as np
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform


def facet(base_chart,
          column=None,
          row=None,
          width=80,
          height=150,
          finalize=False
          ):
    if column is None and row is None:
        raise RuntimeError('Need either column, or row, or both!')
    assert base_chart.data is not None
    if column:
        assert column in base_chart.data.columns, f'{column} is not in {base_chart.data.columns}'
        # sanitize a little:
        base_chart.data[column] = base_chart.data[column].astype(str)
    if row:
        assert row in base_chart.data.columns, f'{row} is not in {base_chart.data.columns}'
        # sanitize a little:
        base_chart.data[row] = base_chart.data[row].astype(str)

    def concat_charts(subdata, groupby_name, row_name='', row_val='', how='hconcat'):

        charts = []
        for i_group, group_val in enumerate(subdata[groupby_name].drop_duplicates().sort_values()):
            title = f"{groupby_name} {group_val} {row_name} {row_val}"  # if i_group == 0 else ''
            charts.append(alt.layer(
                base_chart,
                title=title,
                data=base_chart.data
            ).transform_filter(
                alt.datum[groupby_name] == group_val
            ).resolve_scale(
                y='independent'
            ).properties(width=width, height=height))
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
    if finalize:  # TODO is there a way to do this in theme instead?
        chart = chart.configure_view(
            stroke=None
        )
    return chart


def line_with_highlight(base, x, y, color, detail, highlight=True):
    # Highlight doesnt work with overlays, but at least we get visible dots in legend
    if highlight:
        highlight = alt.selection(type='single', on='mouseover', fields=[color], nearest=True)
        size = alt.condition(~highlight, alt.value(1), alt.value(3))
    else:
        size = alt.value(1.)

    lines = base.mark_line(clip=True, fill=None, opacity=.6).encode(
        size=size,
        x=x,
        color=f'{color}',
        y=alt.Y(f'mean({y})',
                axis=alt.Axis(orient='right'),
                scale=alt.Scale(zero=False,
                                domain=list(np.quantile(base.data[y], [.05, .95])))),
        detail=detail
    )
    points = base.mark_circle(clip=True, opacity=0, filled=True).encode(

        x=x,
        color=f'{color}',
        y=alt.Y(f'mean({y})',
                axis=alt.Axis(title='', orient='right'),
                scale=alt.Scale(zero=False,
                                domain=list(np.quantile(base.data[y], [.05, .95])))),
        detail=detail
    )
    if highlight:
        points.add_selection(
            highlight
        )
    return lines, points


def plot_data(df=None, x=None, y=None, color=None, add_box=True, base_chart=None, detail=':O', highlight=False,
              **kwargs):
    assert (df is not None) or (base_chart is not None)
    if (x == '') or (x[-2] != ':'):
        x = f'{x}:O'
    if color is None:
        color = ':N'
    if color[-2] != ':':
        color = f'{color}:N'
    charts = []

    # Plot data:
    base = base_chart or alt.Chart(df)
    y_domain = list(np.quantile(base.data[y], [.05, .95]))
    if (x != ':O') and (len(base.data[x[:-2]].unique()) > 1):
        charts.extend(line_with_highlight(base, x, y, color, detail, highlight=highlight))
        # charts.append(base.mark_line(clip=True, fill=None, opacity=.6, size=.5).encode(
        #     x=x,
        #     color=f'{color}',
        #     detail=detail,
        #     y=alt.Y(f'mean({y})',
        #             scale=alt.Scale(zero=False,
        #                             domain=list(np.quantile(base.data[y], [.05, .95])))),
        #     tooltip=color,
        # ).interactive())
    else:  # Stripplot
        charts.append(base.mark_tick(clip=True, opacity=1, size=12).encode(
            x=x,
            color=f'{color}',
            detail=detail,
            y=alt.Y(f'mean({y})',
                    axis=alt.Axis(orient='right'),
                    scale=alt.Scale(zero=False, domain=y_domain)),
            tooltip=color
        ))
    # axis = alt.Axis(title='') # labels=False, tickCount=0,

    if add_box:
        # Shift x axis for box so that it doesnt overlap:
        # if len(base[x[:-2]].unique()) == 1:
        #     df['x_box'] = df[x[:-2]] + .01
        charts.append(base.mark_boxplot(clip=True, opacity=.3, size=9, color='black').encode(
            x=x,
            y=alt.Y(f'{y}:Q',
                    axis=alt.Axis(orient='right', title=''),
                    scale=alt.Scale(zero=False, domain=y_domain)
                    )
        ))
    return alt.layer(*charts), y_domain


def plot_posterior(df=None, title='', x=':O', do_make_change=True, base_chart=None, add_zero_line=True, **kwargs):
    assert (df is not None) or (base_chart is not None)
    data = base_chart.data if df is None else df
    if x[-2] != ':':
        x = f'{x}:O'  # Ordinal
    assert 'higher interval' in data.columns
    assert 'lower interval' in data.columns
    assert 'center interval' in data.columns
    base_chart = base_chart or alt.Chart(data=data)

    # Axis limits
    minmax = [float(data['lower interval'].min()), 0,
              float(data['higher interval'].max())]
    scale = alt.Scale(zero=do_make_change is not False,  # Any string or True
                      domain=[min(minmax), max(minmax)])

    # error_bars
    chart = base_chart.mark_rule(size=2).encode(
        x=x,
        y=alt.Y('lower interval:Q',
                scale=scale,
                # axis=alt.Axis(labels=False, tickCount=1, title='')
                axis=alt.Axis(orient='left', title='')
                ),
        y2='higher interval:Q',
    )

    # Make the zero line
    if add_zero_line:
        title = f'Δ {title}'
        base_chart.data['zero'] = 0
        chart += base_chart.mark_rule(color='black', size=.5, opacity=1).encode(
            y=alt.Y(
                'zero',
                scale=scale,
                axis=alt.Axis(title='', orient='left')
            )
        )

    # line or bar for center interval (left axis)
    if x == ':O':
        chart += base_chart.mark_bar(color='black', filled=False, opacity=1, size=17).encode(
            y=alt.Y('center interval:Q',
                    title=title,
                    scale=scale,
                    # impute=alt.ImputeParams(value='value'),
                    axis=alt.Axis(orient='left'),
                    ),
            x=x,
        )
    else:
        chart += base_chart.mark_line(clip=True, point=True, color='black', fill=None).encode(
            y=alt.Y('center interval:Q',
                    title=title,
                    scale=scale,
                    # impute=alt.ImputeParams(value='value'),
                    axis=alt.Axis(orient='left'),
                    ),
            x=x,
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


def plot_posterior_density(base_chart, y, y_scale, trace, posterior, b_name):
    alt.data_transformers.disable_max_rows()

    # Same y domain as in plot_data and plot_posterior:
    if y in base_chart.data.columns:  # eg if we had add_data=True
        scale = alt.Scale(domain=y_scale)
    else:
        scale = alt.Scale()

    # dataframe with posterior (combine chains):
    df = trace.posterior.stack(draws=("chain", "draw")).reset_index(["draws"]).to_dataframe().reset_index()

    # n_draws = float(trace.posterior['chain'].max() * trace.posterior['draw'].max())
    # KDE chart:
    return alt.Chart(df).transform_density(
        b_name,
        as_=[b_name, 'density'],
        extent=[posterior['lower interval'].min(), posterior['higher interval'].max()],
        counts=True,
    ).mark_area(orient='horizontal', clip=False, fillOpacity=.2, color='black').encode(
        y=alt.Y(b_name, scale=scale, title=''),
        x=alt.X('density:Q', stack='center', title='',  # scale=alt.Scale(domain=[-n_draws, n_draws]),
                axis=alt.Axis(labels=False, tickCount=0, title='', values=[0])
                ),
    ).properties(width=30)

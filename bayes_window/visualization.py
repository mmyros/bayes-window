import altair as alt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform


def facet(base_chart: alt.LayerChart,
          column: str = None,
          row: str = None,
          width: int = 80,
          height: int = 150,
          finalize: bool = False
          ) -> alt.Chart:
    """

    Returns
    -------
    object
    """
    if column is None and row is None:
        return base_chart  # Nothing to do
    assert base_chart.data is not None
    if column:
        if column not in base_chart.data.columns:
            raise KeyError(f'{column} is not in {base_chart.data.columns}')
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


def posterior_intercept_chart(data_and_posterior, x, y, group=':O'):
    return alt.Chart(data_and_posterior).mark_tick(color='red').encode(
        x=x,
        y=alt.Y(
            y,
            scale=alt.Scale(domain=[data_and_posterior[y].min(), data_and_posterior[y].max()]),
            title='Intercept estimate'
        ),
        color=alt.Color(group)
    )


def auto_facet(group, condition):
    kwargs = {}
    if group and condition[0]:
        kwargs = {'column': group, 'row': condition[0]}
    elif group:
        kwargs = {'column': group}
    elif condition[0]:
        kwargs = {'column': condition[0]}
    return kwargs


def line_with_highlight(base, x, y, color, detail, highlight=True, y_domain=None):
    # Highlight doesnt work with overlays, but at least we get visible dots in legend
    y_domain = y_domain or list(np.quantile(base.data[y], [.05, .95]))
    if highlight:
        highlight = alt.selection(type='single', on='mouseover', fields=[color], nearest=True)
        size = alt.condition(~highlight, alt.value(1), alt.value(3))
    else:
        size = alt.value(1.)

    if (x != ':O') and (x != ':N') and x[:-2] in base.data.columns and len(base.data[x[:-2]].unique()) < 10:
        long_x_axis = False
    else:
        long_x_axis = True
        x = f'{x[:-1]}Q'  # Change to nominal encoding

    lines = base.mark_line(clip=True, fill=None, opacity=.6).encode(
        size=size,
        x=x,
        color=f'{color}',
        y=alt.Y(f'mean({y})',
                title=y,
                axis=alt.Axis(orient='right'),
                scale=alt.Scale(zero=False, domain=y_domain)),
        detail=detail
    )
    points = base.mark_circle(clip=True, opacity=0, filled=True).encode(

        x=x,
        color=f'{color}',
        y=alt.Y(f'mean({y})',
                axis=alt.Axis(title='', orient='right'),
                scale=alt.Scale(zero=False, domain=y_domain)),
        detail=detail
    )
    if highlight:
        points.add_selection(
            highlight
        )
    return lines, points


def plot_data(df=None, x='', y=None, color=None, base_chart=None, detail=':O', highlight=False, add_box=True,
              y_domain=None, **kwargs):
    assert (df is not None) or (base_chart is not None)
    if (x == '') or (x[-2] != ':'):
        x = f'{x}:O'
    if color is None:
        color = ':N'
    if color[-2] != ':':
        color = f'{color}:N'

    if not ((x != ':O') and (x != ':N') and len(df[x[:-2]].unique()) < 10):
        x = f'{x[:-1]}Q'  # Change to quantitative encoding

    charts = []
    # Plot data:
    base = base_chart or alt.Chart(df)
    y_domain = y_domain or list(np.quantile(base.data[y], [.05, .95]))

    if (x != ':O') and (x != ':Q') and (len(base.data[x[:-2]].unique()) > 1):
        data_line, data_pts = line_with_highlight(base, x, y, color, detail, highlight=highlight, y_domain=y_domain)
        charts.append(data_line)
        charts.append(data_pts)
        # charts.append(base.mark_line(clip=True, fill=None, opacity=.6, size=.5).encode(
        #     x=x,
        #     color=f'{color}',
        #     detail=detail,
        #     y=alt.Y(f'mean({y})',
        #             scale=alt.Scale(zero=False,
        #                             domain=list(np.quantile(base.data[y], [.05, .95])))),
        #     tooltip=color,
        # ).interactive())
    # else:
    #
    #     charts.append(base.mark_circle(clip=True, opacity=0, filled=True).encode(
    #
    #         x=x,
    #         color=f'{color}',
    #         y=alt.Y(f'mean({y})',
    #                 axis=alt.Axis(title='', orient='right'),
    #                 scale=alt.Scale(zero=False,
    #                                 domain=list(np.quantile(base.data[y], [.05, .95])))),
    #         detail=detail
    #     ))
    # else:  # Stripplot
    #     charts.append(base.mark_tick(clip=True, opacity=1, size=12).encode(
    #         x=x,
    #         color=f'{color}',
    #         detail=detail,
    #         y=alt.Y(f'{y}',
    #                 axis=alt.Axis(orient='right'),
    #                 scale=alt.Scale(zero=False, domain=y_domain)),
    #         tooltip=color
    #     ))

    if add_box:
        charts.append(base.mark_boxplot(clip=True, opacity=.3, size=9, color='black',
                                        median=alt.MarkConfig(color='red', strokeWidth=20)).encode(
            x=x,
            y=alt.Y(f'{y}:Q',
                    axis=alt.Axis(orient='right', title=''),
                    scale=alt.Scale(zero=False, domain=y_domain)
                    )
        ))
    return alt.layer(*charts), y_domain


def plot_posterior(df: pd.DataFrame = None, title: str = '', x: str = ':O', do_make_change: bool = True,
                   base_chart: object = None,
                   **kwargs: object) -> object:
    assert (df is not None) or (base_chart is not None)
    data = base_chart.data if df is None else df
    if x[-2] != ':':
        x = f'{x}:O'  # Ordinal
    assert 'higher interval' in data.columns
    assert 'lower interval' in data.columns
    assert 'center interval' in data.columns
    base_chart = base_chart or alt.Chart(data=data)

    # Add zero for zero line
    if 'zero' not in base_chart.data.columns:
        base_chart.data['zero'] = 0
    # Axis limits
    minmax = [float(data['lower interval'].min()), 0,
              float(data['higher interval'].max())]
    scale = alt.Scale(zero=do_make_change is not False,  # Any string or True
                      domain=[min(minmax), max(minmax)])
    if (x != ':O') and (x != ':N') and x[:-2] in data.columns and len(data[x[:-2]].unique()) < 10:
        long_x_axis = False
    else:
        long_x_axis = True
        x = f'{x[:-1]}Q'  # Change to nominal encoding

    # error_bars
    chart_posterior_whiskers = base_chart.mark_rule(
        size=2 if not long_x_axis else .8,
        opacity=.7 if not long_x_axis else .4,
        clip=True).encode(
        x=x,
        y=alt.Y('mean(lower interval):Q',
                scale=scale,
                # axis=alt.Axis(labels=False, tickCount=1, title='')
                axis=alt.Axis(orient='left', title='')
                ),
        y2='mean(higher interval):Q',
    )

    # Make the zero line
    title = f'Δ {title}'
    if (x == ':O') or (x == ':N'):
        chart_zero = base_chart.mark_rule(
            color='black',
        ).encode(
            y=alt.Y(
                'zero',
                scale=scale,
                axis=alt.Axis(title='', orient='left')
            )
        )
    else:
        chart_zero = base_chart.mark_line(color='black', size=.5, opacity=1).encode(
            x=x,
            y=alt.Y(
                'zero',
                scale=scale,
                axis=alt.Axis(title='', orient='left')
            )
        )

    # line or bar for center interval (left axis)
    if (x == ':O') or (x == ':N'):  # Bar
        chart_posterior_center = base_chart.mark_bar(color='black', filled=False, opacity=1, size=17).encode(
            y=alt.Y('center interval:Q',
                    title=title,
                    scale=scale,
                    # impute=alt.ImputeParams(value='value'),
                    axis=alt.Axis(orient='left'),
                    ),
            x=x,
        )
    else:  # Line
        chart_posterior_center = base_chart.mark_line(
            clip=True, point=False, color='black', fill=None,
            size=2 if not long_x_axis else 1,
            # opacity=.7 if not long_x_axis else .5,
        ).encode(
            y=alt.Y('mean(center interval):Q',
                    title=title,
                    scale=scale,
                    # impute=alt.ImputeParams(value='value'),
                    axis=alt.Axis(orient='left'),
                    ),
            x=x,
        )

    return chart_posterior_whiskers, chart_posterior_center, chart_zero


def plot_data_slope_trials(x,
                           y,
                           detail,
                           color=None,
                           base_chart=None,
                           df=None,
                           y_domain=None,
                           **kwargs):
    assert (df is not None) or (base_chart is not None)
    color = color or ':N'
    if x[-2] != ':':
        x = f'{x}:O'  # Ordinal
    base_chart = base_chart or alt.Chart(df)
    y_domain = y_domain or list(np.quantile(base_chart.data[y], [.05, .95]))
    # mean firing rate per trial per mouse
    fig_trials = base_chart.mark_line(fill=None).encode(
        x=alt.X(x),
        y=alt.Y(y, scale=alt.Scale(zero=False,
                                   # domain=[df[y].min(), df[y].max()])),
                                   domain=y_domain,
                                   clamp=True)),
        color=color,
        detail=detail,
        opacity=alt.value(.2),
        size=alt.value(.9),
        # **kwargs
    )
    return fig_trials


def plot_posterior_density(base_chart, y, y_domain, trace, posterior, b_name, do_make_change):
    alt.data_transformers.disable_max_rows()

    # Same y domain as in plot_data and plot_posterior:
    if y in base_chart.data.columns and y_domain is not None:  # eg if we had add_data=True
        scale = alt.Scale(domain=y_domain)
    else:
        # Axis limits
        minmax = [float(posterior['lower interval'].min()), 0,
                  float(posterior['higher interval'].max())]
        scale = alt.Scale(zero=do_make_change is not False,  # Any string or True
                          domain=[min(minmax), max(minmax)])

    # dataframe with posterior (combine chains):
    df = trace.posterior.stack(draws=("chain", "draw")).reset_index(["draws"]).to_dataframe().reset_index()

    # n_draws = float(trace.posterior['chain'].max() * trace.posterior['draw'].max())
    # KDE chart:
    return alt.Chart(df).transform_density(
        b_name,
        as_=[b_name, 'density'],
        extent=[posterior['lower interval'].min(), posterior['higher interval'].max()],
        counts=True,
    ).mark_area(orient='horizontal', clip=False, fillOpacity=.06, color='black', strokeOpacity=.7,
                stroke='black').encode(
        y=alt.Y(b_name, scale=scale, title='', axis=alt.Axis(orient='left', title='')),
        x=alt.X('density:Q', stack='center', title='',  # scale=alt.Scale(domain=[-n_draws, n_draws]),
                axis=alt.Axis(labels=False, tickCount=0, title='', values=[0])
                ),
    ).properties(width=30)


def auto_layer_and_facet(charts, charts_for_facet=None, independent_axes=True, **kwargs):
    facet_requested = ('column' in kwargs.keys()) or ('row' in kwargs.keys())

    # 3. Make layered chart out of posterior and data
    if facet_requested and charts_for_facet is not None:
        _ = alt.layer(*charts_for_facet).facet()  # Sanity check
        chart = alt.layer(*charts_for_facet)
    else:
        chart = alt.layer(*charts)

    if independent_axes:
        chart = chart.resolve_scale(y='independent')

    if facet_requested:
        chart = facet(chart, **kwargs)
    return chart

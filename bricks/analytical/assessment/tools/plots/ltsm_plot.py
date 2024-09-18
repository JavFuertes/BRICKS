import numpy as np
from dash import Dash, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import get_colorscale
from ..utils import prepare_report, compute_param, get_color_from_scale
from matplotlib import cm

def get_colors_from_cmap(cmap_name, num_colors, opacity=0.7):  # Opacity for transparency
    """
    Get a list of rgba colors from a colormap with a specified number of colors and opacity.
    """
    cmap = cm.get_cmap(cmap_name, num_colors)
    colors = []
    for i in range(cmap.N):
        r, g, b, _ = cmap(i)
        colors.append(f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {opacity})')
    return colors

def discrete_colorscale(bvals, colors):
    if len(bvals) != len(colors) + 1:
        raise ValueError('len(boundary values) should be equal to len(colors) + 1')
    
    bvals = sorted(bvals)
    nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]  # Normalize the boundaries

    dcolorscale = []
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])
    
    return dcolorscale

# Set up boundary values and colors
bvals = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
colors = get_colors_from_cmap('RdYlGn_r', len(bvals) - 1, opacity=0.7)
dcolorsc = discrete_colorscale(bvals, colors)

# Calculate tick values and labels
tickvals = [np.mean(bvals[k:k+2]) for k in range(len(bvals) - 1)]
ticktext = ['DL0', 'DL1', 'DL2', 'DL3', 'DL4', 'DL5']

def LTSM_plot(object):
    house = object.house
    app = Dash(__name__)
    
    assess_types = list(object.assessment['ltsm'].keys())
    all_assessment_tabs = []

    tab_style = {'fontFamily': 'Arial, sans-serif', 'color': '#3a4d6b'}
    
    for assessment in assess_types:
        try:
            figs = []
            wltsm = object.assessment['ltsm'][assessment]
            for i, wall in enumerate(house):
                h = object.house[wall]['height']
                int = object.process['int'][wall]

                strain = wltsm['results'][wall]['e_tot']
                
                if assessment == 'greenfield':
                    ltsm_params = wltsm['variables'][wall]
                    ltsm_values = wltsm['values'][wall]

                param = [ltsm_params, ltsm_values]
                for dict_ in param:
                    for key, value in dict_.items(): # Unpacking all variables, not defined
                        globals()[key] = value

                fig = make_subplots(rows=2, cols=1,
                                    vertical_spacing=0.1,
                                    shared_xaxes=True,
                                    shared_yaxes=True,
                                    x_title='Length [m]',
                                    y_title='Profile & Height [mm]',
                                    subplot_titles=('Relative Wall position', 'Subsidence profile'))

                if assessment == 'measurements':
                    psi = object.assessment['ltsm']['measurements']['report'][wall]['epsilon'][0]['psi']
                    z_measurement = int["z_lin"]
                    x_measurement = int["ax_rel"]
                    fig.add_trace(go.Scatter(x=x_measurement,
                                            y=z_measurement,
                                            mode='lines',
                                            name='Subsidence profile', ),
                                row=2, col=1)
                    
                    infl_list = object.soil['shape'][wall]['inflection_points']
                    for infl in infl_list:
                        fig.add_trace(go.Scatter(x=[infl, infl], y=[0, h], mode='lines', name='Inflection point',
                                                line=dict(color='black', width=1, dash='dash')), row=1, col=1)

                        fig.add_trace(go.Scatter(x=[infl, infl], y=[min(z_measurement), max(z_measurement)], mode='lines', name='Inflection point',
                                                line=dict(color='black', width=1, dash='dash')), row=2, col=1)
                    
                    if z_measurement.max() > -5:
                        fig.add_trace(go.Scatter(x=x,
                                                y=np.full(len(x), limitline),
                                                mode='lines', name=f'Limit Line [{limitline} mm]',
                                                line=dict(color='black', dash='longdashdot', width=1)),
                                    row=2, col=1)

                if assessment == 'greenfield':
                    psi = object.assessment['ltsm']['greenfield']['report'][wall]['epsilon'][0]['psi']
                    fig.add_trace(go.Scatter(x=x,
                                            y=w,
                                            mode='lines',
                                            name='Gaussian profile <br>approximation'),
                                row=2, col=1)
                    
                    for z in [-xinflection, xinflection]:
                        fig.add_trace(go.Scatter(x=[z, z], y=[0, h], mode='lines', name='Inflection point',
                                                line=dict(color='black', width=1, dash='dash')), row=1, col=1)

                        fig.add_trace(go.Scatter(x=[z, z], y=[w.min(), w.max()], mode='lines', name='Inflection point',
                                                line=dict(color='black', width=1, dash='dash')), row=2, col=1)

                    for influence in [-xlimit, xlimit]:
                        fig.add_trace(go.Scatter(x=[influence, influence], y=[0, h], mode='lines', name='Influence area',
                                                line=dict(color='black', width=1, dash='dashdot')), row=1, col=1)

                        fig.add_trace(go.Scatter(x=[influence, influence], y=[w.min(), w.max()], mode='lines', name='Influence area',
                                                line=dict(color='black', width=1, dash='dashdot')), row=2, col=1)
                    fig.add_trace(go.Scatter(x=x,
                                            y=np.full(len(x), limitline),
                                            mode='lines', name=f'Limit Line [{limitline} mm]',
                                            line=dict(color='black', dash='longdashdot', width=1)),
                                row=2, col=1)

                if i % 2 == 0:
                    xi = object.house[wall]['y'].max()
                    xj = object.house[wall]['y'].min()
                else:
                    xi = object.house[wall]['x'].max()
                    xj = object.house[wall]['x'].min()
                
                if object.assessment['ltsm'][assessment]['report']:
                    report = object.assessment['ltsm'][assessment]['report']
                    data_matrix, wall_param_labels, sources, description_annotations = prepare_report(report, wall)
                    
                    comments = description_annotations[0]
                    assessments = data_matrix.flatten()
                    
                    dlmax = 5
                    color_matrix = [get_color_from_scale(damage, dcolorsc, dlmax) for damage in assessments]
                    segment_width = (xi - xj) / len(assessments)

                    for i, (color, assess_i, comment, source) in enumerate(zip(color_matrix, assessments, comments, sources)):
                        fig.add_shape(
                            type="rect",
                            x0=i * segment_width,
                            y0=0,
                            x1=(i + 1) * segment_width,
                            y1=h,
                            fillcolor=color[1],  # Using the rgba color with transparency
                            opacity=0.7,
                            line=dict(width=0),
                            row=1, col=1
                        )
                        fig.add_trace(go.Scatter(
                            x=[(i + 0.5) * segment_width],
                            y=[h / 2],
                            text=f"Source: {source} <br>DL: {assess_i} <br>Assessment: {comment}",
                            mode='markers',
                            showlegend=False,
                            marker=dict(size=0.1, color='rgba(0,0,0,0)'),
                            hoverinfo='text'
                        ))

                    fig.add_shape(type="rect",
                                    x0=0,
                                    y0=0,
                                    x1=xi - xj,
                                    y1=h,
                                    fillcolor='rgba(49,50,51,0)',
                                    line=dict(color='black', width=2),
                                    row=1, col=1)

                else:
                    try:
                        ecolor, cat = compute_param(strain)
                    except:
                        ecolor = 'rgba(49,50,51,0)'
                        cat = 'No assessment'
                    fig.add_shape(
                        type="rect",
                        x0=0, y0=0,
                        x1=xi - xj, y1=h,
                        fillcolor=ecolor,
                        row=1, col=1)
                
                ind = np.argmax(assessments)
                cat = comments[ind]
                eg = object.house[wall]['eg_rat']
                dl = assessments.max()

                fig.update_layout(
                            title= f'{wall} | E/G = {eg:.1f}   ε_tot = {strain:.2e} | DL = {dl:.0f}',
                            legend=dict(traceorder="normal"),
                            template='plotly_white'
                        )

                if assessment == 'measurements':
                    valr2 = x_measurement.max() * 1.5 
                    x_range = [-valr2, valr2]
                    
                    fig.update_xaxes(range=x_range, row=1, col=1)
                    fig.update_xaxes(range=x_range, row=2, col=1)

                names = set()
                fig.for_each_trace(
                    lambda trace:
                    trace.update(showlegend=False)
                    if (trace.name in names) else names.add(trace.name))
                
                figs.append(fig)

                # Add color bar as a separate trace with specific height and position
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(
                        colorscale=dcolorsc,  # Using the discrete color scale
                        cmin=0,
                        cmax=5,
                        colorbar=dict(
                            thickness=25,
                            tickvals=[0, 1, 2, 3, 4, 5],  # Ensure that all damage levels (0-5) are present
                            ticktext=['DL0', 'DL1', 'DL2', 'DL3', 'DL4', 'DL5'],  # Corresponding text for each damage level
                            title=dict(
                                text='Damage Level<br>(per assessment)',
                                font=dict(size=14),
                                side='right'
                            ),
                            len=0.6,  
                            yanchor='bottom',
                            y = 0,
                        )
                    ),
                    showlegend=False
                ))

        except Exception as e:
            print(f'{e} for assessment {assessment}')
            continue

        wall_tabs = [dcc.Tab(label=f"Wall {i+1}", children=[dcc.Graph(figure=fig)], style=tab_style, selected_style=tab_style) for i, fig in enumerate(figs)]
        assessment_tab = dcc.Tab(label=f"{assessment.capitalize()} assessment", children=[dcc.Tabs(children=wall_tabs)], style=tab_style, selected_style=tab_style)
        all_assessment_tabs.append(assessment_tab)

    app.layout = html.Div([
        dcc.Tabs(children=all_assessment_tabs)
    ])

    return app







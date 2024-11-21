from dash import Dash, dcc, html
import plotly.graph_objects as go
import numpy as np
import matplotlib.cm as cm
from ..utils import prepare_report

def apply_opacity_to_colorscale(colorscale_name, opacity):
    cmap = cm.get_cmap(colorscale_name)
    rgba_colors = []
    for i in range(cmap.N):
        r, g, b, _ = cmap(i)
        rgba_colors.append([i / (cmap.N - 1), f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {opacity})'])
    return rgba_colors

def split_labels_into_words(labels, max_words_per_line):
    return ['<br>'.join([' '.join(label.split()[i:i+max_words_per_line]) for i in range(0, len(label.split()), max_words_per_line)]) for label in labels]

def get_colors_from_cmap(cmap_name, num_colors, opacity=0.7):  # Default opacity of 70%
    cmap = cm.get_cmap(cmap_name, num_colors)
    colors = []
    for i in range(cmap.N):
        r, g, b, _ = cmap(i)
        colors.append(f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {opacity})')
    return colors

bvals = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]  
colors = get_colors_from_cmap('RdYlGn_r', len(bvals) - 1, opacity=0.7)  

def discrete_colorscale(bvals, colors):
    if len(bvals) != len(colors) + 1:
        raise ValueError('len(boundary values) should be equal to len(colors) + 1')
    bvals = sorted(bvals)
    nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]  
    
    dcolorscale = []
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])
    return dcolorscale

dcolorsc = discrete_colorscale(bvals, colors)

tickvals = [np.mean(bvals[k:k+2]) for k in range(len(bvals) - 1)]
ticktext = ['DL0', 'DL1', 'DL2', 'DL3', 'DL4', 'DL5']

def EM_plot(report):
    app = Dash(__name__)
    walls = list(report.keys())
    
    figs = []  

    for wall in walls:
        data_matrix, wall_param_labels, sources, description_annotations = prepare_report(report, wall)
        formatted_sources = split_labels_into_words(sources, 2)
        
        masked_data_matrix = np.nan_to_num(data_matrix, nan=-1)
        heatmap = go.Heatmap(
            z=np.where(masked_data_matrix == -1, np.nan, masked_data_matrix),  
            x=formatted_sources,
            y=wall_param_labels,
            colorscale=dcolorsc,  
            zmin=0,
            zmax=5,
            colorbar=dict(
                thickness=25,
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=dict(size=13),
                title=dict(
                    text='Damage <br>Level',
                    font=dict(size=15),  # Adjust font size for title
                    side='top'  # Position title above the colorbar
                ),
                lenmode='fraction',  
                len=1.0,
            ),
            hoverongaps=False,
            hoverinfo='text',
            text=np.vectorize(lambda desc: f"{desc}" if desc is not np.nan else "No Data")(description_annotations),
            customdata=np.array(description_annotations),
            xgap= 12,  
            ygap= 12   
        )

        layout = go.Layout(
            xaxis=dict(
                title='Literature Source',
                side='bottom',
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,
                tickangle=0,
                tickmode='auto',
                automargin=True,
                ticklen=25,
                title_standoff=15,
                layer='below traces',
                tickfont=dict(size=13),
                titlefont=dict(size=15)
            ),
            yaxis=dict(
                title='SRI Parameter',
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,
                autorange='reversed',
                automargin=True,
                title_standoff=15,
                ticklen=25,
                layer='below traces',
                tickfont=dict(size=13),
                titlefont=dict(size=15)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            template='plotly_white',
            margin=dict(l=120, r=50, t=40, b=120),
            showlegend=True
        )

        fig = go.Figure(data=[heatmap], layout=layout)
        figs.append(fig)

    tab_heading_style = {'fontFamily': 'Arial, sans-serif', 'color': '#3a4d6b'}

    tabs_content = [dcc.Tab(label=f"{wall.capitalize()}", children=[dcc.Graph(figure=fig)], style=tab_heading_style, selected_style=tab_heading_style) for wall, fig in zip(walls, figs)]

    app.layout = html.Div([dcc.Tabs(children=tabs_content)], style={'backgroundColor': 'white'})

    return app

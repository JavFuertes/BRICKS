from dash import Dash, dcc, html
import plotly.graph_objects as go
import numpy as np
import matplotlib.cm as cm
from ..utils import prepare_report

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

def EM_plot(report):
    """
    Generate an annotated heatmap plot for empirical assessment.

    Args:
        report (dict): A dictionary containing the empirical assessment report.

    Returns:
        app (Dash): The Dash application object.
    """    
    app = Dash(__name__)
    walls = list(report.keys())
    
    figs = []  
    cscale = apply_opacity_to_colorscale('RdYlGn_r', 0.75)
    max_words_per_line = 2  

    for wall in walls:
        data_matrix, wall_param_labels, sources, description_annotations = prepare_report(report, wall)
        formatted_sources = split_labels_into_words(sources, max_words_per_line)
    
        heatmap = go.Heatmap(
            z=data_matrix,
            x=formatted_sources,
            y=wall_param_labels,
            colorscale=cscale,
            zmin=0,
            zmax=5,
            colorbar=dict(
                title='Damage Level:<br> <br>',
                titleside='top',
                tickmode='array',
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=['0-None', '1-Negligible', '2-Moderate', '3-Severe', '4-Very Severe', '5-Extreme']
            ),
            hoverongaps=False,
            hoverinfo='text',
            text=np.vectorize(lambda desc: f"{desc}")(description_annotations),
            customdata=np.array(description_annotations)
        )

        layout = go.Layout(
            xaxis=dict(
                title='Literature Source',
                side='bottom',
                showgrid=True,
                gridcolor='lightgray',
                tickangle=0,  # Keep labels horizontal
                tickmode='auto'
            ),
            yaxis=dict(
                title='SRI Parameter',
                showgrid=True,
                gridcolor='lightgray',
                autorange='reversed'
            ),
            template='plotly_white',
            margin=dict(l=100, r=20, t=40, b=150)  # Adjust bottom margin to fit multi-line labels
        )

        fig = go.Figure(data=heatmap, layout=layout)
        figs.append(fig)

    tab_heading_style = {
        'fontFamily': 'Arial, sans-serif',
        'color': '#3a4d6b'
    }

    tabs_content = [dcc.Tab(label=f"{wall.capitalize()}", children=[dcc.Graph(figure=fig)], style=tab_heading_style, selected_style=tab_heading_style) for wall, fig in zip(walls, figs)]

    app.layout = html.Div([
        dcc.Tabs(children=tabs_content)], style={'backgroundColor': 'white'})

    if __name__ == '__main__':
        app.run_server(debug=False)

    return app
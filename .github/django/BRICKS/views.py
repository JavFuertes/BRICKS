from django.shortcuts import render
from .forms import WallDictForm
import numpy as np
import sys

# Add the path where your 'bricks' module is located
module_path = r'C:\Users\fuertesguadarramaj\OneDrive - Delft University of Technology\Year 2\Q3 & Q4\CIEM0500 - MS Thesis Project\!content\Experimentation\!Ijsselsteinseweg\bricks'
sys.path.append(module_path)

from analytical import house
from analytical.assessment.tools.plots import subsurface, EM_plot, LTSM_plot


def extract_figures_from_dash_app(dash_app):
    """
    Extract figures from a Dash app and convert them into HTML.

    :param dash_app: The Dash app object.
    :return: A dictionary where keys are tab names and values are HTML strings of the Plotly figures.
    """
    html_figures = {}

    if hasattr(dash_app.layout, 'children'):
        for child in dash_app.layout.children:
            if isinstance(child, dcc.Graph):
                figure = child.figure
                html_figures['default'] = pio.to_html(figure, full_html=False)
            elif hasattr(child, 'children'):
                for tab in child.children:
                    if isinstance(tab, dcc.Tab):
                        tab_label = tab.label
                        for tab_content in tab.children:
                            if isinstance(tab_content, dcc.Graph):
                                figure = tab_content.figure
                                html_figures[tab_label] = pio.to_html(figure, full_html=False)

    return html_figures

def index(request):
    form = WallDictForm(request.POST or None)
    context = {'form': form}

    # Define the walls dictionary directly in the view
    walls = {
        'Wall 1': {"x": np.array([0, 0, 0]), "y": np.array([0, 3.5, 7]), "z": np.array([0, -72, -152]), 'phi': np.array([1/200, 1/200]), 'height': 5250, 'thickness': 27, 'area': 34.25, 'opening': 4.86},
        'Wall 2': {"x": np.array([0, 4.5, 8.9]), "y": np.array([7, 7, 7]), "z": np.array([-152, -163, -188]), 'phi': np.array([1/33, 1/50]), 'height': 5250, 'thickness': 27, 'area': 37, 'opening': 9.36},
        'Wall 3': {"x": np.array([8.9, 8.9]), "y": np.array([3.6, 7]), "z": np.array([-149, -188]), 'phi': np.array([0, 0]), 'height': 5250, 'thickness': 27, 'area': 24.35, 'opening': 4.98},
        'Wall 4': {"x": np.array([8.9, 10.8]), "y": np.array([3.6, 3.6]), "z": np.array([-149, -138]), 'phi': np.array([0, 0]), 'height': 2850, 'thickness': 27, 'area': 8.09, 'opening': 1.68},
        'Wall 5': {"x": np.array([10.8, 10.8]), "y": np.array([0, 3.6]), "z": np.array([-104, -138]), 'phi': np.array([0, 0]), 'height': 2850, 'thickness': 27, 'area': 9.15, 'opening': 1},
        'Wall 6': {"x": np.array([0, 5.2, 6.4, 8.9, 10.8]), "y": np.array([0, 0, 0, 0, 0]), "z": np.array([0, -42, -55, -75, -104]), 'phi': np.array([1/100, 1/100]), 'height': 5000, 'thickness': 27, 'area': 47.58, 'opening': 4.42},
    }

    damage = {
        'crack_1': {'wall_id': 'Wall 2', 'c_w': 4, 'c_l': 890,},
        'crack_5': {'wall_id': 'Wall 2', 'c_w': 3, 'c_l': 1760,},
        'crack_9': {'wall_id': 'Wall 2', 'c_w': 2, 'c_l': 994,},
    }

    ijsselsteinseweg = house(measurements=walls)
    ijsselsteinseweg.state = damage
    ijsselsteinseweg.interpolate()
    ijsselsteinseweg.fit_function(i_guess=1, tolerance=1e-2, step=1)

    try:
        subsurface_app = subsurface(ijsselsteinseweg, *ijsselsteinseweg.soil['house'].values())
        em_app = EM_plot(ijsselsteinseweg.soil['sri'])
        ltsm_app = LTSM_plot(ijsselsteinseweg)

        subsurface_figs = extract_figures_from_dash_app(subsurface_app)
        em_figs = extract_figures_from_dash_app(em_app)
        ltsm_figs = extract_figures_from_dash_app(ltsm_app)

        context.update({
            'subsurface_plot': subsurface_figs.get('default', ''),
            'em_plot': em_figs.get('default', ''),
            'ltsm_plot': ltsm_figs.get('default', ''),
        })

        print(context)  # Add this line for debugging

    except Exception as e:
        context['error'] = f"Error generating plots: {e}"
        print(f"Error: {e}")  # Add this line for debugging

    return render(request, 'BRICKS/index.html', context)

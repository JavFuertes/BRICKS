import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html
from plotly.subplots import make_subplots

from ....assessment.utils import gaussian_shape

def plot_surface(x_mesh, y_mesh, z_lin, z_qint):

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                        subplot_titles=('Through differential settlements 1st order interpolation', 'Through sagging-hogging soil profile 2nd order interpolation'),
                        )
    
    fig.add_trace(
        go.Surface(x=x_mesh, y=y_mesh, z=z_lin, colorscale='solar', opacity=0.8),
        row=1, col=1
    )
    fig.add_trace(
        go.Surface(x=x_mesh, y=y_mesh, z=z_qint, colorscale='solar', opacity=0.8),
        row=1, col=2
    )

    fig.update_traces(contours_z=dict(show=True, usecolormap=False))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X [m]', color='black'),
            yaxis=dict(title='Y [m]',color='black'),
            zaxis=dict(title='Z [mm]',color='black'),
            camera_eye=dict(x=0.5, y=1.2, z=0.4),
            aspectratio=dict(x= 2, y= 3, z= 0.5),
        ),
        scene2=dict(
            xaxis=dict(title='X [m]',color='black'),
            yaxis=dict(title='Y [m]',color='black'),
            zaxis=dict(title='Z [mm]',color='black'),
            camera_eye=dict(x=0.5, y=1.2, z=0.4),
            aspectratio=dict(x= 2, y= 3, z= 0.5),
        ),
        title='Aproximated subsidence surface',
        template='plotly_white',
    )

    return fig

def wall_displacement(OBJECT):
    fig = go.Figure()
    colors = ['#00a6d6', '#A50034', '#FFB81C', '#0076c2', '#EC6842']  
    line_types = ['solid', 'dash']
    process = OBJECT.process

    for i, wall in enumerate(OBJECT.house):
        color_index = i % len(colors)
        fig.add_trace(go.Scatter(x= process['int'][wall]['ax_rel'], 
                                y= process['int'][wall]["z_lin"], 
                                mode='lines',
                                line=dict(color=colors[color_index], dash=line_types[0]),
                                name=f'Wall {i+1} Linear'))
        fig.add_trace(go.Scatter(x= process['int'][wall]['ax_rel'], 
                                y= process['int'][wall]['z_q'], 
                                mode='lines',
                                line=dict(color=colors[color_index], dash=line_types[1]),
                                name=f'Wall {i+1} Quadratic'))

    fig.update_layout(title='Displacement profile of different Walls',
                    xaxis_title='Wall length [m]',
                    yaxis_title='Displacement W[mm]',
                    legend=dict(x=1.05, y=1, traceorder="normal"), 
                    template='plotly_white',)
    return fig

def building_traces(OBJECT):
    
    x_bound = []
    y_bound = []
    wall_val = OBJECT.house.values()

    for wall in wall_val:
        x = np.concatenate([wall['x'], wall['x'][::-1]])   
        x_bound += [wall['x'].min(), wall['x'].max()] 
        x = np.vstack([x,x])

        y = np.concatenate([wall['y'], wall['y'][::-1]])
        y = np.vstack([y,y])
        y_bound += [wall['y'].min(), wall['y'].max()]

        z = np.empty((2,len(wall)))
        z[0,:] = np.full((1,len(wall)), fill_value= 0)
        z[1,:] = np.full((1,len(wall)), fill_value = wall['height'])
        
        x_bound = np.array(x_bound) # Floor and roof
        y_bound = np.array(y_bound)
        z0 = np.zeros(x_bound.shape)
        z1 = np.full(x_bound.shape, wall['height'])
        h = [z0,z1]

        return {'walls': {'x:': x,
                          'y': y,
                          'h': h,},
                'floors': {'x': x_bound,
                           'y': y_bound,
                           'h': h}}

def subsidence(OBJECT, building = False, soil = False, deformation = False):
    """
    Plot subsidence data.

    ## Args:
        OBJECT: The object containing the subsidence data.
        building (bool): Whether to plot building data. Default is False.
        soil (bool): Whether to plot soil data. Default is False.
        deformation (bool): Whether to plot deformation data. Default is True.

    ## Returns:
        go.Figure: The plotly figure object.
    """    
    fig = go.Figure()
    x_mesh, y_mesh, z_lin, z_qint = OBJECT.soil['house'].values()

    if deformation: #Plot estimation of measured deformation surface
        fig.add_trace(
            go.Surface(x=x_mesh,
                    y=y_mesh, 
                    z=z_qint, 
                    colorscale='solar', 
                    opacity=0.8,
                    showscale=False,)
        )

        house = OBJECT.house  
        color = ['black','#EC6842','#00B8C8']
        for i, wall in enumerate(house):
            wall = house[wall]
            if np.all(wall['x'] == wall['x'][0]):  # Wall is along the y axis
                y_lin = np.linspace(wall["y"].min() , wall["y"].max() , 100)
                interp_vals = np.interp(y_lin, wall['y'], wall['z'])
                x = np.full_like(y_lin, wall['x'][0])  # Xcoord constant
                fig.add_trace(go.Scatter3d(x=x, y=y_lin, 
                                        z=interp_vals, 
                                        mode='lines', 
                                        name='Wall',
                                        line = dict(color=color[0])))
                fig.add_trace(go.Scatter3d(x=np.full_like(wall['y'], wall['x'][0]), 
                                            y=wall['y'], 
                                            z= wall['z'],
                                            mode='markers', 
                                            marker=dict(size=4, color=color[0]),
                                            showlegend=False))
            else:  # Wall is along the x axis
                x_lin = np.linspace(wall["x"].min() , wall["x"].max() , 100)
                interp_vals = np.interp(x_lin, wall['x'], wall['z'])
                y = np.full_like(x_lin, wall['y'][0])  # Ycoord constant
                fig.add_trace(go.Scatter3d(x=x_lin, 
                                                y=y, 
                                                z=interp_vals, 
                                                mode='lines',
                                                name='Wall',
                                                line = dict(color=color[0])))
                # 
                fig.add_trace(go.Scatter3d(x= wall['x'],
                                                y=np.full_like(wall['x'], 
                                                wall['y'][0]), 
                                                z=wall['z'], 
                                                mode='markers',
                                                marker=dict(size=4, color=color[0]),
                                                showlegend=False))
        int_soil = OBJECT.soil['soil']
        for key, value in int_soil.items():
            globals()[key] = value

        x = OBJECT.soil['soil']['x']
        y = OBJECT.soil['soil']['y']
        z = OBJECT.soil['soil']['z']

        fig.add_trace(
            go.Surface(x=x, 
                        y=y,
                        z=z,
                        colorscale='blackbody',
                        opacity=0.5,
                        contours = {
                        "z": {"show": True},},
                        showscale=True,
                            ),
        )
    
    # if building:
    #     # ------------------------------------ FIX ----------------------------------- #
    #     bwalls, bfloors = building_traces(OBJECT).values()
    #     x, y, z = bwalls.values()
    #     x_bound, y_bound, h = bfloors.values()

    #     wcolor = []
    #     if OBJECT.dfltsm != None:
    #         for z in z:strain = wltsm['results']['e_tot']
    #         ecolor, cat = compute_param(strain)
    #         wcolor.append(ecolor)
            
    #     fig.add_trace(go.Surface(x=x,
    #                          y=y,
    #                          z=z,
    #                          colorscale=[[0, ecolor], [1, ecolor]],
    #                          showscale=False))
    #     for z in h:
    #         fig.add_trace(go.Surface(x= x_bound,
    #                                 y= y_bound,
    #                                 z=z,
    #                                 colorscale=[[0, 'rgba(0,0,255,0.5)'], [1, 'rgba(0,0,255,0.5)']], 
    #                                 showscale=False))

    if soil:
        x_x =[]
        x_y =[]
        y_x = []
        y_y = []
        for i, key in enumerate(house):
            wall = house[key]
            w_param = OBJECT.process['params'][key]
            
            xnormal = w_param['ax']
            x = w_param['x_gauss']
            svmax = w_param['s_vmax']
            xinflection = w_param['x_inflection']
            z_curr = gaussian_shape(x,svmax,xinflection)

            if i % 2 == 0:  # Wall is along the y axis
                y_y += [xnormal.min(),xnormal.max()]
                y_x += [wall['x'].min()]*2
                x_const = np.full(xnormal.shape[0], wall['x'].min())
                fig.add_trace(go.Scatter3d(x=x_const,
                                                y= xnormal,
                                                z=z_curr,
                                                mode='lines',
                                                name='Approximated subsidence profile',
                                                line=dict(color=color[0], dash='dash')))
            
            else:  # Wall is along the x axis
                x_x += [xnormal.max(),xnormal.min()]
                x_y += [wall['y'].min()]*2
                y_const = np.full(xnormal.shape[0], wall['y'].min())
                fig.add_trace(go.Scatter3d(x= xnormal,
                                            y=y_const,
                                            z=z_curr,
                                            mode='lines',
                                            name='Approximated subsidence profile',
                                            line=dict(color=color[0], dash='dash')))
    
    # ------------------------ Handle plot functionalities ----------------------- #
    z_min = -min(OBJECT.house[wall]["z"].min() for wall in OBJECT.house)/100
    z_rat = z_min / x_mesh.max()
    y_rat = y_mesh.max() / x_mesh.max()

    fig.update_layout(
        title='Approximated subsidence surface',
        scene=dict(
            xaxis_title='X [m]',
            yaxis_title='Y [m]',
            zaxis_title='Z [mm]',
            aspectmode="manual",
            aspectratio=dict(x=1, y=y_rat, z=z_rat*0.25)
        ),
        coloraxis_colorbar=dict(
            title='Disp Uz [mm]', 
            yanchor="bottom", 
            y=1, 
            x=0,
            len=0.5,  # 70% of the plot area
            ticks="inside"
        ),
        showlegend = False,
        template='plotly_white'
    )

    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))
    return fig

def subsurface(OBJECT, *params):
    app = Dash(__name__)

    figures_info = [
        ('Wall Displacement', wall_displacement(OBJECT)),
        #('Plot Surface', plot_surface(*params)),
        ('Subsidence', subsidence(OBJECT, building=False, soil=True, deformation=True))
    ]
    tab_style = {
        'fontFamily': 'Arial, sans-serif',
        'color': '#3a4d6b'
    }

    tabs = [
        dcc.Tab(label=label, children=[
            dcc.Graph(figure=figure),],
            style=tab_style, selected_style= tab_style)
            for label, figure in figures_info
    ]

    app.layout = html.Div([
        dcc.Tabs(tabs)
    ])

    if __name__ == '__main__':
        app.run_server(debug=False)

    return app


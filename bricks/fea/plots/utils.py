import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


def add_shaded_areas_cw(ax, max_y):
    """
    Add shaded areas and labels to an axis based on damage levels (DL).

    Args:
        ax (Axes): The axes object to add shaded areas.
        max_y (float): The maximum y-value to consider for shading.

    Returns:
        None
    """
    DL = [0, 1, 2, 3, 4, 5]
    Name = ['Negligible', 'Very slight', 'Slight', 'Moderate', 'Severe', 'Very severe']
    CW = [0, 1, 5, 15, 25, max(25, max_y)]
    
    for i in range(len(CW) - 1):
        if CW[i] < max_y:
            upper_limit = min(CW[i + 1], max_y*1.1)
            alpha_value = 0.03 + 0.04 * i 
            ax.axhspan(CW[i], upper_limit, facecolor='gray', alpha=alpha_value)
            ax.text(ax.get_xlim()[1] -0.05 , (CW[i] + upper_limit) / 2, f'DL{DL[i]} {Name[i]}',
                    va='center', ha='right', fontsize=8, color='black',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    # Set y-axis upper limit to ensure all annotations are visible
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], max_y * 1.1)
            
def add_shaded_areas_psi(ax, max_psi):
    """
    Add shaded areas and labels to an axis based on PSI levels.

    Args:
        ax (Axes): The axes object to add shaded areas.
        max_psi (float): The maximum PSI value to consider for shading.

    Returns:
        None
    """

    psilim = [0, 1, 1.5, 2.5, 3.5, float('inf')]
    DL = [0, 1, 2, 3, 4, 5]
    Name = ['Negligible', 'Very slight', 'Slight', 'Moderate', 'Severe', 'Very severe']
    
    for i in range(len(psilim) - 1):
        if psilim[i] < max_psi:
            upper_limit = min(psilim[i + 1], max_psi*1.1)
            alpha_value = 0.03 + 0.04 * i 
            ax.axhspan(psilim[i] , upper_limit, facecolor='gray', alpha=alpha_value)
            ax.text(ax.get_xlim()[1] -0.05 , (psilim[i] + upper_limit) / 2, f'DL{DL[i]} {Name[i]}',
                    va='center', ha='right', fontsize=8, color='black',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], max_psi * 1.1)

def add_image_to_plot(ax, image_path, zoom=0.12):
    """
    Add an image to a plot at a specific location.

    Args:
        ax (Axes): The axes object to add the image to.
        image_path (str): The file path of the image to add.
        zoom (float): Zoom factor for the image. Default is 0.12.

    Returns:
        None
    """

    image = Image.open(image_path)
    image = np.array(image)
    im = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(im, (1, 0), xycoords='axes fraction', frameon=False,
                        box_alignment=(0.99, 0.01))  # Bottom right corner
    ax.add_artist(ab)

def individual_plot(data, plot_settings):
    """
    Plot the given data with the specified plot settings.

    Parameters:
    - data: numpy array, where each column represents a variable and each row represents a data point.
    - plot_settings: dict, contains settings for the plot including labels, title, scientific notation, and traces.

    Returns:
    - fig: The matplotlib figure object.
    - ax: The matplotlib axes object.
    - title: The title of the plot, if specified.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    x_val = data[:, 0]
    if x_val.min() >= 1:  # In terms of load factor or not
        load_factor = np.arange(1, len(x_val) + 1) / max(x_val)
    else:
        load_factor = data[:, 0]

    if plot_settings:
        if 'labels' in plot_settings:
            x_label, y_label = plot_settings['labels']
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

        if 'scientific' in plot_settings and plot_settings['scientific']:
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Use scientific notation

        if 'titles' in plot_settings:
            title = plot_settings['titles']
            ax.set_title(title)
        else:
            title = None

        for i in range(1, data.shape[1]):
            values = data[:, i]
            trace = None
            if 'traces' in plot_settings:
                if isinstance(plot_settings['traces'], list):
                    trace = plot_settings['traces'][i - 1] if i - 1 < len(plot_settings['traces']) else None
                else:
                    trace = plot_settings['traces']

            ax.plot(load_factor, values, '-', label=trace, linewidth=0.5, marker=None)

        if 'traces' in plot_settings:
            ax.legend(loc='best')

    return fig, ax, title


import os  
import re  
from datetime import datetime  
import matplotlib.pyplot as plt 

from ..processing.out import *
from ..plots.plots import * 

def single_out_analysis(file_path, **kwargs):
    """
    Perform analysis on a given file and save the resulting figures.

    Parameters:
    -----------
    - file_path (str): The path to the file to be analyzed.
    - minfo (dict, optional): A dictionary containing model information.
    - plot (bool, optional): Whether to create and save plots. Default is False.
    - merge (bool, optional): Whether to merge plots or perform separate analysis. Default is False.
    - **kwargs: Additional keyword arguments.

    Returns:
    --------
    - None

    Raises:
    -------
    - None
    """
    # Get the directory and create an analysis folder if it doesn't exist
    directory = os.path.dirname(file_path)
    analysis_dir = os.path.join(directory, 'analysis/convergence')
    os.makedirs(analysis_dir, exist_ok=True)

    # Get additional parameters
    plot = kwargs.get('plot', False)
    merge = kwargs.get('merge', False)
    minfo = kwargs.get('minfo', {})  # Default to an empty dictionary if not provided

    # Read and parse the file
    try:
        lines = read_file(file_path)  # Replace `dir` with `file_path`
        iter, ncsteps = parse_lines(lines)
    except Exception as e:
        print(f"Error reading or parsing file {file_path}: {e}")
        return

    if plot:
        out_plots(iter, ncsteps, analysis_dir, minfo=minfo, merge=merge)

# ----------------------------------- Utils ---------------------------------- #

def out_plots(iterations, ncsteps, analysis_dir, **kwargs):
    """
    Generate and save convergence plots based on analysis results.

    Parameters:
    -----------
    - iter (int): Number of iterations extracted from the file.
    - ncsteps (int): Number of steps extracted from the file.
    - analysis_dir (str): Path to the directory where figures will be saved.
    - minfo (dict, optional): A dictionary containing model information.
    - merge (bool, optional): Whether to merge plots or perform separate analysis.
    - **kwargs: Additional keyword arguments.

    Returns:
    --------
    - None
    """
    merge = kwargs.get('merge', False)
    minfo = kwargs.get('minfo', {})  

    figures, titles = plot_convergence(iterations, ncsteps, minfo=minfo, merge=merge)
    
    if merge:
        try:
            merged_fig = merge_plots(figures, titles, x_label='X Axis', y_label='Y Axis', title='Merged Convergence Plot')
        except Exception as e:
            print(f"Error merging plots: {e}")

    fig = figures + merged_fig

    for i, fig in enumerate(fig, start=1):
        fig_path = os.path.join(analysis_dir, f'{titles[i-1]}.png')
        if os.path.exists(fig_path):
            os.remove(fig_path)  
        fig.savefig(fig_path)
        plt.close(fig)  

    return figures

def write_to_txt(directory, data):
    
    info_path = os.path.join(directory, 'info.txt')
    if os.path.exists(info_path):
        os.remove(info_path)  # remove the file if it already exists
    
    with open(info_path, 'w') as file:
        model_name = data['Model'][0]
        print(f'MODEL ANALYSIS information for model: {model_name.capitalize()}', file=file)
        print('----------------------------------------------------------\n', file=file)
        
        # Write headers
        headers = '\t'.join(data.keys())
        print(headers, file=file)
        
        # Find the maximum number of rows in the data
        max_rows = max(len(value) for value in data.values())
        
        # Write the data rows
        for i in range(max_rows):
            row = []
            for key in data.keys():
                if i < len(data[key]):
                    row.append(data[key][i])
                else:
                    row.append('')  # Empty string if there is no value for this column
            print('\t'.join(row), file=file)

    return file

def calculate_runtime(filename):
        
    start_time = end_time = None
    pattern_start = r"/DIANA/AP/NL41\s+(\d{2}:\d{2}:\d{2})"
    pattern_end = r"/DIANA/DC/END\s+(\d{2}:\d{2}:\d{2})"

    with open(filename, 'r') as file:
        for line in file:
            match_start = re.search(pattern_start, line)
            if match_start:
                time_str = match_start.group(1)
                start_time = datetime.strptime(time_str, "%H:%M:%S")
            
            match_end = re.search(pattern_end, line)
            if match_end:
                time_str = match_end.group(1)
                end_time = datetime.strptime(time_str, "%H:%M:%S")

    if start_time and end_time:
        difference = end_time - start_time
        runtime = time.strftime('%H:%M:%S', time.gmtime(difference.total_seconds()))
    else:
        runtime = 'NaN'

    return runtime

def model_info(file_path,directory):
    runtime = calculate_runtime(file_path)
    subdirectories = directory.split(os.sep)
    model_name = subdirectories[-1]

    data = {
        'Model': [model_name],
        'Run time': [runtime],
    }
    return data

def read_file(filepath):
    with open(filepath, "r") as fileOUT:
        lines = fileOUT.readlines()
    return lines

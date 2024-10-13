import os

import pandas as pd
import numpy as np

from .crackdetection import *
from ..processing.tabulated import *

def single_tb_analysis(file_path, analysis_info):
    """
    Perform tabulated analysis on a single file.

    Args:
        file_path (str): The path to the tabulated file.
        analysis_info (dict): Information about the analysis.
        plot_settings (dict): Settings for plotting the analysis.

    Returns:
        tuple: A tuple containing two elements:
            - minfo (dict): Information about the analysis results, including the number of elements and nodes.
            - data (dict): The analyzed data.

    """
    file_path = os.path.normpath(file_path)
    directory = os.path.dirname(file_path)
    analysis_dir = os.path.join(directory, 'analysis', 'results') 
    
    try:
        os.makedirs(analysis_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {analysis_dir}: {e}")
    
    df = process_tb(file_path)
    data = analyse_tabulated(df, analysis_info)
        
    minfo = {
        'N Elements': [len(df['Element'].unique())],
        'N Nodes':  [len(df['Node'].unique())]
    }

    data_file_path = os.path.join(analysis_dir, 'analysis_results.txt')
    with open(data_file_path, 'w') as file:
        for key, df_ in data.items():
            file.write(f"DataFrame for: {key}\n\n")
            if isinstance(df_, pd.DataFrame) and len(df_) > 0:
                file.write(df_.to_string())  # Convert DataFrame to string and write
            file.write("\n\n")
    
    return minfo

def analyse_tabulated(df, analysis_info):
    """
    Analyzes tabulated data based on the given analysis information.

    Args:
        df (pandas.DataFrame): The tabulated data to be analyzed.
        analysis_info (dict): Information about the analysis to be performed.

    Returns:
        dict: A dictionary containing the analysis results for each analysis type.

    """
    data = {}
    for analysis in analysis_info:
        vals = []
        
        if 'Relative' in analysis:
            for node in analysis_info[analysis]['Node Nr']:
                u = df[df['Node'] == node][['Step nr.', 'TDtY']]
                vals.append(u)
        
        if 'Mutual' in analysis:
            sets = analysis_info[analysis]['Node Nr']
            typos = analysis_info[analysis]['Reference']
            for set,type in zip(sets, typos):
                merged_df = pd.DataFrame(df['Step nr.'].unique(), columns=['Step nr.'])
                for node,axis in zip(set,type):
                    temp_df = df[df['Node'] == node][['Step nr.',axis]].copy()
                    temp_df.rename(columns={'TDtY': f'{axis} Node {node}'}, inplace=True)
                    merged_df = pd.merge(merged_df, temp_df, on='Step nr.', how='left')
                merged_df.drop(columns=['Step nr.'], inplace=True)
                vals = merged_df

        if 'Crack' in analysis:
            
            if analysis_info[analysis].get('parameters'):
                mesh_size = analysis_info[analysis]['parameters']['mesh']
                
                temp = []
                for step in df['Step nr.'].unique():
                    max_cw = df[df['Step nr.'] == step]['Ecw1'].max()
                    df_filtered = df[(df['Step nr.'] == step) & 
                                    (df['Ecw1'] >= max_cw/10) & 
                                    (pd.notna(df['Element']))][['Element', 'Integration Point', 'X0', 'Y0', 'Ecw1']]                            

                    crack_dict = analyze_cracks(df_filtered, mesh_size)
                    temp.append({'step': step, 'Cracks': crack_dict})
            
                vals = pd.DataFrame(temp)

            else: # Manual mode
                for crack_set in analysis_info[analysis]['EOI']:
                    cw = find_max_cw(crack_set, df)
                    vals.append(cw)       

        if 'Damage' in analysis:
            if analysis_info[analysis].get('parameters'):
                mesh_size = analysis_info[analysis]['parameters']['mesh']
            
                temp = []
                for step in df['Step nr.'].unique():
                    max_cw = df[df['Step nr.'] == step]['Ecw1'].max()
                    df_filtered = df[(df['Step nr.'] == step) & 
                                    (df['Ecw1'] >= max_cw/10) & 
                                    (pd.notna(df['Element']))][['Element', 'Integration Point', 'X0', 'Y0', 'Ecw1']]                            

                    crack_dict = analyze_cracks(df_filtered, mesh_size)
                    psi = compute_damage_parameter(crack_dict)
                    temp.append({'step': step, 'psi': psi})
                vals = pd.DataFrame(temp)
            else:
                for crack_set in analysis_info[analysis]['parameters']['cracks']: 
                    c_w = compute_damage_parameter_manual(df, crack_set)
                    vals.append(c_w)
        
        data[analysis] = vals
    return data

def tb_plot(file_path, analysis_info, plot_settings = None):
    directory = os.path.dirname(file_path)
    analysis_dir = os.path.join(directory, 'analysis', 'results') 
    
    if plot_settings:
        figures, titles = plot_analysis(data, analysis_info, plot_settings)

        for i, fig in enumerate(figures, start=1):
            fig_path = os.path.join(analysis_dir, f'{titles[i-1]}.png')
            if os.path.exists(fig_path):
                os.remove(fig_path)  
            fig.savefig(fig_path)
            close()

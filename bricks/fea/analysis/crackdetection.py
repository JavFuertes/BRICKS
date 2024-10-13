import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import Tuple, Dict, List

def analyze_cracks(df_filtered: pd.DataFrame, mesh_size: float) -> Dict[str, Dict]:
    """
    Main function to analyze cracks in the data.
    
    Args:
        df (pd.DataFrame): Original dataframe containing the crack data.
        mesh_size (float): The mesh size used for calculating the distance threshold.
    
    Returns:
        Dict[str, Dict]: Dictionary containing the crack properties.
    """
    points = df_filtered[['X0', 'Y0']].values
    dist_matrix = distance_matrix(points, points)
    
    d_threshold = np.sqrt(2 * (mesh_size / 2)**2)  # Max diagonal distance between int points in quadratic mesh
    n_components, labels = find_connected_components(dist_matrix, d_threshold)
    
    df_filtered['Component'] = labels
    cracks = calculate_crack_properties(df_filtered, n_components, d_threshold)
    
    return cracks

def find_connected_components(dist_matrix: np.ndarray, d_threshold: float) -> Tuple[int, np.ndarray]:
    """
    Find connected components in a distance matrix based on a distance threshold.

    Args:
        dist_matrix (np.ndarray): A 2D array representing the distance matrix.
        d_threshold (float): The distance threshold to determine connectivity.

    Returns:
        Tuple[int, np.ndarray]: A tuple containing:
            - n_components (int): The number of connected components found.
            - labels (np.ndarray): An array where each element represents the component label of the corresponding node.
    """
    connectivity = dist_matrix <= d_threshold
    connectivity_sparse = csr_matrix(connectivity)
    n_components, labels = connected_components(csgraph=connectivity_sparse, directed=False, return_labels=True)
    return n_components, labels

def calculate_crack_properties(df_filtered: pd.DataFrame, n_components: int, d_threshold: float) -> Dict[str, Dict]:
    """
    Calculate properties of cracks from filtered DataFrame.

    Args:
        df_filtered (pd.DataFrame): Filtered DataFrame containing crack data with columns 'Component', 'X0', 'Y0', 'Element', and 'Ecw1'.
        n_components (int): Number of components to analyze.
        d_threshold (float): Distance threshold for determining significant crack length.

    Returns:
        Dict[str, Dict]: A dictionary containing crack properties for each component that meets the criteria.
    """  
    cracks = {}
    
    for component in range(n_components):
        component_data = df_filtered[df_filtered['Component'] == component]
        component_points = component_data[['X0', 'Y0']].values
        component_elements = component_data['Element'].unique()
        
        if len(component_points) > 1:
            component_dist_matrix = distance_matrix(component_points, component_points)
            crack_length = np.max(component_dist_matrix)
            components = component_elements.tolist()
            max_mean_crack_width = component_data['Ecw1'].mean()

            if len(components) > 2 and crack_length > 3 * d_threshold:  # Crack Length should go through a minimum of 3 IntPoints
                cracks[f'Crack {component}'] = {
                    'length': crack_length,
                    'max_mean_width': max_mean_crack_width,
                    'component': components,
                }

    return cracks

def compute_damage_parameter(crack_dict) -> float:
    """
    Compute the damage parameter based on the given dataframe and damage dictionary.

    Parameters:
    - df: The dataframe containing the data.
    - damage: A dictionary containing the damage information.

    Returns:
    - The computed damage parameter.

    """
    n_c = 0
    c_w_n = []
    c_w_d = []
    
    for crack in crack_dict.values():
        n_c += 1
        
        c_w = crack['max_mean_width']
        l_c = crack['length']
        
        c_w_n += [c_w**2 * l_c]
        c_w_d += [c_w * l_c]
        
    c_w = sum(c_w_n) / sum(c_w_d) if (len(c_w_d) != 0) and (sum(c_w_d) != 0) else 0
    psi = 2 * n_c**0.15 * c_w**0.3
    return psi

# ---------------- Manual model evaluation of damage parameter --------------- #
def compute_damage_parameter_manual(df, damage: dict = None) -> float:
    """
    Compute the damage parameter based on the given dataframe and damage dictionary.

    Parameters:
    - df: The dataframe containing the data.
    - damage: A dictionary containing the damage information.

    Returns:
    - The computed damage parameter.

    """
    n_c = 0
    c_w_n = []
    c_w_d = []
    el_size = damage['element_size']

    for elements in damage['EOI']:
        n_c += 1
        n_el = find_nel(df, elements)
        l_c = el_size * n_el
        c_w_df = find_max_cw(elements, df)
        c_w = c_w_df['Ecw1']
        c_w_n += [c_w**2 * l_c]
        c_w_d += [c_w * l_c]
        
    c_w = sum(c_w_n) / sum(c_w_d) if len(c_w_d) != 0 else 0
    c_w_df['Ecw1'] = c_w

    psi = 2 * n_c**0.15 * c_w**0.3
    c_w_df['psi'] = psi
    c_w_df = c_w_df.fillna(0)
    c_w_df.drop(columns=['Ecw1'], inplace=True)
    return c_w_df

def find_nel(df,elements):
    filtered_df = df[df['Element'].isin(elements)]
    n_el = filtered_df.groupby('Step nr.')['Ecw1'].apply(lambda x: x.dropna().count())
    return n_el

def find_mean_cw(elements_of_interest,df):
    filtered_df = df[df['Element'].isin(elements_of_interest)]
    grouped = filtered_df.groupby(['Step nr.', 'Element'])['Ecw1'].mean().reset_index()
    final_avg = grouped.groupby('Step nr.')['Ecw1'].mean().reset_index()
    return final_avg

def find_max_cw(elements_of_interest,df):
    filtered_df = df[df['Element'].isin(elements_of_interest)]
    grouped = filtered_df.groupby(['Step nr.', 'Element'])['Ecw1'].max().reset_index()
    final_avg = grouped.groupby('Step nr.')['Ecw1'].max().reset_index()
    return final_avg



import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse import csr_matrix


def calculate_distances(points):
    """Calculates the pairwise distance matrix for given points."""
    return distance_matrix(points, points)

def find_connected_components(dist_matrix, d_threshold):
    """Finds connected components based on a distance threshold."""
    connectivity = dist_matrix <= d_threshold
    connectivity_sparse = csr_matrix(connectivity)
    n_components, labels = connected_components(csgraph=connectivity_sparse, directed=False)
    return n_components, labels

def calculate_crack_properties(df_filtered, n_components):
    """Calculates the crack width and length for each component."""
    
    cracks = {}

    for component in range(n_components):
        component_points = df_filtered[df_filtered['Component'] == component][['X0', 'Y0']].values
        component_elements = df_filtered[df_filtered['Component'] == component]['Element'].unique()
        
        if component_points.shape[0] > 1:
            component_dist_matrix = distance_matrix(component_points, component_points)
            max_distance = np.max(component_dist_matrix)
            crack_length = max_distance
        else:
            crack_length = 0
        average_crack_width = df_filtered[df_filtered['Component'] == component]['Ecw1'].mean()
        
        crack_info = {f'Crack {component}': {'length': crack_length,
                                            'average_width': average_crack_width,
                                            'component': component,
                                            'elements': component_elements.tolist(),  # Convert to list for JSON serialization compatibility
                                            }}
        cracks.update(crack_info)
            
    return cracks

def analyze_cracks(df_filtered, d_threshold):
    """Main function to analyze cracks in the data.
        
    Arguments:
    df_filtered (pandas.DataFrame): Filtered dataframe containing the crack data.
    d_threshold (float): The distance threshold between adjacent entries. 
                         If the outputted values have been set to the nodes, d_threshold can be set to 1mm.
                         If the outputted values have been set to the integration points, 
                         the maximum distance will be in the diagonal based on the element size and order.
                         For quadrilateral linear elements, d_threshold == ((l/2)**2)**(1/2).
                         
    Returns:
    pandas.DataFrame: DataFrame containing the crack properties.
    """
    points = df_filtered[['X0', 'Y0']].values
    dist_matrix = calculate_distances(points)
    n_components, labels = find_connected_components(dist_matrix, d_threshold)
    df_filtered['Component'] = labels
    cracks = calculate_crack_properties(df_filtered, n_components)
    
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
        
        c_w = crack['average_width']
        l_c = crack['length']
        
        c_w_n += [c_w**2 * l_c]
        c_w_d += [c_w * l_c]
        
    c_w = sum(c_w_n) / sum(c_w_d) if (len(c_w_d) != 0) and (sum(c_w_d) != 0) else 0
    psi = 2 * n_c**0.15 * c_w**0.3
    return psi

def eval_dl(psi):
    
    psi_thresholds = [1,1.5,2.5,3.5,float('inf')]
    dl = [0,1,2,3,4]
    for i,threshold in enumerate(psi_thresholds):
        if psi > threshold:
            continue
        else:
            break
    return dl[i]

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
        c_w_df = find_mean_cw(elements, df)
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



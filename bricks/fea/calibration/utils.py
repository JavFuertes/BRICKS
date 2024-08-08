import traceback
import numpy as np
import pandas as pd


from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def read_file(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()
    return lines

def process_data_row(data_string):
    """
    Process a data row string and convert it into a list of floating-point values.

    Args:
        data_string (str): The input data string to be processed.

    Returns:
        list: A list of floating-point values extracted from the data string.

    Raises:
        ValueError: If the string cannot be converted to a float.

    """
    count = 0
    values = []

    while count < len(data_string):
        string = data_string[count:count+10]
        string = string.strip()
        if string:
            try:
                value = float(string)
            except ValueError as e:
                print(f"Error converting string to float: {string} - {e}")
                value = np.nan
        else:
            value = np.nan
        values.append(value)
        count += 11  # Move past the 10 spaces and the character after it

    return values

def process_tb(file_path):
    """
    Process a tabulated file and return a pandas DataFrame.

    Args:
        file_path (str): The path to the tabulated file.

    Returns:
        pandas.DataFrame: The processed data as a DataFrame.

    Raises:
        Exception: If an error occurs during processing.

    """
    data_list = []
    info = {}
    errors = []

    step_n = None
    nodes = None
    intpnt = None
    values = False
    coordinates = False

    lines = read_file(file_path)

    for lin_num, line in enumerate(lines):
        try:
            words = line.split()

            if not words:  # Skip empty lines
                values = False
                coordinates = False
                continue    

            if words[:2] == ['Analysis', 'type']:  # Extract analysis type
                atype = words[-1]
                info['Analysis type'] = atype

                if lines[lin_num+1].split()[:2] == ['Step', 'nr.']:
                    step_n = lines[lin_num+1].split()[-1]
                    info['Step nr.'] = int(step_n)
                if lines[lin_num+2].split()[:2] == ['Load', 'factor']:
                    lf = lines[lin_num+2].split()[-1]
                    info['Load factor'] = float(lf)    
                continue

            if words[:2] == ['Elmnr', 'Intpt']: 
                coord = ['X0','Y0','Z0']
                if words[-3:] == coord:
                    variables = words[2:-3]
                    coordinates = True
                    ncoord = len([equal for equal in words if equal in set(coord)])
                else: 
                    variables = words[2:]
                values = True
                intpnt = True
                nodes = False
                continue
            elif words[0] == 'Nodnr':
                coord = ['X0','Y0']
                if words[-2:] == coord:
                    variables = words[1:-2]
                    coordinates = True
                    ncoord = len([equal for equal in words if equal in set(coord)])
                else: 
                    variables = words[1:]
                values = True 
                nodes = True
                intpnt = False
                continue

            if values and intpnt:  # Improve implementation not very resilient
                
                if line[1:6].strip().isdigit():
                    elmn_n = int(line[1:6])
                nodn_n = int(line[7:12])
                
                vals = {'Element': elmn_n, 'Integration Point': nodn_n}

                if coordinates:
                    count = ncoord*10 + (ncoord-1) + 3
                    data_string = line[15:-count]
                else:
                    data_string = line[15:]
                data = process_data_row(data_string)
                for j, var in enumerate(variables):
                    vals[var] = float(data[j])

                coord_val = line[-count+2:]  ## +2 Will give issues with negative coordinates, +1 gives issues with exponents
                x,y,z = process_data_row(coord_val)
                coord_vals = {'X0': x, 'Y0': y,'Z0': z}
                record = {**info,**coord_vals,**vals}
                data_list.append(record)
                continue

            if values and nodes:  # Improve implementation not very resilient
                nodn_n = int(line[1:6])
                vals = {'Node': nodn_n}

                if coordinates:
                    count = ncoord*10 + (ncoord-1) + 3
                    data_string = line[9:-count]
                else:
                    data_string = line[9:]

                data = process_data_row(data_string)
                
                for j, var in enumerate(variables):
                    vals[var] = float(data[j])
                
                coord_val = line[-count+3:]
                x,y = process_data_row(coord_val)
                coord_vals = {'X0': x, 'Y0': y}
                
                record = {**info, **coord_vals, **vals}
                data_list.append(record)
                continue
            
        except Exception as e:
            errors.append((lin_num, str(e)))
            traceback.print_exc()
            if len(errors) >= 1:
                print("Error limit reached, stopping processing.")
                return errors

    df = pd.DataFrame(data_list)
    return df

def find_connected_components(dist_matrix, d_threshold):
    """Finds connected components based on a distance threshold.

    Args:
        dist_matrix (numpy.ndarray): The distance matrix.
        d_threshold (float): The distance threshold.

    Returns:
        int: The number of connected components.
        numpy.ndarray: The labels indicating the component membership of each node.
    """
    """Finds connected components based on a distance threshold."""
    connectivity = dist_matrix <= d_threshold
    connectivity_sparse = csr_matrix(connectivity)
    n_components, labels = connected_components(csgraph=connectivity_sparse, directed=False)
    return n_components, labels

def calculate_crack_properties(df_filtered, n_components):
    """Calculates the crack width and length for each component.
    Args:
        df_filtered (DataFrame): The filtered DataFrame containing the crack data.
        n_components (int): The number of components.
    Returns:
        dict: A dictionary containing the crack properties for each component. The keys are the crack indices and the values are dictionaries with the following properties:
            - length (float): The length of the crack.
            - average_width (float): The average width of the crack.
            - component (int): The component index.
            - elements (list): A list of unique element IDs associated with the crack.
    """
    
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
    """
    Analyzes cracks in a filtered dataframe.
    Parameters:
    - df_filtered (DataFrame): The filtered dataframe containing crack data.
    - d_threshold (float): The distance threshold for determining connected components.
    Returns:
    - cracks (DataFrame): The dataframe containing calculated crack properties.
    """
    
    points = df_filtered[['X0', 'Y0']].values
    dist_matrix = distance_matrix(points, points)
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
        
    c_w = sum(c_w_n) / sum(c_w_d) if len(c_w_d) != 0 else 0
    
    psi = 2 * n_c**0.15 * c_w**0.3
    return psi

import re
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def read_file(filepath: str) -> list:
    """
    Read the contents of a file and return its lines.

    Args:
        filepath (str): The path to the file to be read.

    Returns:
        list: A list of strings, where each string is a line from the file.
    """
    with open(filepath, "r") as file:
        return file.readlines()

def parse_lines(lines: list) -> tuple:
    """
    Parse the lines from a DIANA output file and extract iteration information.

    Args:
        lines (list): A list of strings, each representing a line from the file.

    Returns:
        tuple: A tuple containing two elements:
            - dict: A dictionary with phase information and iteration data.
            - list: A list of step numbers that did not converge.
    """
    Iterations = {}
    NoConvergenceSteps = []
    CurrentStepIndex = 0
    TotalStepIndex = 0
    ener_norm_temp = 0.0
    force_norm_temp = 0.0
    disp_norm_temp = 0.0

    PhaseYN = 0
    # Check that lines contains information
    for line in lines:
        fileOUT_string = line.split()

        if len(fileOUT_string) == 0:
            continue

        if (fileOUT_string[0] == '/DIANA/AP/PH40') and (fileOUT_string[5] == 'BEGIN'):
            # Turn on the flag and read the Phase number in the next row
            PhaseYN = 1

        # Check if the current row contains the start of a PHASE
        if (fileOUT_string[0] == 'PHASE') and (PhaseYN == 1):
            #Save phase number and make dictionary for the phase
            Temporary = fileOUT_string[1]
            KeyLabel = 'Phase ' 
            CurrentPhase = KeyLabel
            Iterations[KeyLabel] = {'Plastic_int': [], 'Crack_points': [], 'no_iter': [],
                                    "force_norm": [], "disp_norm": [], "energy_norm": [],
                                    "force_limit": 0, "disp_limit": 0, "energy_limit": 0}

        # Check for step initiation
        if (fileOUT_string[0] == 'STEP') and (fileOUT_string[2] == 'INITIATED:'):
            if PhaseYN == 0:
                KeyLabel = 'Phase '
                CurrentPhase = KeyLabel
                Iterations[KeyLabel] = {'Plastic_int': [], 'Crack_points': [], 'no_iter': [],
                                        "force_norm": [], "disp_norm": [], "energy_norm": [],
                                        "force_limit": 0, "disp_limit": 0, "energy_limit": 0}
                PhaseYN = 2

            CurrentStepIndex = int(fileOUT_string[1])
            TotalStepIndex += 1
            NoDisplConv = False
            NoForceConv = False
            NoEnerConv = False

        if len(fileOUT_string) > 7:
            if (fileOUT_string[3] == 'DISPLACEMENT') and (fileOUT_string[7] == 'TOLERANCE'):
                Expctd_displ_norm = float(fileOUT_string[9])
                Iterations[KeyLabel]["disp_limit"] = Expctd_displ_norm

            if (fileOUT_string[3] == 'FORCE') and (fileOUT_string[7] == 'TOLERANCE'):
                Expctd_force_norm = float(fileOUT_string[9])
                Iterations[KeyLabel]["force_limit"] = Expctd_force_norm

            if (fileOUT_string[3] == 'ENERGY') and (fileOUT_string[7] == 'TOLERANCE'):
                Expctd_ener_norm = float(fileOUT_string[9])
                Iterations[KeyLabel]["energy_limit"] = Expctd_ener_norm

        if (fileOUT_string[0] == 'RELATIVE') and (fileOUT_string[1] == 'DISPLACEMENT'):
            displ_norm = float(fileOUT_string[4])
            if Expctd_displ_norm < displ_norm:
                NoDisplConv = True
            else:
                disp_norm_temp = displ_norm

        if (fileOUT_string[0] == 'RELATIVE') and (fileOUT_string[1] == 'OUT'):
            force_norm = float(fileOUT_string[6])
            if Expctd_force_norm < force_norm:
                NoForceConv = True
            else:
                force_norm_temp = force_norm

        if (fileOUT_string[0] == 'RELATIVE') and (fileOUT_string[1] == 'ENERGY'):
            ener_norm = float(fileOUT_string[4])
            if Expctd_ener_norm < ener_norm:
                NoEnerConv = True
            else:
                ener_norm_temp = ener_norm

        if (fileOUT_string[0] == 'TOTAL' and fileOUT_string[1] == 'MODEL'):
            Temporary = int(fileOUT_string[2])
            if len(fileOUT_string) <= 8:
                Iterations[CurrentPhase]["Plastic_int"].append(Temporary)
            else:
                Iterations[CurrentPhase]["Crack_points"].append(Temporary)

        if (fileOUT_string[0] == 'STEP') and (fileOUT_string[2] == 'TERMINATED,'):
            if fileOUT_string[3] == 'NO':
                n_iter = re.findall(r'\d+', fileOUT_string[5])
                Temporary = int(n_iter[0])
                NoConvergenceSteps.append(CurrentStepIndex)
                with open("Convergence.txt", "a") as a_file:
                    a_file.write(f"Non-converged step number: {CurrentStepIndex}\n\n")
                    if NoDisplConv:
                        a_file.write("No displacement convergence found\n")
                        a_file.write(f"Relative displacement variation at non-convergence: {displ_norm}\n")
                        Iterations[CurrentPhase]["energy_norm"].append(displ_norm)
                    if NoForceConv:
                        a_file.write("No Force convergence found\n")
                        a_file.write(f"Relative Out-of-Balance force at non-convergence: {force_norm}\n")
                        Iterations[CurrentPhase]["energy_norm"].append(force_norm)
                    if NoEnerConv:
                        a_file.write("No Energy convergence found\n")
                        a_file.write(f"Relative Energy variation at non-convergence: {ener_norm}\n\n")
                        Iterations[CurrentPhase]["energy_norm"].append(ener_norm)
            else:
                Temporary = int(fileOUT_string[5])
                Iterations[CurrentPhase]["energy_norm"].append(ener_norm_temp)
                Iterations[CurrentPhase]["disp_norm"].append(disp_norm_temp)
                Iterations[CurrentPhase]["force_norm"].append(force_norm_temp)
            Iterations[CurrentPhase]["no_iter"].append(Temporary)

    return Iterations, NoConvergenceSteps

def process_tb_chunk(file_path: str, chunk_size: int = 100000) -> pd.DataFrame:
    """
    Process a tabulated file in chunks and return a pandas DataFrame.

    Args:
        file_path (str): The path to the tabulated file.
        chunk_size (int): Number of lines to process at a time. Default is 100000.

    Returns:
        pd.DataFrame: The processed data as a DataFrame.

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

    def line_generator(filepath):
        with open(filepath, 'r') as file:
            for line in file:
                yield line

    lines = line_generator(file_path)
    lin_num = -1  # Initialize line number

    for line in lines:
        lin_num += 1
        try:
            words = line.split()

            if not words:  # Skip empty lines
                values = False
                coordinates = False
                continue    

            if words[:2] == ['Analysis', 'type']:  # Extract analysis type
                atype = words[-1]
                info['Analysis type'] = atype

                next_line = next(lines)
                lin_num += 1
                if next_line.split()[:2] == ['Step', 'nr.']:
                    step_n = next_line.split()[-1]
                    info['Step nr.'] = int(step_n)
                next_line = next(lines)
                lin_num += 1
                if next_line.split()[:2] == ['Load', 'factor']:
                    lf = next_line.split()[-1]
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

            if values and intpnt:  # Process Integration Points
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

                coord_val = line[-count+2:]  
                x,y,z = process_data_row(coord_val)
                coord_vals = {'X0': x, 'Y0': y,'Z0': z}
                record = {**info,**coord_vals,**vals}
                data_list.append(record)
                if len(data_list) >= chunk_size:
                    yield pd.DataFrame(data_list)
                    data_list = []
                continue

            if values and nodes:  # Process Nodes
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
                if len(data_list) >= chunk_size:
                    yield pd.DataFrame(data_list)
                    data_list = []
                continue
            
        except Exception as e:
            errors.append((lin_num, str(e)))
            traceback.print_exc()
            if len(errors) >= 1:
                print("Error limit reached, stopping processing.")
                return errors

    if data_list:
        yield pd.DataFrame(data_list)

def process_data_row(data_string: str) -> list:
    """
    Process a data row string and convert it into a list of floating-point values.

    Args:
        data_string (str): The input data string to be processed.

    Returns:
        list: A list of floating-point values extracted from the data string.
    """
    return [float(data_string[i:i+10].strip() or np.nan) for i in range(0, len(data_string), 11)]

def process_tb(file_path: str) -> pd.DataFrame:
    """
    Process the entire tabulated file and return a consolidated DataFrame.

    Args:
        file_path (str): The path to the tabulated file.

    Returns:
        pd.DataFrame: The final processed data as a DataFrame.
    """
    return pd.concat(process_tb_chunk(file_path) for _ in iter(int, 1))

def find_connected_components(dist_matrix: np.ndarray, d_threshold: float) -> tuple:
    """
    Find connected components based on a distance threshold.

    Args:
        dist_matrix (np.ndarray): Distance matrix between points.
        d_threshold (float): Distance threshold for connectivity.

    Returns:
        tuple: Number of components and labels for each point.
    """
    connectivity = dist_matrix <= d_threshold
    connectivity_sparse = csr_matrix(connectivity)
    return connected_components(csgraph=connectivity_sparse, directed=False)

def calculate_crack_properties(df_filtered: pd.DataFrame, n_components: int) -> dict:
    """
    Calculate the crack width and length for each component.

    Args:
        df_filtered (pd.DataFrame): Filtered DataFrame containing crack data.
        n_components (int): Number of connected components.

    Returns:
        dict: A dictionary containing properties of each crack.
    """
    cracks = {}
    for component in range(n_components):
        component_points = df_filtered[df_filtered['Component'] == component][['X0', 'Y0']].values
        component_elements = df_filtered[df_filtered['Component'] == component]['Element'].unique()
        
        crack_length = np.max(distance_matrix(component_points, component_points)) if component_points.shape[0] > 1 else 0
        average_crack_width = df_filtered[df_filtered['Component'] == component]['Ecw1'].mean()
        
        cracks[f'Crack {component}'] = {
            'length': crack_length,
            'average_width': average_crack_width,
            'component': component,
            'elements': component_elements.tolist(),
        }
            
    return cracks

def analyze_cracks(df_filtered: pd.DataFrame, d_threshold: float) -> dict:
    """
    Analyze cracks in the data.

    Args:
        df_filtered (pd.DataFrame): Filtered DataFrame containing crack data.
        d_threshold (float): Distance threshold for connectivity.

    Returns:
        dict: A dictionary containing properties of all cracks.
    """
    points = df_filtered[['X0', 'Y0']].values
    dist_matrix = distance_matrix(points, points)
    n_components, labels = find_connected_components(dist_matrix, d_threshold)
    df_filtered['Component'] = labels
    return calculate_crack_properties(df_filtered, n_components)

def compute_damage_parameter(crack_dict: dict) -> float:
    """
    Compute the damage parameter based on the given crack dictionary.

    Args:
        crack_dict (dict): A dictionary containing the crack information.

    Returns:
        float: The computed damage parameter.
    """
    n_c = len(crack_dict)
    c_w_n = [(crack['average_width']**2 * crack['length']) for crack in crack_dict.values()]
    c_w_d = [(crack['average_width'] * crack['length']) for crack in crack_dict.values()]
    
    c_w = sum(c_w_n) / sum(c_w_d) if c_w_d and sum(c_w_d) != 0 else 0
    return 2 * n_c**0.15 * c_w**0.3
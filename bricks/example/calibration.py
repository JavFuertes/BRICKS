## Imports 
import os
import traceback
import numpy as np
import pandas as pd
import torch
import time
import pickle
import itertools
import logging
import re

from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.stats import norm
from scipy.stats import qmc

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

from sklearn.preprocessing import MinMaxScaler  
import torch.nn.functional as F  

# ---------------------------------------------------------------------------- #
#                                   Utils Class                                #
# ---------------------------------------------------------------------------- #
class MaterialParameters:
    def __init__(self):
        # Initialize material parameters with their mean and standard deviation
        self.parameters = {
            'fmy': (17.7, 6.73),
            'Emy': (9348, 3271),
            'fmx': (11.0, 2.53),
            'Emx': (5470, 547),
            'fw': (0.43, 0.07),
            'fx2': (1.22, 0.11),
            'fx3': (0.76, 0.21),
            'G': (1252, 200),
            'fv': (0.35, 0),
            'u': (0.67, 0)
        }

    def sample_parameter(self, param_name, percentile):
        """
        Samples the xth percentile value from a normal distribution for a given parameter.
        
        Args:
            param_name (str): The name of the parameter to sample (e.g., 'fw').
            percentile (float): The percentile to sample (0-100).
        
        Returns:
            float: The value at the given percentile in the distribution.
        """
        confidence = percentile / 100
        mean, std_dev = self.parameters[param_name]
        return norm.ppf(confidence, loc=mean, scale=std_dev)

    def get_bounds(self, param_name, percentile):
        """
        Generates the lower and upper bounds for a given parameter based on a confidence interval.
        
        Args:
            param_name (str): The name of the parameter (e.g., 'fw').
            confidence (float): The confidence level for the bounds (e.g., 0.95).
        
        Returns:
            tuple: A tuple containing the lower and upper bounds for the parameter.
        """
        confidence = percentile / 100
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        lower_bound = self.sample_parameter(param_name, lower_percentile)
        upper_bound = self.sample_parameter(param_name, upper_percentile)
        return lower_bound, upper_bound

    @staticmethod
    def tensile_strength(fw):
        """
        Calculates the tensile strength based on the provided fw value.
        
        Args:
            fw (float): The value of fw.
        
        Returns:
            float: The tensile strength.
        """
        return 0.8 * fw

    @staticmethod
    def tensile_fracture_energy(fw, mode):
        """
        Calculates the tensile fracture energy based on the provided fw value and mode.
        
        Args:
            fw (float): The value of fw.
            mode (str): The mode, either 'mortar' or 'brick'.
        
        Returns:
            float: The tensile fracture energy.
        """
        if mode == 'mortar':
            fm_mortar = fw / 0.036  # Mpa
            GfI = 0.025 * (fm_mortar / 10) ** 0.7  # N/mm
        elif mode == 'brick':
            GfI = 10 * 0.16 * MaterialParameters.tensile_strength(fw)  # N/mm
        return GfI

    @staticmethod
    def compressive_fracture_energy(fmy):
        """
        Calculates the compressive fracture energy based on the provided fmy value.
        
        Args:
            fmy (float): The value of fmy.
        
        Returns:
            float: The compressive fracture energy.
        """
        GfC = 3.09 * fmy
        return GfC

    @staticmethod
    def shear_fracture_energy(fm):
        """
        Calculates the shear fracture energy based on the provided fm value.
        
        Args:
            fm (float): The value of fm.
        
        Returns:
            float: The shear fracture energy.
        """
        Gft_m = 0.025 * (fm / 10) ** 0.7  # N/mm
        GfII = 10 * Gft_m  # N/mm
        return GfII

def read_file(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()
    return lines

def parse_lines(lines):
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

def process_tb_chunk(file_path, chunk_size=100000):
    """
    Process a tabulated file and return a pandas DataFrame.

    Args:
        file_path (str): The path to the tabulated file.
        chunk_size (int): Number of lines to process at a time.

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
    dfs = []
    for chunk_df in process_tb_chunk(file_path):  # Call the correct function
        dfs.append(chunk_df)
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df

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
                                            'elements': component_elements.tolist(),  
                                            }}
        cracks.update(crack_info)
            
    return cracks

def analyze_cracks(df_filtered, d_threshold):
    """Main function to analyze cracks in the data."""
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
        
    c_w = sum(c_w_n) / sum(c_w_d) if (len(c_w_d) != 0) and (sum(c_w_d) != 0) else 0
    psi = 2 * n_c**0.15 * c_w**0.3

    return psi

def generate_screenshot(screenshot_dir, run):
    script_config = {
            'results': [
                {
                    'component': 'Ecw1',
                    'result': 'Crack-widths',
                    'type': 'Element',
                    'location': 'mappedintpnt',
                    'limits': [0, 1, 2, 3, 4, 5, 10, 15, 20]
                }
            ],
            'script': {
                'analysis': "NLA",
                'load_cases': ['Building', 'Sub Deformation'],
                'load_steps': [1, 360],
                'load_factors_init': [1, 1/360],
                'snapshots': 1,
                'view_settings': {
                    'view_point': [0, 0, 25.0, 0, 1, 0, 5.2, 3.1, 5.5e-17, 19, 3.25],
                    'title_font_size': 36,
                    'legend_font_size': 34,
                    'annotation_font_size': 28
                }
            }
        }

    try:
        # Set view and save the screenshot using the appropriate directory
        addViewSetting("PY")
        setActiveViewSetting("PY")
        saveViewPoint("plot_vp", script_config['script']['view_settings']['view_point'])
        setViewPoint("plot_vp")

        for result in script_config['results']:
            step_name = f"Load-step {int(sum(script_config['script']['load_steps']))}"
            lf_value = script_config['script']['load_factors_init'][-1] * script_config['script']['load_steps'][-1]
            formatted_lf_value = format(lf_value, '.4f').rstrip('0').rstrip('.')
            if formatted_lf_value.isdigit():
                lf_name = f"Load-factor {format(lf_value, '.4f')}"
            else:
                lf_name = f"Load-factor {formatted_lf_value}"
            lc_name = script_config['script']['load_cases'][-1]

            if result['type'] == 'Node':
                out = 'Output Diana'
            elif result['type'] == 'Element':
                out = 'Monitor Diana'

            last = f"{step_name}, {lf_name}, {lc_name}"
            showView("RESULT")
            setResultCase([script_config['script']['analysis'], out, last])
            selectResult(result)
            setResultPlot( "contours" )

            # Apply view settings
            setViewSettingValue("PY", "RESULT/TITLE/RANGE", "VISIBLE")
            setViewSettingValue("PY", "RESULT/TITLE/POSIT", "0.0100000 0.990000")
            setViewSettingValue("PY", "RESULT/TITLE/FONT/SIZE", script_config['script']['view_settings']['title_font_size'])
            setViewSettingValue("PY", "RESULT/LEGEND/FONT/SIZE", script_config['script']['view_settings']['legend_font_size'])
            setViewSettingValue("PY", "RESULT/LEGEND/ANNFNT/SIZE", script_config['script']['view_settings']['annotation_font_size'])
            setViewSettingValue("PY", "RESULT/LABEL/EXTREM/LEVEL", "OFF")
            setViewSettingValue("PY", "RESULT/TITLE/BORDER/BACK", False)
            setViewSettingValue("PY", "RESULT/TITLE/BORDER/FRAME", False)
            
            setViewSettingValue("PY", "RESULT/LEGEND/LBLFMT", "AUTO")
            setViewSettingValue("PY", "RESULT/LEGEND/LBLPRC", 2)
            setViewSettingValue("PY", "RESULT/LEGEND/FONT/FAMILY", "ARIAL")
            setViewSettingValue("PY", "RESULT/LEGEND/ANNOTA", "RELFRQ")
            setViewSettingValue("PY", "RESULT/LEGEND/FONT/COLOR", [31, 30, 29, 255])
            setViewSettingValue("PY", "RESULT/LEGEND/ANNFNT/COLOR", [68, 68, 68, 255])
            setViewSettingValue("PY", "RESULT/LEGEND/BORDER/BACK", False)
            setViewSettingValue("PY", "RESULT/LEGEND/BORDER/FRAME", False)
            
            setViewSettingValue("PY", "RESULT/EDGES/RENDEF", "FRE")
            setViewSettingValue("PY", "RESULT/DEFORM/MODE", "ABSOLU")
            setViewSettingValue("PY", "RESULT/DEFORM/ABSOLU/FACTOR", 5)
            setViewSettingValue("PY", "RESULT/DEFORM/DEFX", True)
            setViewSettingValue("PY", "RESULT/DEFORM/DEFY", True)
            setViewSettingValue("PY", "RESULT/DEFORM/DEFZ", True)
            setViewSettingValue("PY", "RESULT/CONTOU/BNDCLR/MAXCLR", [255, 0, 255, 255])
            setViewSettingValue("PY", "RESULT/CONTOU/BNDCLR/MINCLR", [0, 255, 255, 255])

            values = result['limits']
            setViewSettingValue("PY", "RESULT/CONTOU/LEVELS", "SPECIF")
            setViewSettingValue("PY", "RESULT/CONTOU/LEGEND", "DISCRE")
            setViewSettingValue("PY", "RESULT/CONTOU/SPECIF/VALUES", values)
            setViewSettingValue("PY", "RESULT/CONTOU/AUTRNG", "LIMITS")
            setViewSettingValue("PY", "RESULT/CONTOU/LIMITS/MAXVAL", values[-1])
            setViewSettingValue("PY", "RESULT/CONTOU/LIMITS/MINVAL", values[0])
            setViewSettingValue("PY", "RESULT/CONTOU/LIMITS/BOUNDS", "CLAMP")

            # Save the screenshot        
            image_path = os.path.join(screenshot_dir, f"screenshot_{run[0]}_{run[1]}.png")
            saveImage(image_path, 1800, 1100, 1)

    except Exception as e:
            print(f"An error occurred while generating screenshots: {e}")
            traceback.print_exc()

def update_model_parameters(mat_param):
    steps = 360
    lf = 1/steps
    setAnalysisCommandDetail("NLA", "Structural nonlinear", "EXECUT(1)/LOAD/STEPS/EXPLIC/SIZES", "1")
    setAnalysisCommandDetail("NLA", "Structural nonlinear", "EXECUT(2)/LOAD/STEPS/EXPLIC/SIZES", f"{lf} ({steps})")


    young_modulus = mat_param[0]
    tensile_limit = mat_param[1]
    tensile_fracture_energy = mat_param[2]
    setParameter("MATERIAL", "Masonry TS", "LINEAR/ELASTI/YOUNG", young_modulus)
    setParameter("MATERIAL", "Masonry TS", "TENSIL/TENSTR", tensile_limit)
    setParameter("MATERIAL", "Masonry TS", "TENSIL/GF1", tensile_fracture_energy)


# ---------------------------------------------------------------------------- #
#                                   Wall                                       #
# ---------------------------------------------------------------------------- #
class WALL:
    def __init__(self, model_paths, an_paths, targets, bounds):
        """
        Initializes the WALL class with multiple models and targets.
        """
        num_params = len(bounds)
        num_pred = len(targets)

        self.config = {
            "model_directories": model_paths,
            "directories": an_paths,
            "save_directory_base": os.path.join(os.path.dirname(model_paths[0]), '!Results'),
            "targets": targets,
            "scalers": {
                "targets": {f'param_{i}': MinMaxScaler() for i in range(num_params)},  
                "loss": MinMaxScaler() 
            },
            "bounds": bounds
        }

        os.makedirs(self.config["save_directory_base"], exist_ok=True)

        self.state = {
            "monitor_df": pd.DataFrame(columns=["Total Loss", "Targets", "Psi", "Time","Convergence"]),
        }

        self.gp_model = None 

    def create_run_directory(self):
        """
        Create a new directory for the current run, including a subdirectory for screenshots.
        """
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        run_directory = os.path.join(self.config["save_directory_base"], f'Run_{timestamp}')
        screenshot_directory = os.path.join(run_directory, 'screenshots')

        os.makedirs(run_directory, exist_ok=True)
        os.makedirs(screenshot_directory, exist_ok=True)

        self.config["save_directory"] = run_directory
        self.config["screenshot_directory"] = screenshot_directory
        
    def update_and_save(self, x_list, mean_loss, psi_values, elapsed_time):
        """
        Update the monitor DataFrame and save the latest state of the wall instance.
        """
        # Format time
        time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        
        # Create a single row entry with aggregated data
        new_data = pd.DataFrame({
            "Param": [x_list],
            "Total Loss": [mean_loss],
            "Targets": [self.config["targets"]],
            "Psi": [psi_values],
            "Time": [time_formatted]
        })
        
        # Append the row to the monitor DataFrame
        self.state["monitor_df"] = pd.concat([self.state["monitor_df"], new_data], ignore_index=True)

        # Save the monitor DataFrame
        save_path = os.path.join(self.config["save_directory"], 'dfmonitor.csv')
        self.state["monitor_df"].to_csv(save_path, index=False)
        print(self.state["monitor_df"].tail())

        # Save the latest state of the wall instance
        latest_pkl_path = os.path.join(self.config["save_directory"], 'wall_instance_latest.pkl')
        self.save(latest_pkl_path)
        print(f"Saved latest .pkl file to {latest_pkl_path}")

    def save(self, filename):
        """
        Save the GP model, monitor_df, loss history, and scalers in an organized dictionary.
        """
        data_to_save = {
            'gp_model': self.gp_model.state_dict() if self.gp_model else None,  # Save the state_dict of the GP model
            'monitor_df': self.state["monitor_df"],
            'scalers': {
                'targets': {key: scaler for key, scaler in self.config["scalers"]["targets"].items()},
                'loss': self.config["scalers"]["loss"]
            }
        }

        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)

    @staticmethod
    def load(filename):
        """
        Load the saved dictionary and restore the state.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        wall_instance = WALL([], [], [], [])
        wall_instance.gp_model = SingleTaskGP()
        wall_instance.gp_model.load_state_dict(data['gp_model'])
        wall_instance.state["monitor_df"] = data['monitor_df']

        return wall_instance

    def fit_targets(self):
        scalers = self.config["scalers"]["targets"]
        for i, (scaler_key, scaler) in enumerate(scalers.items()):
            low, high = self.config["bounds"][i]
            bounds_array = np.array([low, high]).reshape(-1, 1)
            scaler.fit(bounds_array)

    def fit_losses(self, Y_init_single):
        scaler = self.config["scalers"]["loss"]
        scaler.fit(Y_init_single.reshape(-1, 1))

    def perform_scaling(self, family, mode, x_values):
        x_scaled = []
        if family == 'loss':
            scaler = self.config["scalers"]["loss"]
            scaled_values = self.scaler(mode, x_values, scaler)
            x_scaled.append(scaled_values)
        elif family == 'targets':
            for i, scaler_type in enumerate(self.config["scalers"]["targets"].keys()):
                variables = x_values[:, i].reshape(-1, 1)
                scaler = self.config["scalers"]["targets"][scaler_type]
                scaled_values = self.scaler(mode, variables, scaler)
                x_scaled.append(scaled_values)
        return torch.tensor(np.array(x_scaled).T)

    def generate_initial_samples(self, n_samples):
        bounds = np.array(self.config["bounds"])
        num_params = bounds.shape[0]
        
        # Initialize LHS Sampler
        sampler = qmc.LatinHypercube(d=num_params)
        sample = sampler.random(n=n_samples)
        
        l_bounds = bounds[:, 0]  
        u_bounds = bounds[:, 1]  
        samples = qmc.scale(sample, l_bounds, u_bounds)
        
        x_scaled = self.perform_scaling('targets', 'scale', samples)
        
        return x_scaled

    def scaler(self, mode, x_values, scaler):
        if mode == 'scale':
            return scaler.transform(x_values).flatten()
        elif mode == 'descale':
            return scaler.inverse_transform(x_values).flatten()

    def run_analysis(self, mat_param, model_idx):
        path = self.config['model_directories'][model_idx]
        openProject(path)
        update_model_parameters(mat_param)
        saveProject()
        runSolver([])
        generate_screenshot(self.config["screenshot_directory"], [len(self.state["monitor_df"]), model_idx])

    def examine_convergence(self, dirOUT):
        lines = read_file(dirOUT)
        _, ncsteps = parse_lines(lines)
        
        if not ncsteps:
            return False
        
        if ncsteps[0] == 1:
            return True
        for i in range(len(ncsteps) - 1):
            if ncsteps[i] + 1 == ncsteps[i + 1]:
                return True
            elif ncsteps[i] != ncsteps[i + 1]:  # Checks if there's no convergence
                return True
        return False

    def processPsi(self, dirTS, crackwidth_threshold=1, distance_threshold=145):
        df = process_tb(dirTS)
        step = df['Step nr.'].max()
        df_filtered = df[(df['Step nr.'] == step) & (df['Ecw1'] >= crackwidth_threshold) & (pd.notna(df['Element']))][['Element', 'Integration Point', 'X0', 'Y0', 'Ecw1']]
        cracks = analyze_cracks(df_filtered, distance_threshold)
        psi = compute_damage_parameter(cracks)
        return psi
    
    def loss_function(self, x_list):
        start_time = time.time()        
        psi_values = []
        loss_values = []

        for model_idx, _ in enumerate(self.config["model_directories"]):
            if isinstance(x_list, torch.Tensor):
                x_list = x_list.detach().cpu().numpy().flatten()

            self.run_analysis(x_list, model_idx)
            dir = self.config["directories"][model_idx]
            try:
                nnc = self.examine_convergence(dir + '.out')
                if nnc:
                    raise Exception("Non-convergence detected")
            except Exception as e:
                logging.warning(f"Convergence examination failed or non-convergence detected for model {model_idx}: {e}")
                psi = 7.0
            else:
                psi = self.processPsi(dir + '.tb')
            
            psi_values.append(psi)

            loss = F.huber_loss(torch.tensor(psi), torch.tensor(self.config["targets"][model_idx]), delta=0.25)
            loss_values.append(loss.item())

        mean_loss = np.mean(loss_values)

        # Scale the losses
        losses = np.array(self.state["monitor_df"]["Total Loss"].tolist() + [mean_loss]).reshape(-1, 1)
        self.fit_losses(losses)
        scaled_losses = self.perform_scaling('loss', 'scale', losses)

        elapsed_time = time.time() - start_time
        self.save_state(x_list, mean_loss, psi_values, elapsed_time, nnc)

        return scaled_losses[-1]

    def save_state(self, x_list, mean_loss, psi_values, elapsed_time, nnc):
        """
        Save the current state of the optimization process.
        """
        time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        new_data = pd.DataFrame({
            "Total Loss": [mean_loss],
            "Targets": [x_list],
            "Psi": [psi_values],
            "Time": [time_formatted],
            "Convergence": [nnc],
        })
        
        # Drop any columns in new_data that are completely empty or NA
        new_data.dropna(how='all', axis=1, inplace=True)

        # Only concatenate if new_data is not empty
        if not new_data.empty:
            self.state["monitor_df"] = pd.concat([self.state["monitor_df"], new_data], ignore_index=True)

        save_path = os.path.join(self.config["save_directory"], 'dfmonitor.csv')
        self.state["monitor_df"].to_csv(save_path, index=False)
        print(self.state["monitor_df"].tail())

        latest_pkl_path = os.path.join(self.config["save_directory"], 'wall_instance_latest.pkl')
        self.save(latest_pkl_path)
        print(f"Saved latest .pkl file to {latest_pkl_path}")

    def objective_function(self, x):
        return self.loss_function(x)
    
    def warm_start(self, n_iter_new, batch_size, bounds, Nrestarts):
        X_reco = np.array(list(self.state["monitor_df"]["Targets"]))
        Y_reco = np.array(list(self.state["monitor_df"]["Total Loss"])).reshape(-1, 1)

        # Scale the inputs and outputs
        X_init_single = self.perform_scaling('targets', 'scale', X_reco)
        Y_init_single = self.perform_scaling('loss', 'scale', Y_reco)

        X_init_single = torch.tensor(X_init_single)
        Y_init_single = torch.tensor(Y_init_single).reshape(-1, 1)

        scaled_bounds_low = self.perform_scaling('targets', 'scale', np.array(bounds)[:, 0].reshape(1, -1))
        scaled_bounds_high = self.perform_scaling('targets', 'scale', np.array(bounds)[:, 1].reshape(1, -1))
        scaled_bounds = torch.tensor(np.vstack((scaled_bounds_low, scaled_bounds_high)).T)

        # Continue Bayesian Optimization with the existing GP model
        X_init_multi, Y_init_multi = SingleBOPT(
            Y_init_single, X_init_single, n_iter_new, batch_size, 
            scaled_bounds, Nrestarts, self.loss_function, self, gp_model=self.gp_model
        )

        return X_init_multi, Y_init_multi

# ---------------------------------------------------------------------------- #
#                                   Optimiser                                  #
# ---------------------------------------------------------------------------- #

def SingleBOPT(Y_init_single, X_init_single, n_iter, r_samples, bounds, Nrestarts, objective_function, wall_instance, gp_model=None):
    
    for _ in range(n_iter):
        
        gp_model = SingleTaskGP(X_init_single, Y_init_single)
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        fit_gpytorch_mll(mll)
        gp_model.set_train_data(inputs=X_init_single, targets=Y_init_single.flatten(), strict=False)
        
        LogEI = LogExpectedImprovement(gp_model,
                                        best_f= Y_init_single.min(),
                                        maximize=False)
        
        new_x, acq_value = optimize_acqf(
            acq_function=LogEI,
            bounds=torch.tensor(bounds, dtype=torch.float32).T,
            q=1,
            num_restarts=Nrestarts,
            raw_samples=raw_samples,
        )
                
        # Scale and evaluate new point
        new_x_unnorm = WALL2.perform_scaling('targets', 'descale', new_x)
        Y_new = objective_function(new_x_unnorm)
                        
        # Update training data
        X_init_single = torch.cat((X_init_single, new_x), dim=0)
        Y_init_single = torch.cat((Y_init_single.flatten(), Y_new.clone().detach()), dim=0).reshape(-1,1)

        WALL2.gp_model = gp_model

    return X_init_single, Y_init_single

# ---------------------------------------------------------------------------- #
#                                   MAIN LOGIC                                 #
# ---------------------------------------------------------------------------- #
model_path_outer = r'C:\Users\javie\OneDrive - Delft University of Technology\Year 2\Q3 & Q4\CIEM0500 - MS Thesis Project\!content\Experimentation\Calibration\TSCM Single\2DW2O - TS.dpf'
an_path_outer = r'C:\Users\javie\OneDrive - Delft University of Technology\Year 2\Q3 & Q4\CIEM0500 - MS Thesis Project\!content\Experimentation\Calibration\TSCM Single\2DW2O_-_TS_NLA'

material_params = MaterialParameters()
confidence_level = 95

Ey_bounds = material_params.get_bounds('Emy', confidence_level)
fw_bounds = material_params.get_bounds('fw', confidence_level)
tensile_strength_bounds = tuple([material_params.tensile_strength(value) for value in fw_bounds])
tensile_fracture_energy_bounds = tuple([material_params.tensile_fracture_energy(value, 'mortar') for value in fw_bounds])

bounds = [Ey_bounds, tensile_strength_bounds, tensile_fracture_energy_bounds]

outer_target = 3.5
targets = [outer_target]

model_paths = [model_path_outer]
an_paths = [an_path_outer]

n_samples = 25
n_iter = 200
raw_samples = 25  # Batch size cannot be smaller than n_samples
Nrestarts = 15  # Nrestarts cannot be smaller than n_samples

WALL2 = WALL(model_paths, an_paths, targets, bounds)
WALL2.create_run_directory()

pickle_file_path = r'C:\Users\javie\OneDrive - Delft University of Technology\Year 2\Q3 & Q4\CIEM0500 - MS Thesis Project\!content\Experimentation\Calibration\TSCM Single\!Results\Run_2024-08-27_17-32-04\wall_instance_latest.pkl'
with open(pickle_file_path, 'rb') as f:
    saved_data = pickle.load(f)

WALL2.config["scalers"] = saved_data["scalers"]
WALL2.state["monitor_df"] = saved_data["monitor_df"]

# Remove rows where "Total Loss" is greater than 1
WALL2.state["monitor_df"] = WALL2.state["monitor_df"][WALL2.state["monitor_df"]["Total Loss"] <= 1]

Y_reco = np.array(list(WALL2.state["monitor_df"]["Total Loss"])).reshape(-1, 1)
WALL2.fit_losses(Y_reco)
Y_init_single = WALL2.perform_scaling('loss', 'scale', Y_reco)

X_reco = np.array(list(WALL2.state["monitor_df"]["Targets"]))
WALL2.fit_targets()
X_init_single = WALL2.perform_scaling('targets', 'scale', X_reco)

# Restore the GP model state if it exists
if saved_data['gp_model'] is not None:
    gp_model = SingleTaskGP(X_init_single, Y_init_single)
    gp_model.load_state_dict(saved_data['gp_model'])
    WALL2.gp_model = gp_model

# Generate initial samples
x_values = WALL2.generate_initial_samples(n_samples=n_samples)
x_unscale = WALL2.perform_scaling('targets', 'descale', x_values)

losses = []
for x in x_unscale: 
    loss = WALL2.loss_function(x)
    losses.append(loss)
    
X_init_single = torch.tensor(x_values)
Y_init_single = torch.tensor(np.array(losses)).reshape(-1, 1)

scaled_bounds_low = WALL2.perform_scaling('targets', 'scale', np.array(bounds)[:, 0].reshape(1, -1))
scaled_bounds_high = WALL2.perform_scaling('targets', 'scale', np.array(bounds)[:, 1].reshape(1, -1))
scaled_bounds = torch.tensor(np.vstack((scaled_bounds_low, scaled_bounds_high)).T)





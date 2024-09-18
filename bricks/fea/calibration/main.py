## Notes

# Because the software interpreter used to run this script does not support dynamic imports, all necessary functions and classes
# from external modules have been defined within this script. This approach ensures that the script can be executed independently,
# without relying on external files or packages that may not be accessible during runtime.

# This script also requires to modify the output type to read the process_tb function. Please see FEA readme documentation to set this properly
# As well model parameters need to be set manually i.e name of load case and load steps as well as material names

## Imports 
import io
import os
import shutil
import traceback
import numpy as np
import pandas as pd
import torch
import time
import pickle
import itertools

from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse import csr_matrix
from scipy.stats import norm

from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
from botorch.models.transforms import Standardize
from botorch.models import SingleTaskGP, MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

from tabulate import tabulate
from contextlib import redirect_stdout
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def combine_scripts(output_directory, output_filename="combined_script.py", **user_params):
    """
    Combines multiple script files in a specified order into a single script file.
    It also allows user-defined parameters to be inserted at specific locations in the script.

    Args:
        output_directory (str): The directory where the combined script will be saved.
        output_filename (str): The name of the output file (default is 'combined_script.py').
        user_params (dict): Dictionary of user-defined parameters to insert into the initiate script.
    """
    script_order = ["imports.py", "utils.py", "material.py", "wall.py", "optimize.py", "initiate.py"]

    combined_content = ""

    for script in script_order:
        script_path = os.path.join(os.path.dirname(__file__), script)
        
        # Read the content of the script
        with open(script_path, 'r') as file:
            script_content = file.read()

            if script == "initiate.py":
                user_params_content = generate_user_params_section(**user_params)
                script_content = user_params_content + "\n\n" + script_content

            combined_content += f"# ---- {script} ----\n\n"
            combined_content += script_content + "\n\n"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_path = os.path.join(output_directory, output_filename)

    with open(output_path, 'w') as output_file:
        output_file.write(combined_content)

    print(f"Combined script created at: {output_path}")

def generate_user_params_section(**user_params):
    """
    Generates a string with user-defined parameters to insert into the initiate.py script.

    Args:
        user_params (dict): Dictionary of user-defined parameters.

    Returns:
        str: A formatted string of user-defined parameters.
    """
    params_section = "# ---- User-defined parameters ----\n\n"
    
    for param, value in user_params.items():
        if isinstance(value, str):
            params_section += f"{param} = r'{value}'\n"
        else:
            params_section += f"{param} = {value}\n"
    
    return params_section

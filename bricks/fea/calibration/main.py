## Notes

# Because the software interpreter used to run this script does not support dynamic imports, all necessary functions and classes
# from external modules have been defined within this script. This approach ensures that the script can be executed independently,
# without relying on external files or packages that may not be accessible during runtime.

# This script also requires to modify the output type to read the process_tb function. Please see FEA readme documentation to set this properly
# As well model parameters need to be set manually i.e name of load case and load steps as well as material names

import os

def combine_scripts(output_directory, output_filename="combined_script.py", **user_params):
    """
    Combines multiple script files in a specified order into a single script file.
    It also allows user-defined parameters to be inserted at specific locations in the script.

    Args:
        output_directory (str): The directory where the combined script will be saved.
        output_filename (str): The name of the output file (default is 'combined_script.py').
        user_params (dict): Dictionary of user-defined parameters to insert into the initiate script.
    """
    # List of script filenames in the order they should be combined
    script_order = ["imports.py", "utils.py", "material.py", "wall.py", "optimize.py", "initiate.py"]

    # Initialize an empty string to hold the combined script content
    combined_content = ""

    # Loop through each script in the specified order
    for script in script_order:
        # Define the full path to the script file
        script_path = os.path.join(os.path.dirname(__file__), script)
        
        # Read the content of the script
        with open(script_path, 'r') as file:
            script_content = file.read()

            # If we're processing the 'initiate.py' script, insert user_params at the beginning
            if script == "initiate.py":
                user_params_content = generate_user_params_section(**user_params)
                script_content = user_params_content + "\n\n" + script_content

            # Append the script content to the combined content
            combined_content += f"# ---- {script} ----\n\n"
            combined_content += script_content + "\n\n"

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define the full path for the output file
    output_path = os.path.join(output_directory, output_filename)

    # Write the combined content to the output file
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


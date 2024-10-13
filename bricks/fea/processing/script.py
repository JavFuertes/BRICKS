#%% Import Modules
import os
import shutil
from string import Template

current_dir = os.path.dirname(__file__)
script_path = os.path.join(current_dir, 'script_template.py')

# Read the template content from the script_template.py file
with open(script_path, 'r') as file:
    script_content = file.read()

#%% Functions
def generate_scripts(file_path, plots_folder, results, script_config):
    """
    Generate Python scripts for analysis based on results.

    Args:
        file_path (str): Path to the .dnb file.
        plots_folder (str): Path to the plots folder.
        results (list): List of result dictionaries containing component and type.
        script_config (dict): Dictionary containing configuration for the script.
    """
    analysis = script_config['analysis']
    load_cases = script_config['load_cases']
    load_steps = script_config['load_steps']
    load_factors_init = script_config['load_factors_init']
    snapshots = script_config['snapshots']
    view_settings = script_config['view_settings']

    steps = []
    load_factors = []

    for i in range(len(load_steps)):
        if i == 0:
            steps.append(load_steps[i])
            load_factor = load_steps[i] * load_factors_init[i]
            load_factors.append(load_factor)
        else:
            val = (snapshots - 1) // (len(load_steps) - 1)
            step_interval = (load_steps[i]) / val
            for j in range(1, val + 1):
                delta = sum(load_steps[:i]) + j * step_interval
                steps.append(int(delta))
                load_factor = (j * step_interval) * load_factors_init[i]
                load_factors.append(round(load_factor, 5))

    # Substitute variables in the template using string.Template
    template = Template(script_content)
    final_script_content = template.safe_substitute(
        file_path=file_path,
        plots_folder=plots_folder,
        results_str=repr(results),
        analysis=analysis,
        load_cases=repr(load_cases),
        load_steps=repr(load_steps),
        load_factors_init=repr(load_factors_init),
        snapshots=snapshots,
        steps=repr(steps),
        load_factors=repr(load_factors),
        view_point=view_settings['view_point'],
        legend_font_size=view_settings['legend_font_size'],
        annotation_font_size=view_settings['annotation_font_size'],
        title_size = view_settings['legend_font_size']
    )

    script_path = os.path.join(plots_folder, "run_analysis.py")
    with open(script_path, 'w') as script_file:
        script_file.write(final_script_content)


# Bricks.FEA

The **Bricks.FEA** module provides a suite of tools designed to support the analysis, processing, and comparison of Finite Element Models (FEMs). It is specifically built to integrate with DIANA 10.8, streamlining complex model evaluations and facilitating detailed comparative analysis.

## Important Considerations for Using Bricks.FEA with DIANA

All scripts within Bricks.FEA have been written to function with DIANA version 10.8. Although compatibility with other versions of DIANA has not been explicitly tested, it is expected that the scripts may still work without modification. Additionally, all data processing is currently optimized for 2D analysis; therefore, usage in 3D scenarios remains untested and may require adjustments.

---

## Components and Key Considerations

### fea.diana

The primary module for analyzing FEMs with DIANA, `fea.diana` includes essential functions to handle model comparisons and data tabulation, enabling insights into structural responses.

### tabulated.py

This script manages tabulated data analysis with the following required DIANA settings:

- **Device:** TABULATED
- **Combine Output Items:** TRUE
- **Tabula Layout Settings:**  
    - Lines per Page = 0  
    - Columns per Page = 100000  
    - Digits for Analysis Results = 4  
    - Digits for Node Coordinates = 4  
    - Digits for Local Axes Direction = 4  
    
- **Output Variables:**  
    All selected variables must be outputted via `RESULT -> User Selection`. Ensure nodes and integration points are configured in separate outputs as results will later be merged.

- **Units:** SI units are used, with lengths output in millimeters (mm). Please note that the script does not verify units, so ensure consistency prior to exporting values.

---

### Model Comparison Functions

For comparing models, Bricks.FEA offers methods to evaluate relative displacements, crack widths, and damage levels based on `tabulated.py` and `out.py`. These analyses provide insights into three primary metrics:

1. **'Mutual':** Compares displacements across nodes within the same model.
2. **'Crack-width':** Tracks the development of various crack widths, displaying the maximum value at the final crack pattern.
3. **'Damage-level':** Assesses the damage parameter, calculating average crack widths using element sizes and predefined load steps.

Example code for defining analysis information:

```python
analysis_info_TSRCM = {
    'Relative displacement': {
        'Node Nr': [22, 23]
    },
    'Mutual': {
        'Node Nr': [[22, 23], [22, 23]],
        'Reference': [['TDtY', 'TDtY'], ['TDtY', 'TDtX']]
    },
    'Crack width': {
        'EOI': [[177,178,179,435],
                [35, 166, 203, 387, 523, 684, 723, 867],
                [9, 206, 263, 612]],
    },
    'Damage level': { 
        'parameters': {
            'cracks': [{
                'EOI': [[177,178,179,435],
                        [35, 166, 203, 387, 523, 684, 723, 867],
                        [9, 206, 263, 612]],
                'element_size': 200,}]         
    }
}}

plot_settings = {
    'Relative displacement': {
        'traces': ['A - Experienced displacement
[Node 22, Bottom-left]', 'B - Applied Displacement
[Node 23, Top-left]'],
        'labels': ['Load factor $\lambda$', 'Displacement $u_y$ [mm]'],
        'titles': 'Displacements at locations of interest',
        'scientific': True
    },
    'Mutual': {
        'traces': ['$u_{y,B}/u_{y,A}$ [TSCM]','$u_{x,B}/u_{y,A}$ [TSCM]'],
        'labels': ['Displacement A [mm]', 'Displacement B [mm]'],
        'titles': 'Relative displacements at locations of interest',
        'scientific': False
    },
    'Crack width': {
        'traces': ['Crack 1','Crack 5','Crack 9'],
        'labels': ['Load factor $\lambda$', 'Crack Width $c_w$ [mm]'],
        'titles': 'Major crack Width development',
        'scientific': True
    },
    'Damage level': {
        'traces': ['$\psi$F'],
        'labels': ['Load factor $\lambda$', 'Damage Parameter $\psi$'],
        'titles': 'Damage level progression',
        'scientific': True
    }
}
```

Use the `analyse_models` function with `merge_info` if you need to combine multiple related plots:

```python
merge_info = {
    'Titles': ['Force norm', 'Displacement norm'],
    'x_label': 'Load factor $\lambda$',
    'y_label': 'Disp & Force Norm $|\Delta_f| |\Delta_u|$',
    'title': 'Combined Force and Displacement Norms'
}

dir = r'\path_to_models'
analyse_models(dir, analysis_info, plot_settings, merge=merge_info)
```

---

### Cross-Model Comparison

For comparing results across multiple models, use the following structure:

```python
data_TSCM = {
    'name': 'TSCM-O',
    'analysis_info': combined_info_TSRCM,
    'plot_settings': combined_settings_TSRCM,
    'dir': r'path\to\TSCM\model_data.tb'
}

data_EMMO = {
    'name': 'EMM - O',
    'analysis_info': combined_info_EMMS,
    'plot_settings': combined_settings_EMMS,
    'dir': r'path\to\EMM\model_data.tb'
}

plot_data_list = [data_TSCM, data_EMMO]
cfigs = compare_models(plot_data_list)
```

---

### report.py

The `report.py` script generates DIANA-compatible scripts based on a unified configuration. This ensures consistency across analyses when they share similar configuration files. To customize, define settings per case as needed.

```python
base_path = r'C:\path\to\models'

config = {
    'results': [
        {
            'component': 'TDtY',
            'result': 'Displacements',
            'type': 'Node',
            'limits': [-35, -28, -24, -20, -16, -12, -8, -4, 0]
        },
        {
            'component': 'E1',
            'result': 'Total Strains',
            'type': 'Element',
            'location': 'mappedintpnt',
            'limits': [-0.004, -0.002, 0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.08]
        },
        {
            'component': 'S1',
            'result': 'Cauchy Total Stresses',
            'type': 'Element',
            'location': 'mappedintpnt',
            'limits': [-3.5, -2, -1, -0.05, -0.01, 0, 0.01, 0.05, 1, 3, 61]
        },
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
        'load_steps': [30, 720],
        'load_factors_init': [0.0330000, 0.00138800],
        'snapshots': 6,
        'view_settings': {
            'view_point': [0, 0, 25.0, 0, 1, 0, 5.2, 3.1, 5.5e-17, 19, 3.25],
            'legend_font_size': 34,
            'annotation_font_size': 28
        }
    }
}

setup_analysis(base_path, config)
```


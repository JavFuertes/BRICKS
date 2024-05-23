import numpy as np
from .utils import hwall_length

def find_inflection_points_and_regions(wallz, coords):
    """
    Finds the inflection points and regions of a wall based on the given wallz and coords.

    Parameters:
    - wallz (array-like):  A list of the subsidence values along the wall length.
    - coords (array-like): The x-coordinates of the wall.

    Returns:
    A dictionary with the following keys:
    - 'inflection_points' (list): The x-coordinates of the inflection points.
    - 'regions' (list): The regions of the wall (-1 for hogging, 1 for sagging).
    - 'region_lengths' (list): The lengths of each region.

    Note:
    - The wallz and coords should have the same length.
    - The function assumes that the wall is represented by a series of points.
    - The function requires at least 3 points to calculate inflection points and regions.
    """
    n = len(coords)
    inflection_points = []
    region_lengths = []
    regions = []

    first_derivatives = np.gradient(wallz, coords)
    second_derivatives = np.gradient(first_derivatives, coords)

    current_region_start_index = 0
    change_count = 0

    for i in range(1, len(second_derivatives)):
        if second_derivatives[i] * second_derivatives[i - 1] < 0:
            change_count += 1
            if change_count == 1:
                inflection_point = coords[i]
                inflection_points.append(inflection_point)

                current_region_type = -1 if second_derivatives[i] > 0 else 1
                regions.append(current_region_type)
                
            
            if change_count == 2:
                
                # Determine the end of the current region
                if i - 1 != current_region_start_index:
                    region_end_index = i - 1
                else:
                    region_end_index = i
                
                inflection_point = coords[i]
                inflection_points.append(inflection_point)

                region_lengths.append(coords[region_end_index] - coords[current_region_start_index])
                current_region_start_index = region_end_index
                
                current_region_type = -1 if second_derivatives[i] > 0 else 1
                regions.append(current_region_type)
                
                change_count = 0  # Reset the change count after finding a region

    # Handle the last region
    if current_region_start_index < n - 1:
        region_lengths.append(coords[-1] - coords[current_region_start_index])
        if len(regions) < 1:
            current_region_type = -1 if second_derivatives[i] > 0 else 1
            regions.append(current_region_type)
        
    return {
        'inflection_points': inflection_points,
        'regions': regions,
        'region_lengths': region_lengths
    }

def evaluate_reldisp_wall_section(coords, wallz, n):
    """
    Evaluate the maximum relative displacement of a wall section.

    This function calculates the maximum relative displacement of a wall section
    based on the given coordinates and wall heights.

    Args:
        coords (list): List of x-coordinates of the wall section.
        wallz (list): List of corresponding wall heights.
        n (int): Number of points in the wall section.

    Returns:
        float: The maximum relative displacement of the wall section.

    """
    max_rel_disp = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = coords[i], wallz[i]
            x2, y2 = coords[j], wallz[j]
            if x2 != x1:  # Avoid division by zero
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                for k in range(min(i, j), max(i, j) + 1):
                    xk, yk = coords[k], wallz[k]
                    line_y = m * xk + b
                    disp = line_y - yk
                    if abs(disp) > abs(max_rel_disp):
                        max_rel_disp = disp
    return max_rel_disp

def max_relative_displacement(wallz, coords, decimal_places=4):
    """
    Calculate the maximum relative displacement of a wall section.

    Parameters:
    wallz (list): List of wall heights.
    coords (list): List of coordinates corresponding to the wall heights.

    Returns:
    tuple: A tuple containing the maximum relative displacement (d_deflection) and a dictionary (infl_dict_) 
           containing the inflection points and the maximum relative displacement for each deflection zone.
    """    
    max_rel_disp = []
    infl_dict_ = find_inflection_points_and_regions(wallz,coords)

    # Avoid carried floating-point errors from region detection
    coords = [round(coord, decimal_places) for coord in coords]
    infl_dict_['inflection_points'] = [round(pt, decimal_places) for pt in infl_dict_['inflection_points']]
    
    if len(infl_dict_['inflection_points']) == 0:
        n = len(coords)
        d_deflection = evaluate_reldisp_wall_section(coords, wallz, n)
        max_rel_disp.append(abs(d_deflection))
    else:
        inflection_points = [coords[0]] + infl_dict_['inflection_points'] + [coords[-1]]
        
        region_start = inflection_points[0]
        start_idx = coords.index(region_start)
        for i in range(1, len(inflection_points)):
            if i == len(inflection_points) - 1:
                region_end = inflection_points[i]
            else:
                next_inflec_idx = coords.index(inflection_points[i + 1])
                region_end = inflection_points[i + 1] if next_inflec_idx - start_idx > 1 else inflection_points[i]

            end_idx = coords.index(region_end)

            segment_coords = coords[start_idx:end_idx + 1]
            segment_wallz = wallz[start_idx:end_idx + 1]

            n = len(segment_coords)
            if n >= 2:
                d_def = evaluate_reldisp_wall_section(segment_coords, segment_wallz, n)
                max_rel_disp.append(abs(d_def))     

            start_idx = end_idx     

    mrd_ = {'d_deflection_zone': max_rel_disp}     
    infl_dict_.update(mrd_)
    d_deflection = max_rel_disp[np.argmax(np.abs(max_rel_disp))]
    return d_deflection, infl_dict_
            
def calculate_phi(wallz, coords):
    ind_min = np.argmin(wallz)
    wmin = wallz[ind_min]
    adjacent_indices = [ind_min - 1, ind_min + 1]
    adjacent_indices = [i for i in adjacent_indices if 0 <= i < len(wallz)]
    adjacent_values = wallz[adjacent_indices]
    x_adjacent = coords[np.isin(wallz, adjacent_values)]
    if len(x_adjacent) == 2:
        x_close = x_adjacent[np.argmax(np.abs(adjacent_values - wmin))]
        phi = np.arctan((wmin - max(adjacent_values)) / (np.abs(x_close - coords[ind_min]) * 1000))
    elif len(x_adjacent) == 1:
        phi = np.arctan((wmin - adjacent_values[0]) / (np.abs(x_adjacent[0] - coords[ind_min]) * 1000))
    else:
        phi = None
    return phi

def calculate_omega(wallz, x_coords):
    inflection_indices = np.where(np.diff(np.sign(np.diff(wallz))) != 0)[0] + 1
    if len(inflection_indices) < 2:
        omega = np.arctan(np.abs(np.diff(wallz)[0]) / ((x_coords[-1] - x_coords[0])*1000))
    else:
        distances = np.diff(inflection_indices)
        longest_period_index = np.argmax(distances)
        x_inflection = x_coords[inflection_indices]
        omega = np.arctan(np.abs(np.diff(wallz)[0]) / ((x_inflection[longest_period_index] - x_inflection[longest_period_index - 1])*1000))
    return omega

def compute_sri(house, wall_num, key):
    """
    Compute the Structural Reliability Index (SRI) for a given wall in a house.

    ## Parameters:
    - house (dict): A dictionary representing the house.
    - wall_num (int): The number of the wall to compute the SRI for.
    - key (str): The key to access the wall in the house dictionary.

    ## Returns:
    A dictionary containing the following SRI parameters:
    - 'Smax': The maximum absolute displacement of the wall.
    - 'dSmax': The difference between the minimum and maximum absolute displacements of the wall.
    - 'D/L': The ratio of dSmax to the length of the wall.
    - 'drat': The maximum relative displacement of the wall.
    - 'omega': The calculated omega parameter.
    - 'phi': The calculated phi parameter.
    - 'beta': The sum of phi and omega.

    """
    wall = house[key]
    wallz = wall['z'] - max(wall['z']) # Normalize displacements against each other  
    length = hwall_length(wall, wall_num+1) * 1000

    if (wall_num + 1) % 2 == 0:
        x = wall['x']
    else:
        x = wall['y']

    s_vmax = min(wallz) - max(wallz)
    d_deflection, infl_dict_ = max_relative_displacement(wallz, x)
    phi = calculate_phi(wallz, x)
    omega = calculate_omega(wallz, x)
    beta = abs(phi) + abs(omega)

    sri =  {'Smax': abs(min(wall['z'])),
            'dSmax': abs(s_vmax),
            'Defrat': abs(s_vmax)/length,
            'drat': abs(d_deflection),
            'omega': abs(omega),
            'phi': abs(phi),
            'beta': abs(beta)}
    
    return sri, infl_dict_


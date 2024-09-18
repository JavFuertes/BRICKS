import numpy as np
from .utils import hwall_length

def find_inflection_points_and_regions(wallz, coords, tolerance):
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
    regions = []
    region_lengths = []
    region_axes = []

    if n < 3:
        # If less than 3 points, assume a single region
        region_type = 0
        regions.append(region_type)
        region_lengths.append(coords[-1] - coords[0])
        region_axes.extend([[coords[0],coords[-1]]])
        
        return {
            'inflection_points': inflection_points,
            'regions': regions,
            'region_lengths': region_lengths,
            'region_axes':region_axes
        }

    first_derivatives = np.gradient(wallz, coords)
    second_derivatives = np.gradient(first_derivatives, coords)

    # Find inflection points using the first derivatives
    for i in range(1, len(first_derivatives) - 1):
        dgrad = (first_derivatives[i] / first_derivatives[i - 1]) - 1
        if abs(dgrad) > tolerance:
            point = coords[i]
            inflection_points.append(point)

    # Calculate regions and region lengths using inflection points
    if inflection_points:
        start_idx = 0
        for i in range(len(inflection_points) - 1):
            ip = inflection_points[i]
            curr_ip_idx = np.where(coords == ip)[0][0]
            if i + 1 < len(inflection_points):
                next_ip = inflection_points[i + 1]
                end_idx = np.where(coords == next_ip)[0][0]
            else:
                end_idx = len(coords) - 1

            n_points = end_idx - curr_ip_idx + 1
            if n_points == 3:
                end_region_idx = curr_ip_idx + 1
            elif n_points < 3:
                end_region_idx = end_idx
            else:
                if n_points % 2 == 0:
                    end_region_idx = curr_ip_idx + n_points // 2
                else:
                    end_region_idx = curr_ip_idx + n_points // 2 + 1

            length = coords[end_region_idx] - coords[start_idx]

            region_type = -1 if second_derivatives[curr_ip_idx] > 0 else 1
            regions.append(region_type)
            region_axes.extend([[coords[start_idx],coords[end_idx]]])
            region_lengths.append(length)

            if n_points < 3:
                start_idx = end_idx - 1
            else:
                start_idx = end_idx

        # Handle the last region
        curr_ip_idx = -1
        end_idx = len(coords) - 1

        region_type = -1 if second_derivatives[start_idx] > 0 else 1
        regions.append(region_type)
        region_axes.extend([[coords[start_idx],coords[end_idx]]])
        region_lengths.append(coords[end_idx] - coords[start_idx])
    else:
        # If no inflection points are found, assume a single region
        region_type = 0
        regions.append(region_type)
        region_lengths.append(coords[-1] - coords[0])
        region_axes.extend([coords[0],coords[-1]])
         
    return {
        'inflection_points': inflection_points,
        'regions': regions,
        'region_lengths': region_lengths,
        'region_axes':region_axes
    }

def calculate_reldisp_section(coords, wallz, n):
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

def max_relative_displacement(wallz, coords, tolerance, decimal_places=4):
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
    infl_dict_ = find_inflection_points_and_regions(wallz,coords, tolerance = tolerance)
    infl_dict_['inflection_points'] = [round(pt, decimal_places) for pt in infl_dict_['inflection_points']]
    coords = [round(coord, decimal_places) for coord in coords]
    
    for start,end in infl_dict_['region_axes']:
        segment_coords = coords[coords.index(start):coords.index(end)+1]
        segment_wallz = wallz[coords.index(start):coords.index(end)+1] 
        n = len(segment_coords)

        d_def = calculate_reldisp_section(segment_coords, segment_wallz, n)
        max_rel_disp.append(abs(d_def))     

    mrd_ = {'d_deflection_zone': max_rel_disp}     
    infl_dict_.update(mrd_)
    d_deflection = max_rel_disp[np.argmax(np.abs(max_rel_disp))]
    return d_deflection, infl_dict_
            
def calculate_phi(wallz, coords):
    """
    Calculate the phi value based on the wallz and coords arrays.

    Parameters:
    - wallz (array-like): An array containing the wallz values.
    - coords (array-like): An array containing the coordinates.

    Returns:
    - phi (float or None): The calculated phi value. Returns None if the calculation is not possible.

    """
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
    """
    Calculate the omega value based on the wallz and x_coords arrays.

    Parameters:
    - wallz (numpy.ndarray): Array containing the wallz values.
    - x_coords (numpy.ndarray): Array containing the x-coordinates.

    Returns:
    - omega (float): The calculated omega value.

    """
    inflection_indices = np.where(np.diff(np.sign(np.diff(wallz))) != 0)[0] + 1
    if len(inflection_indices) < 2:
        omega = np.arctan(np.abs(np.diff(wallz)[0]) / ((x_coords[-1] - x_coords[0])*1000))
    else:
        distances = np.diff(inflection_indices)
        longest_period_index = np.argmax(distances)
        x_inflection = x_coords[inflection_indices]
        omega = np.arctan(np.abs(np.diff(wallz)[0]) / ((x_inflection[longest_period_index] - x_inflection[longest_period_index - 1])*1000))
    return omega

def calculate_beta(wallz, coords, infl_dict_):
    """
    Calculate the maximum beta value for a given wall profile.

    Parameters:
    wallz (numpy.ndarray): Array of wall heights.
    coords (numpy.ndarray): Array of coordinates corresponding to the wall heights.
    infl_dict_ (dict): Dictionary containing information about the regions of interest.

    Returns:
    float: Maximum beta value.

    """
    max_beta = 0
    d1 = np.gradient(wallz, coords)
    for start, end in infl_dict_['region_axes']:
        start_idx = np.where(coords == start)[0][0]
        end_idx = np.where(coords == end)[0][0]

        d_zone = (wallz[end_idx] - wallz[start_idx]) / (coords[end_idx] - coords[start_idx])
        d_left = d1[start_idx]
        d_right = d1[end_idx]
        
        beta_left = abs(np.arctan(d_left) - np.arctan(d_zone))
        beta_right = np.arctan(d_right) - np.arctan(d_zone)
        max_beta_i = (beta_left + beta_right) / 2
        max_beta = max(max_beta, max_beta_i) 
        
    return max_beta

def compute_sri(house, wall_num, key, tolerance = 0.05):
    """
    Compute the Soil related intensity (SRI) factors for a given wall in a house.

    ## Parameters:
    - house (dict): A dictionary representing the house.
    - wall_num (int): The number of the wall to compute the SRI for.
    - key (str): The key to access the wall in the house dictionary.

    ## Returns:
    A dictionary containing the following SRI parameters:
    - 'Smax': The maximum absolute displacement of the wall.
    - 'ΔSmax': The difference between the minimum and maximum absolute displacements of the wall.
    - 'D/L': The ratio of ΔSmax to the length of the wall.
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
    d_deflection, infl_dict_ = max_relative_displacement(wallz, x, tolerance)
    phi = calculate_phi(wallz, x)
    omega = calculate_omega(wallz, x)
    beta = calculate_beta(wallz,x, infl_dict_)

    sri =  {'Smax': abs(min(wall['z'])),
            'ΔSmax': abs(s_vmax),
            'DefRat': abs(s_vmax)/length,
            'dDef': abs(d_deflection),
            'Omega': abs(omega),
            'Phi': abs(phi),
            'Beta': abs(beta)}
    
    return sri, infl_dict_


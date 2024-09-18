from typing import List, Union, Dict

import numpy as np
from scipy.interpolate import interp1d

from .utils import gaussian_shape, hwall_length
from .sri import calculate_reldisp_section
from .emethods import evaluate_wall
from .elimits_db.elimits_epsilon import ParameterLimits, epsilon_empirical_limits

def greenfield_measures(params):
    """
    Calculate various measures related to greenfield analysis.

    Parameters:
    - params (dict): A dictionary containing the following keys:
        - x_gauss (array-like): X-values for the Gaussian shape.
        - s_vmax (float): Standard deviation for the Gaussian shape.
        - x_inflection (float): X-value for the inflection point.

    Returns:
    - gflmeas_ (dict): Dictionary containing hogging and sagging lengths.
    - gfsmeas_ (dict): Dictionary containing normalized measures for hogging and sagging.
    - gfvar_ (dict): Dictionary containing inflection and limit values.
    """
    W = gaussian_shape(params['x_gauss'], params['s_vmax'], params['x_inflection'])
    X = params['x_gauss']
    x_inflection = np.abs(params['x_inflection'])
    w_inflection = interp1d(X, W, kind='nearest')(x_inflection)

    # Assuming length and limit_line are part of params
    length = params['length']
    limit_line = params['limit_line']
    height = params['height']

    x_limit = np.abs(interp1d(W, X, kind='nearest')(limit_line))
    l_hogging = max((length - x_inflection) * 1e3, 0)
    lh_hogging = l_hogging / height
     
    limit_idx = np.argmin(np.abs(W - limit_line))
    inflection_idx = np.argmin(np.abs(W - w_inflection))

    w_h = W[limit_idx: inflection_idx+1]
    x_h = X[limit_idx: inflection_idx+1]
    n = len(w_h)

    dw_hogging = abs(calculate_reldisp_section(w_h, x_h, n)) 
    dl_hogging = 0 if l_hogging == 0 else dw_hogging / l_hogging

    l_sagging = length * 1e3 - l_hogging
    lh_sagging = l_sagging / (height / 2)
    
    limit_idx = np.argmin(np.abs(W - W.min()))
    w_s = W[inflection_idx:limit_idx+1]
    x_s = X[inflection_idx:limit_idx+1]
    n = len(w_s) 
    dw_sagging = abs(calculate_reldisp_section(w_s, x_s, n))     
    dl_sagging = dw_sagging / l_sagging

    gfldisp_ = {'dw_h': dw_hogging, 
                'dw_s': dw_sagging}

    gflmeas_ = {'l_h': l_hogging,
                'l_s': l_sagging}
    
    gfsmeas_ = {'lh_s': lh_sagging,
                'lh_h': lh_hogging,
                'dl_s': dl_sagging,
                'dl_h': dl_hogging,}
    
    gfvar_ = {'xinflection': x_inflection,
               'xlimit': x_limit}

    return gflmeas_, gfsmeas_, gfvar_, gfldisp_

def measurement_measures(params, height):
    """
    Calculate various measures related to greenfield analysis.

    Parameters:
    - params (dict): A dictionary containing the following keys:
        - x_gauss (array-like): X-values for the Gaussian shape.
        - s_vmax (float): Standard deviation for the Gaussian shape.
        - x_inflection (float): X-value for the inflection point.

    Returns:
    - gflmeas_ (dict): Dictionary containing hogging and sagging lengths.
    - gfsmeas_ (dict): Dictionary containing normalized measures for hogging and sagging.
    - gfvar_ (dict): Dictionary containing inflection and limit values.
    """
    gfl_sag_ = []
    gfl_hog_ = []
    
    for i,region in enumerate(params['regions']):
        length = params['region_lengths'][i] * 1e3
        d_defl = params['d_deflection_zone'][i]
        
        if region == -1:
            l_sagging = length
            lh_sagging = l_sagging / (height / 2)
            dl_sagging = d_defl / l_sagging

            gfl_sag_.append({'l_s': l_sagging,
                        'lh_s': lh_sagging,
                        'dl_s': dl_sagging,
                        'dw_s': d_defl,})
        
        if region == 1:
            l_hogging = length
            lh_hogging = l_hogging / height
            dl_hogging = d_defl / l_hogging

            gfl_hog_.append({'l_h': l_hogging,
                        'lh_h': lh_hogging,
                        'dl_h': dl_hogging,
                        'dw_h': d_defl,})        
        
    return gfl_sag_, gfl_hog_

def e_hog(dl_h,lh_h,eg_rat):    
    e_bending_hogg = dl_h * (3 * lh_h / (1 / 4 * lh_h ** 2 + 1.2 * eg_rat))
    e_shear_hogg = dl_h * (3 * eg_rat / ((0.5 * lh_h ** 2) + 2 * 1.2 * eg_rat))
    return e_bending_hogg, e_shear_hogg

def e_sag(dl_s,lh_s,eg_rat):
    e_bending_sagg = dl_s * (6 * lh_s / (lh_s ** 2 + 2 * eg_rat))
    e_shear_sagg = dl_s * (3 * lh_s / (2 * lh_s ** 2 + 2 * 1.2 * eg_rat))

    return e_bending_sagg, e_shear_sagg

def e_total(e_bending_sagg, e_bending_hogg, e_shear_sagg, e_shear_hogg, e_horizontal):
    e_bending = np.max([e_bending_sagg, e_bending_hogg])
    e_shear = np.max([e_shear_sagg, e_shear_hogg])

    e_bt = e_bending + e_horizontal
    e_dt = e_horizontal / (2 + np.sqrt((e_horizontal / 2) ** 2 + e_shear ** 2))
    e_tot = np.max([e_bt, e_dt])
    return e_tot, e_bt, e_dt

def greenfield_strain_measures(lh_s, lh_h, dl_s, dl_h, height, length, wall, eg_rat=11):
    """
    Calculate strain measures for a given set of parameters.

    Parameters:
    - lh_sagging (float): Length of sagging divided by (height / 2).
    - lh_hogging (float): Length of hogging divided by height.
    - dl_sagging (float): Difference in width for sagging.
    - dl_hogging (float): Difference in width for hogging.
    - height (float): The height of the wall.
    - length (float): The length of the wall.
    - wall (dict): The wall dictionary containing the 'phi' values.
    - eg_rat (int, optional): EG ratio value. Defaults to 11.

    Returns:
    - epmeas_ (dict): Dictionary containing various strain measures.
    """
    ratio = height / (2 * 1e3)
    uxy = (wall['phi'][-1] - wall['phi'][0]) * 1000 * ratio
    e_horizontal = abs(uxy / (length * 1e3))

    e_bending_hogg, e_shear_hogg = e_hog(dl_h,lh_h,eg_rat)
    e_bending_sagg, e_shear_sagg = e_sag(dl_s,lh_s,eg_rat)
    
    e_tot, e_bt, e_dt = e_total(e_bending_sagg, e_bending_hogg, 
                                       e_shear_sagg, e_shear_hogg,
                                         e_horizontal)

    epmeas_ = {'e_tot': e_tot, 'e_bt': e_bt, 'e_dt': e_dt, 'e_bh': e_bending_hogg,
               'e_bs': e_bending_sagg, 'e_sh': e_shear_hogg, 'e_ss': e_shear_sagg,
               'e_h': e_horizontal}

    return epmeas_

def measurement_strain_measures(gfl_sag_, gfl_hog_, height, length, wall, eg_rat=11):
    ratio = height / (2 * 1e3)
    uxy = (wall['phi'][-1] - wall['phi'][0]) * 1000 * ratio
    e_horizontal = uxy / (length * 1e3)

    e_bending_sagg = 0 
    e_shear_sagg = 0
    iter_s = None
    if gfl_sag_:
        for i,item in enumerate(gfl_sag_):
            e_bending_s, e_shear_s = e_sag(item['dl_s'], item['lh_s'], eg_rat)
            if e_bending_s >= e_bending_sagg:
                iter_s = i
            e_bending_sagg = np.max([abs(e_bending_sagg), abs(e_bending_s)])
            e_shear_sagg = np.max([abs(e_shear_sagg), abs(e_shear_s)])
    else:
        e_bending_sagg = 0
        e_shear_sagg = 0

    e_bending_hogg = 0
    e_shear_hogg = 0
    iter_h = None
    if gfl_hog_:
        for i,item in enumerate(gfl_hog_):
            e_bending_h, e_shear_h = e_hog(item['dl_h'], item['lh_h'], eg_rat)
            if e_bending_h >= e_bending_hogg:
                iter_h = i
            e_bending_hogg = np.max([abs(e_bending_hogg), abs(e_bending_h)])
            e_shear_hogg = np.max([abs(e_shear_hogg), abs(e_shear_h)])
    else:
        e_bending_hogg = 0
        e_shear_hogg = 0

    e_tot, e_bt, e_dt = e_total(e_bending_sagg, e_bending_hogg, 
                                e_shear_sagg, e_shear_hogg,
                                e_horizontal)

    epmeas_ = {'e_tot': abs(e_tot), 'e_bt': abs(e_bt), 'e_dt': abs(e_dt), 'e_bh': abs(e_bending_hogg),
               'e_bs': abs(e_bending_sagg), 'e_sh': abs(e_shear_hogg), 'e_ss': abs(e_shear_sagg),
               'e_h': abs(e_horizontal)}
    
    best_gfl_sag = gfl_sag_[iter_s] if iter_s is not None else None
    best_gfl_hog = gfl_hog_[iter_h] if iter_h is not None else None
    
    return epmeas_, best_gfl_sag, best_gfl_hog

def LTSM(object, limit_line, methods):
    """
    Perform LTSM (Long-Term Strain Monitoring) assessment on a given object.

    Parameters:
    - object: The object to perform the assessment on.
    - limit_line: The limit line for the assessment.
    - eg_rat: The ratio of elastic modulus to shear modulus. Default is 11.
    - methods: The assessment methods to use. Default is ['greenfield'].

    Returns:
    - None

    The function updates the 'assessment' attribute of the 'object' with the LTSM results.
    """
    gf_ = {'report': {}, 'results': {}, 'values': {}, 'variables': {}}
    ms_ = {'report': {}, 'results': {}, 'values': {}, 'variables': {}} # Leave for unpacking
    eg_l = []
    object.assessment['ltsm'] = {}

    for wall_ in object.house:
        i = list(object.house.keys()).index(wall_) + 1
        wall = object.house[wall_]
        length = hwall_length(wall, i)
        height = wall['height']
        eg_rat = evaluate_eg_rat(wall)

        if 'greenfield' in methods:
            params = object.process['params'][wall_]
            params.update({'length': length, 'height': height, 'limit_line': limit_line})

            gflmeas_, gfsmeas_, gfvar_, gfldisp_ = greenfield_measures(params)
            epmeas_ = greenfield_strain_measures(**gfsmeas_, height=height, length=length, wall=wall, eg_rat=eg_rat)

            strain_val = {'epsilon': epmeas_['e_tot']}
            gf_['report'][wall_] = evaluate_wall(strain_val, empirical_limits=epsilon_empirical_limits())
            gf_['report'][wall_] = update_damage_parameter(gf_['report'][wall_])
            
            gf_['results'][wall_] = {**epmeas_,
                                     **gfldisp_,
                                     **gflmeas_,
                                     **gfsmeas_}

            gf_['variables'][wall_] = {**gfvar_,
                                       'limitline': limit_line,
                                       's_vmax': params['s_vmax']}

            gf_['values'][wall_] = {'x': params['x_gauss'],
                                    'w': gaussian_shape(params['x_gauss'], params['s_vmax'], params['x_inflection'])}

        if 'measurements' in methods:

            params = object.soil['shape'][wall_]
            gfl_sag_, gfl_hog_ = measurement_measures(params, height)
            epmeas_, best_gfl_sag, best_gfl_hog = measurement_strain_measures(gfl_sag_, gfl_hog_, height, length, wall, eg_rat=11)

            strain_val = {'epsilon': epmeas_['e_tot']}
            ms_['report'][wall_] = evaluate_wall(strain_val, empirical_limits=epsilon_empirical_limits())
            ms_['report'][wall_] = update_damage_parameter(ms_['report'][wall_])

            if best_gfl_sag is None:
                best_gfl_sag = {'l_s': 0, 'lh_s': 0, 'dl_s': 0, 'dw_s': 0}
            if best_gfl_hog is None:
                best_gfl_hog = {'l_h': 0, 'lh_h': 0, 'dl_h': 0, 'dw_h': 0}
            
            ms_['results'][wall_] = {**epmeas_,
                                     **best_gfl_sag,
                                     **best_gfl_hog}

        object.house[wall_]['eg_rat'] = eg_rat

    object.assessment['ltsm']['greenfield'] = gf_
    object.assessment['ltsm']['measurements'] = ms_
    
        
         
def update_damage_parameter(report):    
    for assessment in report['epsilon']:
        psi = calculate_damage_ratio(assessment['value'])
        assessment['psi'] = psi
    return report

def calculate_damage_ratio(epsilon_value: float) -> float:

    # Boscardin & Cording (1989) limits
    limits = [0, 0.5e-3, 0.75e-3, 1.5e-3, 3e-3, 1]
    DL = [0, 1, 2, 3, 4, 5]

    # Corresponding damage parameter ranges from the table
    psi_ranges = [
        (0, 1),    # DL0
        (1, 1.5),  # DL1
        (1.5, 2.5),# DL2
        (2.5, 3.5),# DL3
        (3.5, 10)  # DL4
    ]

    for i in range(1, len(limits)):
        lower_limit = limits[i - 1]
        upper_limit = limits[i]

        if lower_limit <= epsilon_value < upper_limit:
            lower_DL = DL[i - 1]
            upper_DL = DL[i]

            # Use DL4's psi range for DL5
            if upper_DL == 5:
                lower_psi = psi_ranges[4][0]
                upper_psi = psi_ranges[4][1]
            else:
                lower_psi = psi_ranges[i - 1][0]
                upper_psi = psi_ranges[i - 1][1]

            if upper_limit == float('inf'):
                damage_parameter = upper_psi
            else:
                # Calculate the ratio within the DL range
                ratio = (epsilon_value - lower_limit) / (upper_limit - lower_limit)
                damage_parameter = lower_psi + ratio * (upper_psi - lower_psi)
            return damage_parameter

    return psi_ranges[-1][1]

def evaluate_eg_rat(wall):
    # According to Son & Cording 2001
    eg_l = [2.6,8,11]
    ocent = [0,10,30]
    
    try:
        area = wall['area']
        opening = wall['opening']

        opercent = (opening / area)* 100
        for i,val in enumerate(ocent):
            if opercent < val:
                if i == 0:
                    eg_lower = eg_l[0]
                    eg_upper = eg_l[1]
                    ocent_lower = ocent[0]
                    ocent_upper = ocent[1]
                else:
                    eg_lower = eg_l[i-1]
                    eg_upper = eg_l[i]
                    ocent_lower = ocent[i-1]
                    ocent_upper = ocent[i]
                eg = eg_lower + (eg_upper - eg_lower) * (opercent - ocent_lower) / (ocent_upper - ocent_lower)
                return eg
    except:
        Exception('Opening area not defined, E/G value = 2.6 for wall with no openings')
        return eg_l[0]

            


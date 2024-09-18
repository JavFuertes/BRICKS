import os
import shutil
from diana import *

# File path and export folder
file_name = r'$file_path'
path_exp = r'$plots_folder'

an = "$analysis"
lc = $load_cases
ls = $load_steps
lfis = $load_factors_init
snapshots = $snapshots
steps = $steps
load_factors = $load_factors

addViewSetting("PY")
setActiveViewSetting("PY")
saveViewPoint("plot_vp", $view_point)
setViewPoint("plot_vp")

for result in $results_str:
    for i, step in enumerate($steps):
        step_name = f"Load-step {step}"
        lf = $load_factors[i]
        lf_name = f"Load-factor {lf:.5f}"
        lc_name = $load_cases[0] if i == 0 else $load_cases[1]

        if result['type'] == 'Node':
            out = 'Output Diana'
        elif result['type'] == 'Element':
            out = 'Monitor Diana'
        
        last = step_name + ', ' + lf_name + ', ' + lc_name
        setResultCase([an, out, last])
        selectResult(result) # Select result has to go after setResultCase
        
        # Legend & annotations 
        setViewSettingValue("PY", "RESULT/TITLE/RANGE", "VISIBLE")
        setViewSettingValue("PY", "RESULT/TITLE/POSIT", "0.0100000 0.990000")
        setViewSettingValue("PY", "RESULT/TITLE/BORDER/BACK", False)
        setViewSettingValue("PY", "RESULT/TITLE/BORDER/FRAME", False)
        setViewSettingValue("PY", "RESULT/TITLE/FONT/SIZE", $title_size)
        setViewSettingValue("PY", "RESULT/LEGEND/LBLFMT", "AUTO")
        setViewSettingValue("PY", "RESULT/LEGEND/LBLPRC", 2)
        setViewSettingValue("PY", "RESULT/LEGEND/FONT/FAMILY", "ARIAL")
        setViewSettingValue("PY", "RESULT/LEGEND/FONT/SIZE", $legend_font_size)
        setViewSettingValue("PY", "RESULT/LEGEND/ANNOTA", "RELFRQ")
        setViewSettingValue("PY", "RESULT/LEGEND/ANNFNT/SIZE", view_settings['legend_font_size'])
        setViewSettingValue("PY", "RESULT/LEGEND/FONT/COLOR", [31, 30, 29, 255])
        setViewSettingValue("PY", "RESULT/LEGEND/ANNFNT/COLOR", [68, 68, 68, 255])
        setViewSettingValue("PY", "RESULT/LEGEND/BORDER/BACK", False)
        setViewSettingValue("PY", "RESULT/LEGEND/BORDER/FRAME", False)

        # Set show max and minimum
        setViewSettingValue("PY", "RESULT/LABEL/EXTREM/LEVEL", "OFF")

        # Deformation settings
        setViewSettingValue("PY", "RESULT/EDGES/RENDEF", "FRE")
        setViewSettingValue("PY", "RESULT/DEFORM/MODE", "ABSOLU")
        setViewSettingValue("PY", "RESULT/DEFORM/ABSOLU/FACTOR", 5)
        setViewSettingValue("PY", "RESULT/DEFORM/DEFX", True)
        setViewSettingValue("PY", "RESULT/DEFORM/DEFY", True)
        setViewSettingValue("PY", "RESULT/DEFORM/DEFZ", True)
        setViewSettingValue("PY", "RESULT/CONTOU/BNDCLR/MAXCLR", [255, 0, 255, 255])
        setViewSettingValue("PY", "RESULT/CONTOU/BNDCLR/MINCLR", [0, 255, 255, 255])

        # Result-specific settings
        values = result['limits']
        setViewSettingValue("PY", "RESULT/CONTOU/LEVELS", "SPECIF")
        setViewSettingValue("PY", "RESULT/CONTOU/LEGEND", "DISCRE")
        setViewSettingValue("PY", "RESULT/CONTOU/SPECIF/VALUES", values)
        setViewSettingValue("PY", "RESULT/CONTOU/AUTRNG", "LIMITS")
        setViewSettingValue("PY", "RESULT/CONTOU/LIMITS/MAXVAL", values[-1])
        setViewSettingValue("PY", "RESULT/CONTOU/LIMITS/MINVAL", values[0])
        setViewSettingValue("PY", "RESULT/CONTOU/LIMITS/BOUNDS", "CLAMP")

        # Save image
        image_path = os.path.join(path_exp, result['component'], f"{step}.png")
        if os.path.exists(image_path):
            os.remove(image_path)
        saveImage(image_path, 1800, 1100, 1)

    # Animated gif -> 30fps 10s
    duration = 10
    fps = 30
    frames = int(fps * duration)

    enableAnimation(True) # Set animation
    setViewSettingValue("PY", "ANIME/MODE", "MULTI")
    setViewSettingValue("PY", "ANIME/FRMCNT", frames)
    setViewSettingValue("PY", "ANIME/LOOP", "TOEND")

    setViewSettingValue("PY", "RESULT/DEFORM/MODE", "NORMAL") # Animation view settings
    setViewSettingValue("PY", "RESULT/DEFORM/NORMAL/FACTOR", 0.02)
    setViewSettingValue("PY", "ANIME/INTER", False)
    
    anim_path = os.path.join(path_exp, result['component'], f"{result['component']}.ogg")
    if os.path.exists(anim_path):
        os.remove(anim_path)
    saveAnimation(anim_path, "ogg", fps, 1800, 1100)
    enableAnimation(False)

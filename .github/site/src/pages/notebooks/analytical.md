---
title: "Analytical"
description: "Analytical assessment of subsidence-induced masonry structures"
pubDate: "2024-01-01"
---

<p style="text-align: justify;">
  This notebook provides an analytical assessment of a detached, two-story unreinforced masonry structure built in the 1960s and later demolished. The assessment is based on a foundation research report conducted prior to the building's demolition, which documented the structure’s measurements, characteristics, damage state, and the effects of subsidence.
  The report found that the foundation's current capacity was insufficient to support the building’s stability, given the experienced deformations and observed damage. Recommendations were made for countermeasures to address the risks to the building’s integrity due to subsidence effects.
</p>

## 0 | Instantiate your `HOUSE`

### 0.1 | Define your house's dimensions and measurements

<CodeCell 
  cellId="setup"
  code={`\`\`\`python
import numpy as np
import analytical as ba

walls = {
    'Wall 1':{"x": np.array([0, 0, 0]), "y": np.array([0, 3.5, 7]), "z": np.array([0, -72, -152]), 'phi': np.array([1/200,1/200]), 'height': 5250, 'thickness': 27,'area': 34.25, 'opening': 4.86},
    'Wall 2':{"x": np.array([0, 4.5, 8.9]), "y": np.array([7, 7, 7]), "z": np.array([-152, -163, -188]),  'phi': np.array([1/33,1/50]), 'height': 5250, 'thickness': 27,'area': 37,'opening': 9.36},
    'Wall 3':{"x": np.array([8.9, 8.9]), "y": np.array([3.6, 7]), "z": np.array([-149, -188]), 'phi': np.array([0,0]), 'height': 5250, 'thickness': 27,'area': 24.35, 'opening': 4.98},
    'Wall 4':{"x": np.array([8.9, 10.8]), "y": np.array([3.6, 3.6]), "z": np.array([-149,-138]), 'phi': np.array([0,0]), 'height': 2850, 'thickness': 27,'area': 8.09, 'opening': 1.68},
    'Wall 5':{"x": np.array([10.8, 10.8]), "y": np.array([0, 3.6]), "z": np.array([-104, -138]), 'phi': np.array([0,0]), 'height': 2850, 'thickness': 27,'area': 9.15, 'opening': 1},
    'Wall 6':{"x": np.array([0, 5.2, 6.4, 8.9, 10.8]), "y": np.array([0, 0, 0, 0, 0]), "z": np.array([0, -42, -55, -75, -104]), 'phi': np.array([1/100,1/100]), 'height': 5000, 'thickness': 27, 'area': 47.58, 'opening': 4.42},
}
\`\`\`} 
/>


### 0.2 | Estimating the displacement surface & greenfield profile

```python
ijsselsteinseweg.interpolate() 
ijsselsteinseweg.fit_function(i_guess = 1, tolerance = 1e-2, step = 1) 

params = ijsselsteinseweg.soil['house'].values()
app = ba.subsurface(ijsselsteinseweg, *params)
```

## 1 | Assessing the damage of the building

### 1.1 | Assessment through Empirical Methods

```python
ijsselsteinseweg.SRI(tolerance= 0.01)
report = ba.EM(ijsselsteinseweg.soil['sri'])
app = ba.EM_plot(report)
```

### 1.2 | Assessment through the LTSM

```python
limit_line = -1
ba.LTSM(ijsselsteinseweg, limit_line, methods = ['greenfield','measurements'])
app = ba.LTSM_plot(ijsselsteinseweg)
```

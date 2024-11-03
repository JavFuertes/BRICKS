---
title: "Welcome"
---

In the Netherlands, engineering consultants often inspect buildings affected by subsidence. These assessments compile key details, including:

- **Building Information**: Construction year, major renovations, and historical details.
- **Damage Documentation**: Photos highlighting damage severity and locations.
- **Measurement Campaign**: Data on structural distortions shown in skew measurement graphs.
- **Soil Analysis**: Soil load-bearing capacity and foundation layout relative to underlying strata.

The goal of **Bricks.Analytical** is to provide tools that leverage the information from these inspection in order for the evaluation of the structural vulnerability of the structures to be made. Thus, helping engineers make reliable assessments efficiently.

| ![Masonry structure accommodating to a subsidence surface](/buildingdamage.svg)|
|---------------------------------------------------------------------------------------------------------|
| **Figure 1:** Masonry structure accommodating to a subsidence surface |


The module is split into two submodules firstly, 

## Overview of the `bricks.analytical` Module

The Bricks module is designed to support engineers and homeowners in the Netherlands with tools to evaluate building vulnerability. The module includes three primary components:

1. **`analytical.house`**: A class to create a building object, enabling the processing of structural and soil characteristics necessary for further analysis.
2. **`analytical.assessment`**: Contains advanced techniques to perform a preliminary vulnerability assessment. Currently, methods include `assess.py` with options like LTSM and empirical threshold-based assessments of structural KPIs. These methods provide quick insights, though some lack the precision of more advanced FEM approaches.
3. **`analytical.assessment.tools`**: A set of visualization tools to help interpret assessment results and visualize the building's current condition.

An example notebook, `bricks_analytical.ipynb`, demonstrates the module's features against the measurements of a case-study, used to assess a structure and utilize the visualization tools in Bricks.

## Steps for Conducting Your Own Assessment

### 1. Set Up Your Building Object and Calculate Key Values

Using data from a building assessment report or self-recorded measurements, define the building's geometry and subsidence measurements.

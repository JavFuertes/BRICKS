from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import plotly.graph_objects as go
import json
import datetime
import bricks.analytical as ba

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeExecution(BaseModel):
    code: str
    cell_id: str

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.post("/api/execute")
async def execute_cell(data: CodeExecution):
    try:
        # Create namespace with imports
        namespace = {
            'np': np,
            'ba': ba,
            'go': go,
            'Figure': go.Figure,
            'json': json
        }
        
        # Execute the walls definition
        exec(data.code, namespace)
        
        if 'walls' not in namespace:
            raise HTTPException(status_code=400, detail="No walls dictionary defined")
            
        # Create house object and perform analysis
        analysis_code = """
# Create house object
ijsselsteinseweg = ba.house(measurements=walls)

# Interpolate and fit
ijsselsteinseweg.interpolate()
ijsselsteinseweg.fit_function(i_guess=1, tolerance=1e-2, step=1)

# Generate subsurface plot
params = ijsselsteinseweg.soil['house'].values()
app1 = ba.subsurface(ijsselsteinseweg, *params)

# Generate EM plot
ijsselsteinseweg.SRI(tolerance=0.01)
report = ba.EM(ijsselsteinseweg.soil['sri'])
app2 = ba.EM_plot(report)

# Generate LTSM plot
limit_line = -1
ba.LTSM(ijsselsteinseweg, limit_line, methods=['greenfield','measurements'])
app3 = ba.LTSM_plot(ijsselsteinseweg)

# Get figures from apps
figures = []
for app in [app1, app2, app3]:
    for child in app.layout.children:
        if hasattr(child, 'figure'):
            figures.append(child.figure)
"""
        exec(analysis_code, namespace)
        
        return {
            "cell_id": data.cell_id,
            "plot_data": json.dumps({
                "subsurface": namespace['figures'][0],
                "em": namespace['figures'][1], 
                "ltsm": namespace['figures'][2]
            })
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


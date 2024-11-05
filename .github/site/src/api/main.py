from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import plotly.graph_objects as go
import json
import datetime
import analytical as ba

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
        # Create namespace with the correct imports
        namespace = {
            'np': np,
            'ba': ba,
            'go': go
        }
        
        # Clean the code input
        code = data.code.strip()
        if code.startswith('```python'):
            code = code[9:]
        if code.endswith('```'):
            code = code[:-3]
        
        # Execute the code
        exec(code, namespace)
        
        # Handle different types of outputs
        if 'app' in namespace and isinstance(namespace['app'], go.Figure):
            return {
                "cell_id": data.cell_id,
                "plot_data": json.loads(namespace['app'].to_json())
            }
        elif 'app' in namespace:
            return {
                "cell_id": data.cell_id,
                "result": str(namespace['app'])
            }
        
        return {
            "cell_id": data.cell_id,
            "result": "Code executed successfully",
            "variables": {
                k: str(v) for k, v in namespace.items() 
                if k not in ['np', 'ba', 'go', '__builtins__']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Check if all components are working"""
    try:
        # Test numpy
        test_array = np.array([1, 2, 3])
        # Test plotly
        test_plot = go.Figure()
        # Test analytical module
        test_analytical = 'ba' in globals()
        
        return {
            "status": "healthy",
            "components": {
                "numpy": "working",
                "plotly": "working",
                "analytical": "working" if test_analytical else "failed"
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
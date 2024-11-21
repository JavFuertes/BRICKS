from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import bricks.analytical as ba
import plotly.graph_objects as go
from multiprocessing import Process
import traceback

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

# Global variables to store our processes
dash_processes = {}

def create_and_run_dash_app(app_type, hinstance, port, params=None):
    """Create and run a Dash app for different plot types."""
    from dash import Dash, html, dcc
    import flask
    
    # Create a Flask server
    server = flask.Flask(__name__)
    
    # Create the Dash app with the necessary configuration
    app = Dash(__name__, server=server, routes_pathname_prefix='/', requests_pathname_prefix='/')
    
    try:
        if app_type == 'subsurface':
            app.layout = ba.subsurface(hinstance, *params).layout
        elif app_type == 'em':
            report = ba.EM(hinstance.soil['sri'])
            app.layout = ba.EM_plot(report).layout
        elif app_type == 'ltsm':
            app.layout = ba.LTSM_plot(hinstance).layout
        
        # Run server on specified port
        server.run(
            host='0.0.0.0',
            port=port,
            debug=False
        )
        
    except Exception as e:
        print(f"Error creating {app_type} plot: {str(e)}")
        raise

@app.post("/api/execute")
async def execute_code(data: CodeExecution):
    try:
        namespace = {
            'np': np,
            'ba': ba,
            'go': go,
        }
        
        exec(data.code, namespace)
        
        if 'walls' not in namespace:
            return {"error": {
                "type": "ValueError",
                "message": "No walls dictionary defined"
            }}
            
        walls = namespace['walls']
            
        try:
            # Stop existing processes
            for process in dash_processes.values():
                if process and process.is_alive():
                    process.terminate()
                    process.join()
            
            hinstance = ba.house(measurements=walls)
            
            # Preprocess data
            hinstance.interpolate()
            hinstance.fit_function(i_guess=1, tolerance=1e-2, step=1)
            hinstance.SRI(tolerance=0.01)
            
            # Run LTSM analysis
            limit_line = -1
            ba.LTSM(hinstance, limit_line, methods=['greenfield','measurements'])
            
            # Generate and serve plots
            plots_config = [
                ('subsurface', 8050, list(hinstance.soil['house'].values())),
                ('em', 8051, None),
                ('ltsm', 8052, None)
            ]

            for plot_type, port, plot_params in plots_config:
                p = Process(target=create_and_run_dash_app, 
                          args=(plot_type, hinstance, port, plot_params))
                p.start()
                dash_processes[plot_type] = p
            
            return {
                "success": True,
                "ports": {
                    "subsurface": 8050,
                    "em": 8051,
                    "ltsm": 8052
                }
            }
            
        except Exception as e:
            return {"error": {
                "type": "ProcessingError",
                "message": str(e),
                "traceback": traceback.format_exc()
            }}
            
    except Exception as e:
        return {"error": {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }}

@app.on_event("shutdown")
async def shutdown_event():
    for process in dash_processes.values():
        if process and process.is_alive():
            process.terminate()

@app.get("/")
async def root():
    return {"message": "API is running"}


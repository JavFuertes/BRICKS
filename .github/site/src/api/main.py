from fastapi import FastAPI, HTTPException
from bricks.analytical import house
from bricks.fea import analysis
import plotly.graph_objects as go
import json

app = FastAPI()

@app.get("/test")
async def test_imports():
    return {
        "message": "Imports working!",
        "modules_available": {
            "house": str(dir(house)),
            "analysis": str(dir(analysis))
        }
    }

@app.post("/api/execute")
async def execute_cell(cell_id: str, code: str):
    try:
        namespace = {
            'house': house,
            'analysis': analysis,
            'go': go,
        }
        
        exec(code, namespace)
        
        if 'app' in namespace:
            return {
                "cell_id": cell_id,
                "plot_data": json.loads(namespace['app'].to_json())
            }
        
        return {"cell_id": cell_id, "result": "Code executed successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
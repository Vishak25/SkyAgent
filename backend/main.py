
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env from backend/.env first, then repo root .env (optional)
BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(BACKEND_DIR.parent / ".env")

from model import STGNN
from data import AviationGraphHandler

app = FastAPI(title="SkyCast ST-GNN Backend")

def _parse_cors_origins(raw: str) -> list:
    raw = (raw or "").strip()
    if not raw:
        return ["http://localhost:3000"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or ["http://localhost:3000"]


# Enable CORS for frontend integration (configurable via env)
cors_origins = _parse_cors_origins(os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:3000"))
allow_credentials = cors_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
graph_handler = AviationGraphHandler()
# Initialize model (Input dim=4 for node features, hidden=16, output=1 scalar delay)
model = STGNN(in_channels=4, hidden_channels=16, out_channels=1)

@app.on_event("startup")
def load_model():
    weights_path = os.environ.get("MODEL_WEIGHTS_PATH")
    if weights_path:
        p = Path(weights_path)
        if p.exists() and p.is_file():
            state = torch.load(str(p), map_location="cpu")
            model.load_state_dict(state)
    model.eval()
    print("ST-GNN Model Ready.")

@app.get("/")
def root():
    return {"message": "SkyCast ST-GNN Backend API is running.", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health_check():
    return {"status": "active", "version": "1.0.0"}

@app.get("/predict/{flight_number}")
def predict_flight_delay(flight_number: str):
    """
    Endpoints that triggers the ST-GNN inference for a specific flight.
    """
    try:
        result = graph_handler.get_prediction_for_flight(flight_number, model)
        if isinstance(result, dict) and result.get("error"):
            status_code = int(result.pop("_status_code", 400) or 400)
            return JSONResponse(status_code=status_code, content=result)
        return result
    except Exception as e:
        print(f"Error predicting for {flight_number}: {e}")
        return JSONResponse(status_code=500, content={"error": "INTERNAL_ERROR", "detail": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

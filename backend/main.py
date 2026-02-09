
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import uvicorn
import os
from pathlib import Path
from typing import Optional
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
# Model: 5 input features (congestion, vis, wind, category, precip_severity)
model = STGNN(in_channels=5, hidden_channels=32, out_channels=1)

@app.on_event("startup")
def load_model():
    weights_path = os.environ.get("MODEL_WEIGHTS_PATH")

    # Auto-discover trained weights if env var not set
    if not weights_path:
        default_path = BACKEND_DIR / "model_weights" / "stgnn_best.pt"
        if default_path.exists():
            weights_path = str(default_path)

    if weights_path:
        p = Path(weights_path)
        if p.exists() and p.is_file():
            state = torch.load(str(p), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            # Mark model as trained in log-space (output = log(1 + delay_minutes))
            model._log_space = True  # type: ignore[attr-defined]
            print(f"Loaded trained weights from {p.name} (log-space output).")
        else:
            print(f"WARNING: weights path '{weights_path}' not found, using random init.")
    else:
        print("No trained weights found. Using random initialisation (heuristic mode).")

    model.eval()
    print("ST-GNN Model Ready (5-dim features, 32 hidden).")

@app.get("/")
def root():
    return {
        "message": "SkyCast ST-GNN Backend API is running.",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "track": "GET /predict/{flight_number}",
            "suggest": "GET /suggest?origin=ORD&destination=LHR&date=2026-02-10",
        },
    }

@app.get("/health")
def health_check():
    return {"status": "active", "version": "2.0.0"}


# ---------------------------------------------------------------------------
# Mode 1: Track a specific flight
# ---------------------------------------------------------------------------

@app.get("/predict/{flight_number}")
def predict_flight_delay(flight_number: str):
    """Track mode: predict delay for a specific booked flight."""
    try:
        result = graph_handler.get_prediction_for_flight(flight_number, model)
        if isinstance(result, dict) and result.get("error"):
            status_code = int(result.pop("_status_code", 400) or 400)
            return JSONResponse(status_code=status_code, content=result)
        return result
    except Exception as e:
        print(f"Error predicting for {flight_number}: {e}")
        return JSONResponse(status_code=500, content={"error": "INTERNAL_ERROR", "detail": str(e)})


# ---------------------------------------------------------------------------
# Mode 2: Suggest alternative routes (pre-departure)
# ---------------------------------------------------------------------------

@app.get("/suggest")
def suggest_routes(
    origin: str = Query(..., description="Origin airport IATA code (e.g. ORD)"),
    destination: str = Query(..., description="Destination airport IATA code (e.g. LHR)"),
    date: Optional[str] = Query(None, description="Travel date YYYY-MM-DD (optional, defaults to today)"),
):
    """
    Suggest mode: given origin + destination, predict delay risk for direct flights
    and 1-stop connections, ranked from lowest to highest risk.
    Uses TAF (weather forecasts) instead of current METAR for forward-looking prediction.
    """
    try:
        result = graph_handler.suggest_routes(origin, destination, model, date_str=date)
        if isinstance(result, dict) and result.get("error"):
            status_code = int(result.pop("_status_code", 400) or 400)
            return JSONResponse(status_code=status_code, content=result)
        return result
    except Exception as e:
        print(f"Error suggesting routes {origin}->{destination}: {e}")
        return JSONResponse(status_code=500, content={"error": "INTERNAL_ERROR", "detail": str(e)})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

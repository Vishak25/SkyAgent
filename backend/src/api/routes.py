"""
FastAPI route definitions for the SkyAgent backend.

Routes:
  GET /                          - Root info
  GET /health                    - Health check
  GET /predict/{flight_number}   - Track mode: delay for a specific flight
  GET /suggest                   - Suggest mode: rank routes by delay risk
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from src.agents.orchestrator import run_pipeline
from src.models.delay_gnn.graph_builder import AviationGraphHandler
from src.models.delay_gnn.model import STGNN

router = APIRouter()

# These are injected by main.py at startup
_graph_handler: Optional[AviationGraphHandler] = None
_model: Optional[STGNN] = None


def init_router(graph_handler: AviationGraphHandler, model: STGNN) -> None:
    """Called from main.py startup to inject shared singletons."""
    global _graph_handler, _model
    _graph_handler = graph_handler
    _model = model


@router.get("/")
def root():
    return {
        "message": "SkyAgent ST-GNN Backend API is running.",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "track": "GET /predict/{flight_number}",
            "analyze": "GET /analyze/{flight_number}",
            "suggest": "GET /suggest?origin=ORD&destination=LHR&date=2026-02-10",
        },
    }


@router.get("/health")
def health_check():
    return {"status": "active", "version": "3.0.0"}


@router.get("/predict/{flight_number}")
def predict_flight_delay(flight_number: str):
    """Track mode: predict delay for a specific booked flight."""
    try:
        result = _graph_handler.get_prediction_for_flight(flight_number, _model)
        if isinstance(result, dict) and result.get("error"):
            status_code = int(result.pop("_status_code", 400) or 400)
            return JSONResponse(status_code=status_code, content=result)
        return result
    except Exception as e:
        print(f"Error predicting for {flight_number}: {e}")
        return JSONResponse(status_code=500, content={"error": "INTERNAL_ERROR", "detail": str(e)})


@router.get("/analyze/{flight_number}")
async def analyze_flight(flight_number: str):
    """Agent mode: full LangGraph pipeline — flight data + ST-GNN + LLM narrative."""
    try:
        result = await run_pipeline(flight_number)
        if isinstance(result, dict) and result.get("error"):
            status_code = int(result.pop("_status_code", 400) or 400)
            return JSONResponse(status_code=status_code, content=result)
        return result
    except Exception as e:
        print(f"Error in agent pipeline for {flight_number}: {e}")
        return JSONResponse(status_code=500, content={"error": "PIPELINE_ERROR", "detail": str(e)})


@router.get("/suggest")
def suggest_routes(
    origin: str = Query(..., description="Origin airport IATA code (e.g. ORD)"),
    destination: str = Query(..., description="Destination airport IATA code (e.g. LHR)"),
    date: Optional[str] = Query(None, description="Travel date YYYY-MM-DD (optional)"),
):
    """Suggest mode: rank direct + 1-stop itineraries by predicted delay risk."""
    try:
        result = _graph_handler.suggest_routes(origin, destination, _model, date_str=date)
        if isinstance(result, dict) and result.get("error"):
            status_code = int(result.pop("_status_code", 400) or 400)
            return JSONResponse(status_code=status_code, content=result)
        return result
    except Exception as e:
        print(f"Error suggesting routes {origin}->{destination}: {e}")
        return JSONResponse(status_code=500, content={"error": "INTERNAL_ERROR", "detail": str(e)})

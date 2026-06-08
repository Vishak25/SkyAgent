"""
SkyAgent FastAPI application entry point.

Run with:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
  (from backend/ directory)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config.settings import parse_cors_origins, BACKEND_DIR, CORS_ALLOW_ORIGINS
from src.models.delay_gnn.model import STGNN
from src.models.delay_gnn.graph_builder import AviationGraphHandler
from src.api.routes import router, init_router

# ---------------------------------------------------------------------------
# Global singletons — loaded during lifespan, shared across requests
# ---------------------------------------------------------------------------

graph_handler = AviationGraphHandler()
model = STGNN(in_channels=5, hidden_channels=32, out_channels=1)


def _load_model_weights() -> None:
    weights_path = os.environ.get("MODEL_WEIGHTS_PATH")

    if not weights_path:
        default_path = BACKEND_DIR / "model_weights" / "stgnn_best.pt"
        if default_path.exists():
            weights_path = str(default_path)

    if weights_path:
        p = Path(weights_path)
        if p.exists() and p.is_file():
            state = torch.load(str(p), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model._log_space = True  # type: ignore[attr-defined]
            print(f"[startup] Loaded trained weights from {p.name} (log-space output).")
        else:
            print(f"[startup] WARNING: weights path '{weights_path}' not found — using random init.")
    else:
        print("[startup] No trained weights found — using random initialisation (heuristic mode).")

    model.eval()
    print("[startup] ST-GNN model ready (5-dim features, 32 hidden).")


# ---------------------------------------------------------------------------
# Lifespan: load model + wire routes + init pipeline before first request
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_weights()
    init_router(graph_handler, model)

    from src.agents.orchestrator import init_pipeline
    init_pipeline(graph_handler, model)
    print("[startup] LangGraph pipeline initialised with trained model.")

    yield
    # nothing to tear down


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SkyAgent ST-GNN Backend",
    version="3.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

cors_origins = parse_cors_origins(CORS_ALLOW_ORIGINS)
allow_credentials = cors_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(router, prefix="/api")

static_dir = BACKEND_DIR / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8080, reload=True)

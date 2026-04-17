"""
SkyAgent FastAPI application entry point.

Run with:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
  (from backend/ directory)

For Hopper deployment, use the SLURM agent_service.slurm script.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import parse_cors_origins, BACKEND_DIR, CORS_ALLOW_ORIGINS
from src.models.delay_gnn.model import STGNN
from src.models.delay_gnn.graph_builder import AviationGraphHandler
from src.api.routes import router, init_router

app = FastAPI(title="SkyAgent ST-GNN Backend", version="3.0.0")

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
# Global singletons (initialised at startup)
# ---------------------------------------------------------------------------

graph_handler = AviationGraphHandler()
model = STGNN(in_channels=5, hidden_channels=32, out_channels=1)


@app.on_event("startup")
def load_model():
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
            print(f"Loaded trained weights from {p.name} (log-space output).")
        else:
            print(f"WARNING: weights path '{weights_path}' not found, using random init.")
    else:
        print("No trained weights found. Using random initialisation (heuristic mode).")

    model.eval()
    print("ST-GNN Model Ready (5-dim features, 32 hidden).")

    # Wire routes
    init_router(graph_handler, model)


# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8080, reload=True)

"""
Rerouting Agent — finds and ranks alternative routes when delay risk is high.

Wraps suggest_routes() for standalone use.  The model must be injected —
it never falls back to a randomly-initialised STGNN.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.models.delay_gnn.graph_builder import AviationGraphHandler
from src.models.delay_gnn.model import STGNN


class ReroutingAgent:
    """Finds and scores alternative routes when delay risk exceeds threshold."""

    DELAY_THRESHOLD_MINUTES = 30

    def __init__(self, model: STGNN, graph_handler: Optional[AviationGraphHandler] = None):
        if model is None:
            raise ValueError(
                "ReroutingAgent requires a trained STGNN instance. "
                "Never pass None — that would silently use random weights."
            )
        self.model = model
        self.graph_handler = graph_handler or AviationGraphHandler()

    async def run(self, origin: str, destination: str, date_str: Optional[str] = None) -> Dict[str, Any]:
        """Return ranked alternative itineraries."""
        result = self.graph_handler.suggest_routes(origin, destination, self.model, date_str=date_str)
        return result

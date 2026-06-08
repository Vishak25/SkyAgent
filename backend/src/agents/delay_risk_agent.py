"""
Delay Risk Agent — runs ST-GNN inference and computes propagation risk score.

In the main LangGraph pipeline the delay calculation happens inside the track node
(via get_prediction_for_flight).  This class is kept for standalone / testing use.
The model must be injected — it never falls back to a randomly-initialised STGNN.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.models.delay_gnn.graph_builder import AviationGraphHandler
from src.models.delay_gnn.model import STGNN


class DelayRiskAgent:
    """Runs GNN-based delay prediction and risk scoring."""

    def __init__(self, model: STGNN, graph_handler: Optional[AviationGraphHandler] = None):
        if model is None:
            raise ValueError(
                "DelayRiskAgent requires a trained STGNN instance. "
                "Never pass None — that would silently use random weights."
            )
        self.model = model
        self.graph_handler = graph_handler or AviationGraphHandler()

    async def run(self, origin: str, destination: str) -> Dict[str, Any]:
        """Build graph, run inference, return delay prediction."""
        from src.config.settings import BASE_AIRPORTS, uniq_preserve
        airports = uniq_preserve(list(BASE_AIRPORTS) + [origin, destination])
        data, ctx = self.graph_handler.build_graph(origin, destination, airports=airports)
        predicted_delay = self.graph_handler._run_model(self.model, data, ctx, destination)
        return {"predictedDelayMinutes": predicted_delay, "origin": origin, "destination": destination}

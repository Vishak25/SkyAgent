"""
Delay Risk Agent — runs ST-GNN inference and computes propagation risk score.

Phase 3 implementation. Uses the trained STGNN model via GNN tools to:
  - Build the airport graph with current weather + congestion features
  - Run ST-GNN inference to predict delay in minutes
  - Compute a composite risk score (weather × congestion × traffic)
  - Return structured risk assessment for the orchestrator

TODO Phase 3: Implement as a LangGraph node with vLLM tool-calling.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.models.delay_gnn.graph_builder import AviationGraphHandler
from src.models.delay_gnn.model import STGNN


class DelayRiskAgent:
    """Runs GNN-based delay prediction and risk scoring."""

    def __init__(self, model: Optional[STGNN] = None):
        self.graph_handler = AviationGraphHandler()
        self.model = model or STGNN(in_channels=5, hidden_channels=32, out_channels=1)

    async def run(self, origin: str, destination: str) -> Dict[str, Any]:
        """
        Build graph, run inference, return delay prediction + risk score.
        TODO Phase 3: Wrap with vLLM tool-calling via LangGraph node.
        """
        from src.config.settings import BASE_AIRPORTS, uniq_preserve
        airports = uniq_preserve(list(BASE_AIRPORTS) + [origin, destination])
        data, ctx = self.graph_handler.build_graph(origin, destination, airports=airports)
        predicted_delay = self.graph_handler._run_model(self.model, data, ctx, destination)
        return {"predictedDelayMinutes": predicted_delay, "origin": origin, "destination": destination}

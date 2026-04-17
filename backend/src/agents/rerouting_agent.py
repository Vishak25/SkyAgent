"""
Rerouting Agent — finds and ranks alternative routes when delay risk is high.

Phase 3 implementation. Builds on the suggest_routes logic to:
  - Enumerate direct + 1-stop alternatives using AeroAPI schedules
  - Score each itinerary with the ST-GNN via TAF forecast features
  - Generate a human-readable recommendation using the vLLM model
  - Return ranked alternatives with natural language justification

TODO Phase 3: Implement as a LangGraph node with vLLM tool-calling.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.models.delay_gnn.graph_builder import AviationGraphHandler
from src.models.delay_gnn.model import STGNN


class ReroutingAgent:
    """Finds and scores alternative routes when delay risk exceeds threshold."""

    DELAY_THRESHOLD_MINUTES = 30

    def __init__(self, model: Optional[STGNN] = None):
        self.graph_handler = AviationGraphHandler()
        self.model = model or STGNN(in_channels=5, hidden_channels=32, out_channels=1)

    async def run(self, origin: str, destination: str, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Return ranked alternative itineraries.
        TODO Phase 3: Enhance with vLLM-generated natural language explanation.
        """
        result = self.graph_handler.suggest_routes(origin, destination, self.model, date_str=date_str)
        return result

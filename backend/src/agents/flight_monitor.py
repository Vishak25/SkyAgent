"""
Flight Monitor Agent — tracks live flight status and position.

Phase 3 implementation. Uses AeroAPI + OpenSky tools to:
  - Fetch real-time flight status (en-route, delayed, cancelled)
  - Retrieve live ADS-B position data
  - Detect delay events and trigger downstream agents

TODO Phase 3: Implement as a LangGraph node with vLLM tool-calling.
"""
from __future__ import annotations

from typing import Any, Dict

from src.tools.flight_tools import AeroAPIClient
from src.tools.opensky_tools import OpenSkyClient


class FlightMonitorAgent:
    """Monitors a specific flight for delay events."""

    def __init__(self):
        self.aeroapi = AeroAPIClient()
        self.opensky = OpenSkyClient()

    async def run(self, flight_number: str) -> Dict[str, Any]:
        """
        Fetch live flight status and position.
        TODO Phase 3: Wrap with vLLM tool-calling via LangGraph node.
        """
        status = self.aeroapi.get_flight_status(flight_number)
        position = self.aeroapi.get_flight_position(flight_number)
        return {"status": status, "position": position}

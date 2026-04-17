"""
Weather Agent — fetches and interprets METAR observations and TAF forecasts.

Phase 3 implementation. Uses CheckWX tools to:
  - Retrieve decoded METAR for origin and destination airports
  - Retrieve TAF (forecast) for pre-departure risk assessment
  - Classify weather severity and flag operational hazards

TODO Phase 3: Implement as a LangGraph node with vLLM tool-calling.
"""
from __future__ import annotations

from typing import Any, Dict

from src.tools.weather_tools import CheckWXClient
from src.config.settings import IATA_TO_ICAO


class WeatherAgent:
    """Fetches and interprets airport weather for delay risk analysis."""

    def __init__(self):
        self.checkwx = CheckWXClient()

    def _to_icao(self, iata: str) -> str:
        return IATA_TO_ICAO.get(iata.upper(), iata)

    async def run(self, origin: str, destination: str, *, use_taf: bool = False) -> Dict[str, Any]:
        """
        Fetch weather for origin and destination.
        TODO Phase 3: Wrap with vLLM tool-calling via LangGraph node.
        """
        o_icao = self._to_icao(origin)
        d_icao = self._to_icao(destination)

        if use_taf:
            weather_origin = self.checkwx.get_taf_data(o_icao)
            weather_dest = self.checkwx.get_taf_data(d_icao)
        else:
            weather_origin = self.checkwx.get_metar_data(o_icao)
            weather_dest = self.checkwx.get_metar_data(d_icao)

        return {"weatherOrigin": weather_origin, "weatherDest": weather_dest}

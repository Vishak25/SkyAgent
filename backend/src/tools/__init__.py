"""
SkyAgent tools — API client wrappers used by agents for data retrieval.

- flight_tools: FlightAware AeroAPI (flight status, airports, scheduling)
- weather_tools: CheckWX (METAR observations, TAF forecasts)
- opensky_tools: OpenSky Network (live ADS-B traffic data)
"""
from src.tools.flight_tools import AeroAPIClient, AirportInfo
from src.tools.weather_tools import CheckWXClient, extract_precip_severity
from src.tools.opensky_tools import OpenSkyClient

__all__ = [
    "AeroAPIClient",
    "AirportInfo",
    "CheckWXClient",
    "OpenSkyClient",
    "extract_precip_severity",
]

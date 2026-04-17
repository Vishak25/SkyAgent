"""
OpenSky Network API tools — provides live air traffic data.

Used by agents to assess airspace congestion and incoming traffic density.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Hashable, List, Optional, Tuple

import requests

from src.config.settings import USE_FIXTURES, clamp, norm_code
from src.tools._cache import TTLCache
from src.tools._fixtures import load_fixture


class OpenSkyClient:
    """OpenSky Network states API (for nearby traffic density)."""

    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        self._fixture_mode = USE_FIXTURES
        self.client_id = client_id or os.environ.get("OPENSKY_USER")
        self.client_secret = client_secret or os.environ.get("OPENSKY_PASSWORD")
        self.base_url = "https://opensky-network.org/api/states/all"
        self.token_url = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
        self.session = requests.Session()
        self.token: Optional[str] = None
        self._states_cache = TTLCache(ttl_seconds=15, maxsize=512)
        if not self._fixture_mode:
            self._authenticate()

    def _authenticate(self) -> None:
        if not self.client_id or not self.client_secret:
            return
        try:
            payload = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }
            resp = self.session.post(self.token_url, data=payload, timeout=10)
            if resp.status_code == 200:
                self.token = resp.json().get("access_token")
            else:
                self.token = None
        except Exception:
            self.token = None

    def _get_states_bbox(self, lamin: float, lomin: float, lamax: float, lomax: float) -> List[list]:
        if self._fixture_mode:
            data = load_fixture("opensky", "states_ORD")
            return (data or {}).get("states", []) or []

        cache_key = (round(lamin, 3), round(lomin, 3), round(lamax, 3), round(lomax, 3), bool(self.token))
        cached = self._states_cache.get(cache_key)
        if cached is not None:
            return cached
        params = {"lamin": lamin, "lomin": lomin, "lamax": lamax, "lomax": lomax}
        headers: Dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            resp = self.session.get(self.base_url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                states = (resp.json() or {}).get("states", []) or []
                self._states_cache.set(cache_key, states)
                return states
        except Exception:
            pass
        self._states_cache.set(cache_key, [])
        return []

    def get_incoming_aircraft_distance(
        self,
        airport_iata: str = "ORD",
        airline_icao: str = "UAL",
        *,
        airport_lat: Optional[float] = None,
        airport_lon: Optional[float] = None,
        radius_km: float = 180.0,
    ) -> List[list]:
        airline_icao = norm_code(airline_icao)
        if airport_lat is None or airport_lon is None:
            lamin, lomin, lamax, lomax = 41.0, -89.0, 43.0, -87.0
        else:
            lat_delta = radius_km / 111.0
            lon_scale = max(0.2, math.cos(math.radians(airport_lat)))
            lon_delta = radius_km / (111.0 * lon_scale)
            lamin, lamax = airport_lat - lat_delta, airport_lat + lat_delta
            lomin, lomax = airport_lon - lon_delta, airport_lon + lon_delta
            lamin = clamp(lamin, -90.0, 90.0)
            lamax = clamp(lamax, -90.0, 90.0)
            lomin = clamp(lomin, -180.0, 180.0)
            lomax = clamp(lomax, -180.0, 180.0)
        states = self._get_states_bbox(lamin, lomin, lamax, lomax)
        if not states or not airline_icao:
            return []
        return [s for s in states if s and len(s) > 1 and s[1] and str(s[1]).strip().upper().startswith(airline_icao)]

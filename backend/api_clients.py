from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, TypeVar

import requests

T = TypeVar("T")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _norm_code(code: str) -> str:
    return (code or "").strip().upper()


class TTLCache:
    def __init__(self, ttl_seconds: float, maxsize: int = 2048):
        self.ttl_seconds = ttl_seconds
        self.maxsize = maxsize
        self._lock = Lock()
        self._data: Dict[Hashable, Tuple[float, Any]] = {}

    def get(self, key: Hashable) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: Hashable, value: Any) -> None:
        now = time.time()
        with self._lock:
            if len(self._data) >= self.maxsize:
                # simple eviction: drop the item expiring soonest
                drop_key = min(self._data.items(), key=lambda kv: kv[1][0])[0]
                self._data.pop(drop_key, None)
            self._data[key] = (now + self.ttl_seconds, value)


@dataclass(frozen=True)
class AirportInfo:
    iata: Optional[str] = None
    icao: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    name: Optional[str] = None


class OpenSkyClient:
    """
    OpenSky Network states API (for nearby traffic density).

    Notes:
    - We use states in a bounding box around an airport as a proxy for traffic pressure.
    - Callsign prefix filtering (e.g. UAL/AAL) is a heuristic.
    """

    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        self.client_id = client_id or os.environ.get("OPENSKY_USER")
        self.client_secret = client_secret or os.environ.get("OPENSKY_PASSWORD")
        self.base_url = "https://opensky-network.org/api/states/all"
        self.token_url = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
        self.session = requests.Session()
        self.token: Optional[str] = None
        self._states_cache = TTLCache(ttl_seconds=15, maxsize=512)

        self._authenticate()

    def _authenticate(self) -> None:
        if not self.client_id or not self.client_secret:
            # Anonymous access still works (with tighter limits).
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
                # Don't crash startup if auth fails; we can proceed anonymously.
                self.token = None
        except Exception:
            self.token = None

    def _get_states_bbox(self, lamin: float, lomin: float, lamax: float, lomax: float) -> List[list]:
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
        """
        Returns OpenSky state vectors in a bbox near the airport, filtered by callsign prefix.

        Backwards compatible with the existing signature. If lat/lon are not provided,
        we fall back to an ORD-area bounding box (legacy behavior).
        """
        airline_icao = _norm_code(airline_icao)

        if airport_lat is None or airport_lon is None:
            # Legacy fallback: ORD region.
            lamin, lomin, lamax, lomax = 41.0, -89.0, 43.0, -87.0
        else:
            # Rough conversion: 1° lat ~ 111km; lon scaled by cos(lat)
            lat_delta = radius_km / 111.0
            lon_scale = max(0.2, math.cos(math.radians(airport_lat)))
            lon_delta = radius_km / (111.0 * lon_scale)
            lamin, lamax = airport_lat - lat_delta, airport_lat + lat_delta
            lomin, lomax = airport_lon - lon_delta, airport_lon + lon_delta

            lamin = _clamp(lamin, -90.0, 90.0)
            lamax = _clamp(lamax, -90.0, 90.0)
            lomin = _clamp(lomin, -180.0, 180.0)
            lomax = _clamp(lomax, -180.0, 180.0)

        states = self._get_states_bbox(lamin, lomin, lamax, lomax)
        if not states or not airline_icao:
            return []

        # states[x][1] is "callsign" in the OpenSky schema
        return [s for s in states if s and len(s) > 1 and s[1] and str(s[1]).strip().upper().startswith(airline_icao)]


class AeroAPIClient:
    """
    FlightAware AeroAPI v4 client.

    Used for:
    - Flight status + position (per flight)
    - Airport delay counts (per airport)
    - Airport info (IATA/ICAO + lat/lon) for normalization
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("FLIGHTAWARE_API_KEY")
        self.enabled = bool(key)
        self.headers = {"x-apikey": key} if key else {}
        self.base_url = "https://aeroapi.flightaware.com/aeroapi/"
        self.session = requests.Session()

        self._airport_cache = TTLCache(ttl_seconds=60 * 60 * 24 * 7, maxsize=4096)  # 7 days
        self._congestion_cache = TTLCache(ttl_seconds=120, maxsize=2048)  # 2 min
        self._flight_status_cache = TTLCache(ttl_seconds=15, maxsize=2048)
        self._flight_position_cache = TTLCache(ttl_seconds=10, maxsize=2048)

    def _get_json(self, url: str, *, params: Optional[Dict[str, Any]] = None, timeout: float = 7) -> Tuple[int, Any]:
        if not self.enabled:
            return 0, None
        try:
            resp = self.session.get(url, headers=self.headers, params=params, timeout=timeout)
            status = resp.status_code
            if status == 204:
                return status, None
            if status >= 200 and status < 300:
                return status, resp.json()
            return status, (resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text)
        except Exception as e:
            return -1, str(e)

    def get_airport_info(self, airport_code: str) -> Optional[AirportInfo]:
        """
        Best-effort lookup. Accepts IATA or ICAO.
        """
        airport_code = _norm_code(airport_code)
        if not airport_code:
            return None

        cached = self._airport_cache.get(airport_code)
        if cached is not None:
            return cached

        url = f"{self.base_url}airports/{airport_code}"
        status, payload = self._get_json(url, timeout=7)
        if status >= 200 and status < 300 and isinstance(payload, dict):
            info = AirportInfo(
                iata=_norm_code(payload.get("iata")) or None,
                icao=_norm_code(payload.get("icao")) or None,
                latitude=payload.get("latitude"),
                longitude=payload.get("longitude"),
                name=payload.get("name"),
            )
            self._airport_cache.set(airport_code, info)
            return info

        self._airport_cache.set(airport_code, None)
        return None

    def get_gate_congestion(self, airport_code: str = "KORD") -> float:
        """
        Uses airport delay counts as a congestion proxy (0..1).
        """
        airport_code = _norm_code(airport_code)
        if not airport_code:
            return 0.0
        if not self.enabled:
            return 0.5

        cached = self._congestion_cache.get(airport_code)
        if cached is not None:
            return float(cached)

        url = f"{self.base_url}airports/{airport_code}/delays"
        status, payload = self._get_json(url, timeout=7)
        if status >= 200 and status < 300 and isinstance(payload, dict):
            dep = float(payload.get("departure_delay_count", 0) or 0)
            arr = float(payload.get("arrival_delay_count", 0) or 0)
            # heuristic normalization: 0..200 delays -> 0..1
            congestion = _clamp((dep + arr) / 200.0, 0.0, 1.0)
            self._congestion_cache.set(airport_code, congestion)
            return congestion

        # On errors, return a small non-zero baseline.
        self._congestion_cache.set(airport_code, 0.1)
        return 0.1

    def get_flight_status(self, flight_ident: str) -> Dict[str, Any]:
        flight_ident = _norm_code(flight_ident)
        if not flight_ident:
            return {"error": "INVALID_FLIGHT", "detail": "Missing flight identifier."}
        if not self.enabled:
            return {"error": "AEROAPI_KEY_MISSING", "detail": "FLIGHTAWARE_API_KEY is not set."}

        cached = self._flight_status_cache.get(flight_ident)
        if cached is not None:
            return cached

        url = f"{self.base_url}flights/{flight_ident}"
        status, payload = self._get_json(url, timeout=7)

        if status == 429:
            result = {"error": "API_QUOTA_EXCEEDED", "detail": "FlightAware API quota limit reached."}
            self._flight_status_cache.set(flight_ident, result)
            return result

        if status >= 200 and status < 300 and isinstance(payload, dict):
            flights = payload.get("flights", [])
            result = flights[0] if isinstance(flights, list) and flights else payload
            self._flight_status_cache.set(flight_ident, result)
            return result

        result = {"error": "FLIGHT_LOOKUP_FAILED", "detail": f"Flight lookup failed (status={status})."}
        self._flight_status_cache.set(flight_ident, result)
        return result

    def get_flight_position(self, flight_ident: str) -> Dict[str, Any]:
        flight_ident = _norm_code(flight_ident)
        if not flight_ident:
            return {}
        if not self.enabled:
            return {}

        cached = self._flight_position_cache.get(flight_ident)
        if cached is not None:
            return cached

        url = f"{self.base_url}flights/{flight_ident}/position"
        status, payload = self._get_json(url, params={"max_pages": 1}, timeout=7)

        if status >= 200 and status < 300 and isinstance(payload, dict):
            if "last_position" in payload and isinstance(payload["last_position"], dict):
                pos = payload["last_position"]
            else:
                positions = payload.get("positions", [])
                pos = positions[0] if isinstance(positions, list) and positions else {}
            self._flight_position_cache.set(flight_ident, pos)
            return pos

        self._flight_position_cache.set(flight_ident, {})
        return {}


class CheckWXClient:
    """
    CheckWX decoded METAR client.
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("CHECKWX_API_KEY")
        self.enabled = bool(key)
        self.headers = {"X-API-Key": key} if key else {}
        self.base_url = "https://api.checkwx.com/metar/"
        self.session = requests.Session()
        self._metar_cache = TTLCache(ttl_seconds=120, maxsize=4096)  # 2 min

    def get_metar_data(self, airport_iata: str = "KORD") -> Dict[str, Any]:
        """
        Retrieves decoded METAR data for an ICAO airport code (e.g. KORD).
        """
        icao = _norm_code(airport_iata)
        if not icao:
            return {}

        cached = self._metar_cache.get(icao)
        if cached is not None:
            return cached

        if not self.enabled:
            result = {
                "visibility_miles": 10.0,
                "wind_speed_kts": 5.0,
                "ceiling_ft": 20000.0,
                "flight_category": "VFR",
                "temp_c": None,
                "temp_f": None,
            }
            self._metar_cache.set(icao, result)
            return result

        url = f"{self.base_url}{icao}/decoded"
        try:
            resp = self.session.get(url, headers=self.headers, timeout=7)
            if resp.status_code == 200:
                json_data = resp.json() or {}
                data_list = json_data.get("data", []) or []
                if not data_list:
                    self._metar_cache.set(icao, {})
                    return {}

                data = data_list[0] or {}
                temp_c = (data.get("temperature") or {}).get("celsius")
                temp_f = None
                try:
                    if temp_c is not None:
                        temp_f = float(temp_c) * 9.0 / 5.0 + 32.0
                except Exception:
                    temp_f = None

                result = {
                    "visibility_miles": (data.get("visibility") or {}).get("miles_float", 10.0),
                    "wind_speed_kts": (data.get("wind") or {}).get("speed_kts", 0.0),
                    "ceiling_ft": (data.get("ceiling") or {}).get("feet", 10000.0),
                    "flight_category": data.get("flight_category", "VFR"),
                    "temp_c": temp_c,
                    "temp_f": temp_f,
                }
                self._metar_cache.set(icao, result)
                return result
        except Exception:
            pass

        self._metar_cache.set(icao, {})
        return {}

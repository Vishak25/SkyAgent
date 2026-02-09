from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, TypeVar

import requests

T = TypeVar("T")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _norm_code(code: str) -> str:
    return (code or "").strip().upper()


# ---------------------------------------------------------------------------
# TTL Cache (thread-safe, in-process)
# ---------------------------------------------------------------------------

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
                drop_key = min(self._data.items(), key=lambda kv: kv[1][0])[0]
                self._data.pop(drop_key, None)
            self._data[key] = (now + self.ttl_seconds, value)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AirportInfo:
    iata: Optional[str] = None
    icao: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    name: Optional[str] = None
    timezone: Optional[str] = None   # IANA tz, e.g. "America/Chicago"


# ---------------------------------------------------------------------------
# Precipitation helpers (winter ops)
# ---------------------------------------------------------------------------

# CheckWX decoded METAR "conditions" array entries contain a "code" field.
# Map codes to a severity float (0 = none, higher = worse for ops).
_PRECIP_SEVERITY: Dict[str, float] = {
    # Rain
    "RA": 0.3, "-RA": 0.2, "+RA": 0.5,
    "DZ": 0.15, "-DZ": 0.1, "+DZ": 0.25,
    "SHRA": 0.35, "-SHRA": 0.2, "+SHRA": 0.55,
    # Snow
    "SN": 0.65, "-SN": 0.45, "+SN": 0.85,
    "SHSN": 0.7, "-SHSN": 0.5, "+SHSN": 0.9,
    "SG": 0.55,
    # Ice / freezing
    "FZRA": 0.95, "-FZRA": 0.8, "+FZRA": 1.0,
    "FZDZ": 0.8, "-FZDZ": 0.6, "+FZDZ": 0.9,
    "PL": 0.75, "IC": 0.7, "GR": 0.85, "GS": 0.6,
    # Thunderstorms
    "TS": 0.8, "+TS": 1.0, "TSRA": 0.9, "+TSRA": 1.0,
    "TSSN": 0.95,
    # Obscuration
    "FG": 0.5, "BR": 0.2, "HZ": 0.1, "FU": 0.15,
    "FZFG": 0.75,
    # Blowing
    "BLSN": 0.7, "BLDU": 0.3, "BLSA": 0.3,
    "SS": 0.5, "+SS": 0.7,
}


def _extract_precip_severity(conditions: Optional[List[Dict[str, Any]]]) -> Tuple[float, str]:
    """
    Returns (severity 0..1, human-readable condition string).
    """
    if not conditions:
        return 0.0, "None"

    max_sev = 0.0
    label_parts: List[str] = []
    for cond in conditions:
        code = str(cond.get("code") or cond.get("text") or "").strip().upper()
        text = str(cond.get("text") or code)
        if not code:
            continue
        sev = _PRECIP_SEVERITY.get(code, 0.0)
        if sev == 0.0:
            # Try without intensity prefix
            for suffix, sv in _PRECIP_SEVERITY.items():
                if code.endswith(suffix) or suffix.endswith(code):
                    sev = max(sev, sv)
        max_sev = max(max_sev, sev)
        if text:
            label_parts.append(text)

    label = ", ".join(label_parts[:3]) if label_parts else "None"
    return _clamp(max_sev, 0.0, 1.0), label


# ---------------------------------------------------------------------------
# OpenSky
# ---------------------------------------------------------------------------

class OpenSkyClient:
    """OpenSky Network states API (for nearby traffic density)."""

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
        airline_icao = _norm_code(airline_icao)
        if airport_lat is None or airport_lon is None:
            lamin, lomin, lamax, lomax = 41.0, -89.0, 43.0, -87.0
        else:
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
        return [s for s in states if s and len(s) > 1 and s[1] and str(s[1]).strip().upper().startswith(airline_icao)]


# ---------------------------------------------------------------------------
# AeroAPI — helpers
# ---------------------------------------------------------------------------


def _pick_best_flight(flights: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the operationally current flight from AeroAPI /flights.

    Why this exists:
    - AeroAPI returns many entries for one ident (past arrivals, current flight,
      and upcoming schedules).
    - We must avoid showing a stale arrived leg when today's delayed/scheduled
      leg is the one users care about.
    """
    if not flights:
        return {}

    def _parse_ts(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None

    now = datetime.now(timezone.utc)
    enriched: List[Dict[str, Any]] = []

    for f in flights:
        status = str(f.get("status") or "").lower()
        progress = f.get("progress_percent")
        cancelled = bool(f.get("cancelled"))

        actual_out = _parse_ts(f.get("actual_out") or f.get("actual_off"))
        actual_in = _parse_ts(f.get("actual_in") or f.get("actual_on"))
        est_out = _parse_ts(f.get("estimated_out") or f.get("scheduled_out") or f.get("scheduled_off"))
        est_in = _parse_ts(f.get("estimated_in") or f.get("scheduled_in") or f.get("scheduled_on"))

        has_arrived = (
            actual_in is not None
            or "arrived" in status
            or "landed" in status
            or "gate arrival" in status
            or (isinstance(progress, (int, float)) and progress >= 100)
        )
        is_en_route = (
            "en route" in status
            or (isinstance(progress, (int, float)) and 0 < progress < 100)
            or (actual_out is not None and actual_in is None and not has_arrived)
        )
        is_pending = not has_arrived and not is_en_route

        if cancelled and not is_en_route:
            continue

        enriched.append(
            {
                "raw": f,
                "status": status,
                "progress": progress if isinstance(progress, (int, float)) else 0.0,
                "actual_out": actual_out,
                "actual_in": actual_in,
                "dep_time": actual_out or est_out,
                "arr_time": actual_in or est_in,
                "is_en_route": is_en_route,
                "is_pending": is_pending,
                "has_arrived": has_arrived,
            }
        )

    if not enriched:
        return flights[0]

    # 1) Prefer live en-route flights.
    en_route = [e for e in enriched if e["is_en_route"]]
    if en_route:
        en_route.sort(
            key=lambda e: (
                float(e["progress"]),
                e["actual_out"] or datetime.min.replace(tzinfo=timezone.utc),
                e["dep_time"] or datetime.min.replace(tzinfo=timezone.utc),
            ),
            reverse=True,
        )
        return en_route[0]["raw"]

    # 2) Prefer current/upcoming operational leg (today-ish) over stale arrivals.
    pending = [e for e in enriched if e["is_pending"]]
    if pending:
        def _pending_key(e: Dict[str, Any]) -> Tuple[int, float, datetime]:
            dep = e["dep_time"]
            if dep is None:
                return (2, float("inf"), datetime.max.replace(tzinfo=timezone.utc))
            delta_min = (dep - now).total_seconds() / 60.0
            # Windowed preference: [-3h, +18h] is likely the operational leg.
            in_window = -180.0 <= delta_min <= 1080.0
            return (
                0 if in_window else 1,
                abs(delta_min),
                dep,
            )

        pending.sort(key=_pending_key)
        return pending[0]["raw"]

    # 3) Otherwise, take the most recently arrived.
    arrived = [e for e in enriched if e["has_arrived"]]
    if arrived:
        arrived.sort(
            key=lambda e: e["actual_in"] or e["arr_time"] or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return arrived[0]["raw"]

    # 4) Fallback.
    return enriched[0]["raw"]


# ---------------------------------------------------------------------------
# AeroAPI
# ---------------------------------------------------------------------------

class AeroAPIClient:
    """FlightAware AeroAPI v4 client."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("FLIGHTAWARE_API_KEY")
        self.enabled = bool(key)
        self.headers = {"x-apikey": key} if key else {}
        self.base_url = "https://aeroapi.flightaware.com/aeroapi/"
        self.session = requests.Session()
        self._airport_cache = TTLCache(ttl_seconds=60 * 60 * 24 * 7, maxsize=4096)
        self._congestion_cache = TTLCache(ttl_seconds=120, maxsize=2048)
        self._flight_status_cache = TTLCache(ttl_seconds=15, maxsize=2048)
        self._flight_position_cache = TTLCache(ttl_seconds=10, maxsize=2048)
        self._scheduled_cache = TTLCache(ttl_seconds=300, maxsize=1024)  # 5 min

    def _get_json(self, url: str, *, params: Optional[Dict[str, Any]] = None, timeout: float = 7) -> Tuple[int, Any]:
        if not self.enabled:
            return 0, None
        try:
            resp = self.session.get(url, headers=self.headers, params=params, timeout=timeout)
            status = resp.status_code
            if status == 204:
                return status, None
            if 200 <= status < 300:
                return status, resp.json()
            return status, (resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text)
        except Exception as e:
            return -1, str(e)

    # -- Airport info --------------------------------------------------------

    def get_airport_info(self, airport_code: str) -> Optional[AirportInfo]:
        airport_code = _norm_code(airport_code)
        if not airport_code:
            return None
        cached = self._airport_cache.get(airport_code)
        if cached is not None:
            return cached
        url = f"{self.base_url}airports/{airport_code}"
        status, payload = self._get_json(url, timeout=7)
        if 200 <= status < 300 and isinstance(payload, dict):
            info = AirportInfo(
                iata=_norm_code(payload.get("iata")) or None,
                icao=_norm_code(payload.get("icao")) or None,
                latitude=payload.get("latitude"),
                longitude=payload.get("longitude"),
                name=payload.get("name"),
                timezone=payload.get("timezone") or None,
            )
            self._airport_cache.set(airport_code, info)
            return info
        self._airport_cache.set(airport_code, None)
        return None

    # -- Congestion ----------------------------------------------------------

    def get_gate_congestion(self, airport_code: str = "KORD") -> float:
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
        if 200 <= status < 300 and isinstance(payload, dict):
            dep = float(payload.get("departure_delay_count", 0) or 0)
            arr = float(payload.get("arrival_delay_count", 0) or 0)
            congestion = _clamp((dep + arr) / 200.0, 0.0, 1.0)
            self._congestion_cache.set(airport_code, congestion)
            return congestion
        self._congestion_cache.set(airport_code, 0.1)
        return 0.1

    # -- Flight status / position -------------------------------------------

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
            result: Dict[str, Any] = {"error": "API_QUOTA_EXCEEDED", "detail": "FlightAware API quota limit reached."}
            self._flight_status_cache.set(flight_ident, result)
            return result
        if 200 <= status < 300 and isinstance(payload, dict):
            flights = payload.get("flights", [])
            result = _pick_best_flight(flights) if isinstance(flights, list) and flights else payload
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
        if 200 <= status < 300 and isinstance(payload, dict):
            if "last_position" in payload and isinstance(payload["last_position"], dict):
                pos = payload["last_position"]
            else:
                positions = payload.get("positions", [])
                pos = positions[0] if isinstance(positions, list) and positions else {}
            self._flight_position_cache.set(flight_ident, pos)
            return pos
        self._flight_position_cache.set(flight_ident, {})
        return {}

    # -- Scheduled flights between airports ----------------------------------

    def get_scheduled_flights(
        self,
        origin: str,
        destination: str,
        *,
        start_iso: Optional[str] = None,
        end_iso: Optional[str] = None,
        max_results: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Returns scheduled/active flights from origin to destination.
        AeroAPI endpoint: GET /airports/{origin}/flights/to/{dest}
        """
        origin = _norm_code(origin)
        destination = _norm_code(destination)
        if not origin or not destination:
            return []
        if not self.enabled:
            return []

        cache_key = (origin, destination, start_iso, end_iso)
        cached = self._scheduled_cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.base_url}airports/{origin}/flights/to/{destination}"
        params: Dict[str, Any] = {"max_pages": 1}
        if start_iso:
            params["start"] = start_iso
        if end_iso:
            params["end"] = end_iso

        status, payload = self._get_json(url, params=params, timeout=10)
        if 200 <= status < 300 and isinstance(payload, dict):
            flights = payload.get("scheduled_arrivals", []) or payload.get("flights", []) or []
            if not isinstance(flights, list):
                flights = []
            flights = flights[:max_results]
            self._scheduled_cache.set(cache_key, flights)
            return flights

        self._scheduled_cache.set(cache_key, [])
        return []


# ---------------------------------------------------------------------------
# CheckWX (METAR + TAF)
# ---------------------------------------------------------------------------

class CheckWXClient:
    """CheckWX decoded METAR + TAF client with precipitation extraction."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("CHECKWX_API_KEY")
        self.enabled = bool(key)
        self.headers = {"X-API-Key": key} if key else {}
        self.base_url = "https://api.checkwx.com/"
        self.session = requests.Session()
        self._metar_cache = TTLCache(ttl_seconds=120, maxsize=4096)
        self._taf_cache = TTLCache(ttl_seconds=600, maxsize=4096)  # 10 min

    # -- METAR ---------------------------------------------------------------

    def get_metar_data(self, icao_code: str = "KORD") -> Dict[str, Any]:
        icao = _norm_code(icao_code)
        if not icao:
            return {}

        cached = self._metar_cache.get(icao)
        if cached is not None:
            return cached

        if not self.enabled:
            result: Dict[str, Any] = {
                "visibility_miles": 10.0,
                "wind_speed_kts": 5.0,
                "ceiling_ft": 20000.0,
                "flight_category": "VFR",
                "temp_c": None,
                "temp_f": None,
                "precip_severity": 0.0,
                "precip_label": "None",
                "conditions_raw": [],
            }
            self._metar_cache.set(icao, result)
            return result

        url = f"{self.base_url}metar/{icao}/decoded"
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

                conditions_raw = data.get("conditions") or []
                precip_sev, precip_label = _extract_precip_severity(conditions_raw)

                result = {
                    "visibility_miles": (data.get("visibility") or {}).get("miles_float", 10.0),
                    "wind_speed_kts": (data.get("wind") or {}).get("speed_kts", 0.0),
                    "ceiling_ft": (data.get("ceiling") or {}).get("feet", 10000.0),
                    "flight_category": data.get("flight_category", "VFR"),
                    "temp_c": temp_c,
                    "temp_f": temp_f,
                    "precip_severity": precip_sev,
                    "precip_label": precip_label,
                    "conditions_raw": conditions_raw,
                }
                self._metar_cache.set(icao, result)
                return result
        except Exception:
            pass

        self._metar_cache.set(icao, {})
        return {}

    # -- TAF (Terminal Aerodrome Forecast) -----------------------------------

    def get_taf_data(self, icao_code: str = "KORD") -> Dict[str, Any]:
        """
        Retrieves decoded TAF (forecast) for an ICAO code.
        Returns summary of worst-case conditions over the forecast period.
        """
        icao = _norm_code(icao_code)
        if not icao:
            return {}

        cached = self._taf_cache.get(icao)
        if cached is not None:
            return cached

        if not self.enabled:
            result: Dict[str, Any] = {
                "forecast_visibility_miles": 10.0,
                "forecast_wind_speed_kts": 5.0,
                "forecast_flight_category": "VFR",
                "forecast_precip_severity": 0.0,
                "forecast_precip_label": "None",
                "forecast_periods": 0,
            }
            self._taf_cache.set(icao, result)
            return result

        url = f"{self.base_url}taf/{icao}/decoded"
        try:
            resp = self.session.get(url, headers=self.headers, timeout=7)
            if resp.status_code == 200:
                json_data = resp.json() or {}
                data_list = json_data.get("data", []) or []
                if not data_list:
                    self._taf_cache.set(icao, {})
                    return {}

                taf = data_list[0] or {}
                forecasts = taf.get("forecast") or []

                # Aggregate worst-case across forecast periods
                worst_vis = 10.0
                worst_wind = 0.0
                worst_precip_sev = 0.0
                worst_precip_label = "None"
                worst_cat = "VFR"
                cat_rank = {"VFR": 0, "MVFR": 1, "IFR": 2, "LIFR": 3}

                for period in forecasts:
                    vis = float((period.get("visibility") or {}).get("miles_float", 10.0) or 10.0)
                    wind = float((period.get("wind") or {}).get("speed_kts", 0.0) or 0.0)
                    cat = str(period.get("flight_category", "VFR") or "VFR").upper()
                    conds = period.get("conditions") or []
                    psev, plabel = _extract_precip_severity(conds)

                    worst_vis = min(worst_vis, vis)
                    worst_wind = max(worst_wind, wind)
                    if psev > worst_precip_sev:
                        worst_precip_sev = psev
                        worst_precip_label = plabel
                    if cat_rank.get(cat, 0) > cat_rank.get(worst_cat, 0):
                        worst_cat = cat

                result = {
                    "forecast_visibility_miles": worst_vis,
                    "forecast_wind_speed_kts": worst_wind,
                    "forecast_flight_category": worst_cat,
                    "forecast_precip_severity": worst_precip_sev,
                    "forecast_precip_label": worst_precip_label,
                    "forecast_periods": len(forecasts),
                }
                self._taf_cache.set(icao, result)
                return result
        except Exception:
            pass

        self._taf_cache.set(icao, {})
        return {}

"""
FlightAware AeroAPI v4 tools — flight status, airport info, congestion, scheduling.

Used by the Flight Monitor Agent and Rerouting Agent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.config.settings import USE_FIXTURES, clamp, norm_code
from src.tools._cache import TTLCache
from src.tools._fixtures import load_fixture


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
    timezone: Optional[str] = None


# ---------------------------------------------------------------------------
# Flight selection helper
# ---------------------------------------------------------------------------

def pick_best_flight(flights: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the operationally current flight from AeroAPI /flights results."""
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

    # 1) Prefer live en-route flights
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

    # 2) Prefer current/upcoming operational leg
    pending = [e for e in enriched if e["is_pending"]]
    if pending:
        def _pending_key(e: Dict[str, Any]) -> Tuple[int, float, datetime]:
            dep = e["dep_time"]
            if dep is None:
                return (2, float("inf"), datetime.max.replace(tzinfo=timezone.utc))
            delta_min = (dep - now).total_seconds() / 60.0
            in_window = -180.0 <= delta_min <= 1080.0
            return (0 if in_window else 1, abs(delta_min), dep)

        pending.sort(key=_pending_key)
        return pending[0]["raw"]

    # 3) Most recently arrived
    arrived = [e for e in enriched if e["has_arrived"]]
    if arrived:
        arrived.sort(
            key=lambda e: e["actual_in"] or e["arr_time"] or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return arrived[0]["raw"]

    return enriched[0]["raw"]


# ---------------------------------------------------------------------------
# AeroAPI Client
# ---------------------------------------------------------------------------

class AeroAPIClient:
    """FlightAware AeroAPI v4 client."""

    def __init__(self, api_key: Optional[str] = None):
        self._fixture_mode = USE_FIXTURES
        key = api_key or os.environ.get("FLIGHTAWARE_API_KEY")
        self.enabled = bool(key) or self._fixture_mode
        self.headers = {"x-apikey": key} if key else {}
        self.base_url = "https://aeroapi.flightaware.com/aeroapi/"
        self.session = requests.Session()
        self._airport_cache = TTLCache(ttl_seconds=60 * 60 * 24 * 7, maxsize=4096)
        self._congestion_cache = TTLCache(ttl_seconds=120, maxsize=2048)
        self._flight_status_cache = TTLCache(ttl_seconds=15, maxsize=2048)
        self._flight_position_cache = TTLCache(ttl_seconds=10, maxsize=2048)
        self._scheduled_cache = TTLCache(ttl_seconds=300, maxsize=1024)

    def _get_json(self, url: str, *, params: Optional[Dict[str, Any]] = None, timeout: float = 7) -> Tuple[int, Any]:
        if not self.enabled:
            return 0, None
        if self._fixture_mode:
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
        airport_code = norm_code(airport_code)
        if not airport_code:
            return None
        cached = self._airport_cache.get(airport_code)
        if cached is not None:
            return cached

        if self._fixture_mode:
            payload = load_fixture("aeroapi", f"airport_{airport_code}")
            if isinstance(payload, dict):
                info = AirportInfo(
                    iata=norm_code(payload.get("code_iata") or payload.get("iata")) or None,
                    icao=norm_code(payload.get("code_icao") or payload.get("icao")) or None,
                    latitude=payload.get("latitude"),
                    longitude=payload.get("longitude"),
                    name=payload.get("name"),
                    timezone=payload.get("timezone") or None,
                )
                self._airport_cache.set(airport_code, info)
                return info
            self._airport_cache.set(airport_code, None)
            return None

        url = f"{self.base_url}airports/{airport_code}"
        status, payload = self._get_json(url, timeout=7)
        if 200 <= status < 300 and isinstance(payload, dict):
            info = AirportInfo(
                iata=norm_code(payload.get("code_iata") or payload.get("iata")) or None,
                icao=norm_code(payload.get("code_icao") or payload.get("icao")) or None,
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
        airport_code = norm_code(airport_code)
        if not airport_code:
            return 0.0
        if not self.enabled:
            return 0.5
        cached = self._congestion_cache.get(airport_code)
        if cached is not None:
            return float(cached)

        if self._fixture_mode:
            payload = load_fixture("aeroapi", f"delays_{airport_code}")
            if isinstance(payload, dict):
                dep = float(payload.get("departure_delay_count", 0) or 0)
                arr = float(payload.get("arrival_delay_count", 0) or 0)
                congestion = clamp((dep + arr) / 200.0, 0.0, 1.0)
                self._congestion_cache.set(airport_code, congestion)
                return congestion
            self._congestion_cache.set(airport_code, 0.1)
            return 0.1

        url = f"{self.base_url}airports/{airport_code}/delays"
        status, payload = self._get_json(url, timeout=7)
        if 200 <= status < 300 and isinstance(payload, dict):
            dep = float(payload.get("departure_delay_count", 0) or 0)
            arr = float(payload.get("arrival_delay_count", 0) or 0)
            congestion = clamp((dep + arr) / 200.0, 0.0, 1.0)
            self._congestion_cache.set(airport_code, congestion)
            return congestion
        self._congestion_cache.set(airport_code, 0.1)
        return 0.1

    # -- Flight status / position -------------------------------------------

    def get_flight_status(self, flight_ident: str) -> Dict[str, Any]:
        flight_ident = norm_code(flight_ident)
        if not flight_ident:
            return {"error": "INVALID_FLIGHT", "detail": "Missing flight identifier."}
        if not self.enabled:
            return {"error": "AEROAPI_KEY_MISSING", "detail": "FLIGHTAWARE_API_KEY is not set."}
        cached = self._flight_status_cache.get(flight_ident)
        if cached is not None:
            return cached

        if self._fixture_mode:
            payload = load_fixture("aeroapi", f"flights_{flight_ident}")
            if isinstance(payload, dict):
                flights = payload.get("flights", [])
                result = pick_best_flight(flights) if isinstance(flights, list) and flights else payload
                self._flight_status_cache.set(flight_ident, result)
                return result
            result: Dict[str, Any] = {"error": "FLIGHT_LOOKUP_FAILED", "detail": f"No fixture for {flight_ident}."}
            self._flight_status_cache.set(flight_ident, result)
            return result

        url = f"{self.base_url}flights/{flight_ident}"
        status, payload = self._get_json(url, timeout=7)
        if status == 429:
            result = {"error": "API_QUOTA_EXCEEDED", "detail": "FlightAware API quota limit reached."}
            self._flight_status_cache.set(flight_ident, result)
            return result
        if 200 <= status < 300 and isinstance(payload, dict):
            flights = payload.get("flights", [])
            result = pick_best_flight(flights) if isinstance(flights, list) and flights else payload
            self._flight_status_cache.set(flight_ident, result)
            return result
        result = {"error": "FLIGHT_LOOKUP_FAILED", "detail": f"Flight lookup failed (status={status})."}
        self._flight_status_cache.set(flight_ident, result)
        return result

    def get_flight_position(self, flight_ident: str) -> Dict[str, Any]:
        flight_ident = norm_code(flight_ident)
        if not flight_ident:
            return {}
        if not self.enabled:
            return {}
        cached = self._flight_position_cache.get(flight_ident)
        if cached is not None:
            return cached

        if self._fixture_mode:
            self._flight_position_cache.set(flight_ident, {})
            return {}

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
        origin = norm_code(origin)
        destination = norm_code(destination)
        if not origin or not destination:
            return []
        if not self.enabled:
            return []

        cache_key = (origin, destination, start_iso, end_iso)
        cached = self._scheduled_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._fixture_mode:
            payload = load_fixture("aeroapi", f"scheduled_{origin}_{destination}")
            if isinstance(payload, dict):
                flights = payload.get("scheduled_arrivals", []) or payload.get("flights", []) or []
                if not isinstance(flights, list):
                    flights = []
                flights = flights[:max_results]
                self._scheduled_cache.set(cache_key, flights)
                return flights
            self._scheduled_cache.set(cache_key, [])
            return []

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

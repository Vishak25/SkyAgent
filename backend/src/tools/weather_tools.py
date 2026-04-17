"""
CheckWX METAR + TAF weather tools — provides airport weather observations and forecasts.

Used by the Weather Agent and Delay Risk Agent to assess weather conditions.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.config.settings import USE_FIXTURES, clamp, norm_code
from src.tools._cache import TTLCache
from src.tools._fixtures import load_fixture


# ---------------------------------------------------------------------------
# Precipitation helpers (winter ops)
# ---------------------------------------------------------------------------

PRECIP_SEVERITY: Dict[str, float] = {
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


def extract_precip_severity(conditions: Optional[List[Dict[str, Any]]]) -> Tuple[float, str]:
    """Returns (severity 0..1, human-readable condition string)."""
    if not conditions:
        return 0.0, "None"

    max_sev = 0.0
    label_parts: List[str] = []
    for cond in conditions:
        code = str(cond.get("code") or cond.get("text") or "").strip().upper()
        text = str(cond.get("text") or code)
        if not code:
            continue
        sev = PRECIP_SEVERITY.get(code, 0.0)
        if sev == 0.0:
            for suffix, sv in PRECIP_SEVERITY.items():
                if code.endswith(suffix) or suffix.endswith(code):
                    sev = max(sev, sv)
        max_sev = max(max_sev, sev)
        if text:
            label_parts.append(text)

    label = ", ".join(label_parts[:3]) if label_parts else "None"
    return clamp(max_sev, 0.0, 1.0), label


# ---------------------------------------------------------------------------
# CheckWX Client
# ---------------------------------------------------------------------------

class CheckWXClient:
    """CheckWX decoded METAR + TAF client with precipitation extraction."""

    def __init__(self, api_key: Optional[str] = None):
        self._fixture_mode = USE_FIXTURES
        key = api_key or os.environ.get("CHECKWX_API_KEY")
        self.enabled = bool(key) or self._fixture_mode
        self.headers = {"X-API-Key": key} if key else {}
        self.base_url = "https://api.checkwx.com/"
        self.session = requests.Session()
        self._metar_cache = TTLCache(ttl_seconds=120, maxsize=4096)
        self._taf_cache = TTLCache(ttl_seconds=600, maxsize=4096)

    # -- METAR ---------------------------------------------------------------

    def get_metar_data(self, icao_code: str = "KORD") -> Dict[str, Any]:
        icao = norm_code(icao_code)
        if not icao:
            return {}

        cached = self._metar_cache.get(icao)
        if cached is not None:
            return cached

        # Fixture mode
        if self._fixture_mode:
            fixture_data = load_fixture("checkwx", f"metar_{icao}")
            if isinstance(fixture_data, dict):
                data_list = fixture_data.get("data", []) or []
                if data_list:
                    data = data_list[0] or {}
                    temp_c = (data.get("temperature") or {}).get("celsius")
                    temp_f = None
                    try:
                        if temp_c is not None:
                            temp_f = float(temp_c) * 9.0 / 5.0 + 32.0
                    except Exception:
                        temp_f = None
                    conditions_raw = data.get("conditions") or []
                    precip_sev, precip_label = extract_precip_severity(conditions_raw)
                    result: Dict[str, Any] = {
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
            result = {
                "visibility_miles": 10.0, "wind_speed_kts": 5.0,
                "ceiling_ft": 20000.0, "flight_category": "VFR",
                "temp_c": None, "temp_f": None,
                "precip_severity": 0.0, "precip_label": "None",
                "conditions_raw": [],
            }
            self._metar_cache.set(icao, result)
            return result

        if not self.enabled:
            result = {
                "visibility_miles": 10.0, "wind_speed_kts": 5.0,
                "ceiling_ft": 20000.0, "flight_category": "VFR",
                "temp_c": None, "temp_f": None,
                "precip_severity": 0.0, "precip_label": "None",
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
                precip_sev, precip_label = extract_precip_severity(conditions_raw)

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

    def _parse_taf_payload(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse decoded TAF JSON into a summary dict."""
        data_list = json_data.get("data", []) or []
        if not data_list:
            return {}

        taf = data_list[0] or {}
        forecasts = taf.get("forecast") or []

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
            psev, plabel = extract_precip_severity(conds)

            worst_vis = min(worst_vis, vis)
            worst_wind = max(worst_wind, wind)
            if psev > worst_precip_sev:
                worst_precip_sev = psev
                worst_precip_label = plabel
            if cat_rank.get(cat, 0) > cat_rank.get(worst_cat, 0):
                worst_cat = cat

        return {
            "forecast_visibility_miles": worst_vis,
            "forecast_wind_speed_kts": worst_wind,
            "forecast_flight_category": worst_cat,
            "forecast_precip_severity": worst_precip_sev,
            "forecast_precip_label": worst_precip_label,
            "forecast_periods": len(forecasts),
        }

    def get_taf_data(self, icao_code: str = "KORD") -> Dict[str, Any]:
        """Retrieves decoded TAF (forecast) for an ICAO code."""
        icao = norm_code(icao_code)
        if not icao:
            return {}

        cached = self._taf_cache.get(icao)
        if cached is not None:
            return cached

        # Fixture mode
        if self._fixture_mode:
            fixture_data = load_fixture("checkwx", f"taf_{icao}")
            if isinstance(fixture_data, dict):
                result = self._parse_taf_payload(fixture_data)
                if result:
                    self._taf_cache.set(icao, result)
                    return result
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

        if not self.enabled:
            result = {
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
                result = self._parse_taf_payload(json_data)
                if result:
                    self._taf_cache.set(icao, result)
                    return result
        except Exception:
            pass

        self._taf_cache.set(icao, {})
        return {}

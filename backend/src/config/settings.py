"""
Centralized configuration, constants, and shared helpers for SkyAgent.

Loads environment from .env files and exposes settings + airport data.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[2]  # backend/
PROJECT_ROOT = BACKEND_DIR.parent                   # FlightPredictorAI/

# Load env from backend/.env first, then repo root .env (optional)
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Environment settings
# ---------------------------------------------------------------------------

FLIGHTAWARE_API_KEY = os.environ.get("FLIGHTAWARE_API_KEY", "")
CHECKWX_API_KEY = os.environ.get("CHECKWX_API_KEY", "")
OPENSKY_USER = os.environ.get("OPENSKY_USER", "")
OPENSKY_PASSWORD = os.environ.get("OPENSKY_PASSWORD", "")
CORS_ALLOW_ORIGINS = os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:3000")
MODEL_WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS_PATH", "")
STGNN_OUTPUT_SCALE = float(os.environ.get("STGNN_OUTPUT_SCALE", "100") or 100.0)

# LLM settings — any OpenAI-compatible endpoint (Gemini, vLLM, OpenAI, etc.)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "skyagent-dev")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

USE_FIXTURES = os.environ.get("USE_FIXTURES", "").strip() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Flight category ordinal mapping
# ---------------------------------------------------------------------------

CAT_MAP: Dict[str, float] = {"VFR": 0.0, "MVFR": 0.33, "IFR": 0.66, "LIFR": 1.0}

# ---------------------------------------------------------------------------
# Major hub airports included in every graph for network context
# ---------------------------------------------------------------------------

BASE_AIRPORTS: List[str] = [
    "ORD", "JFK", "LHR", "LAX", "DXB", "HND", "CDG", "AMS", "FRA", "SIN",
    "ATL", "DFW", "DEN", "SFO", "IAD", "BOS", "MIA", "SEA", "EWR", "PHL",
]

# ---------------------------------------------------------------------------
# Approximate lat/lon for known airports (avoids API calls for graph viz)
# ---------------------------------------------------------------------------

AIRPORT_COORDS: Dict[str, Tuple[float, float]] = {
    "ORD": (41.98, -87.90), "JFK": (40.64, -73.78), "LHR": (51.47, -0.46),
    "LAX": (33.94, -118.41), "DXB": (25.25, 55.36), "HND": (35.55, 139.78),
    "CDG": (49.01, 2.55), "AMS": (52.31, 4.77), "FRA": (50.03, 8.57),
    "SIN": (1.35, 103.99), "ATL": (33.64, -84.43), "DFW": (32.90, -97.04),
    "DEN": (39.86, -104.67), "SFO": (37.62, -122.38), "IAD": (38.95, -77.45),
    "BOS": (42.36, -71.01), "MIA": (25.80, -80.29), "SEA": (47.45, -122.31),
    "EWR": (40.69, -74.17), "PHL": (39.87, -75.24), "PHX": (33.44, -112.01),
    "SBN": (41.71, -86.32), "MCO": (28.43, -81.31), "DTW": (42.21, -83.35),
    "CLT": (35.21, -80.94), "MSP": (44.88, -93.22), "SLC": (40.79, -111.98),
    "IAH": (29.98, -95.34), "BWI": (39.18, -76.67), "SAN": (32.73, -117.19),
    "TPA": (27.98, -82.53), "PDX": (45.59, -122.59), "STL": (38.75, -90.37),
    "HNL": (21.32, -157.92), "ANC": (61.17, -150.00), "NRT": (35.76, 140.39),
    "ICN": (37.46, 126.44), "PEK": (40.08, 116.58), "SYD": (-33.95, 151.18),
    "GRU": (-23.43, -46.47), "MEX": (19.44, -99.07), "YYZ": (43.68, -79.63),
    "LGA": (40.78, -73.87),
    "LNK": (40.85, -96.76), "OMA": (41.30, -95.89), "DSM": (41.53, -93.66),
    "MSN": (43.14, -89.34), "MKE": (42.95, -87.90), "CLE": (41.41, -81.85),
    "CMH": (39.99, -82.89), "IND": (39.72, -86.29), "CVG": (39.05, -84.67),
    "BNA": (36.12, -86.68), "MEM": (35.04, -89.98), "OKC": (35.39, -97.60),
    "TUL": (36.20, -95.89), "ABQ": (35.04, -106.61), "ELP": (31.81, -106.38),
    "SAT": (29.53, -98.47), "AUS": (30.20, -97.67), "HOU": (29.65, -95.28),
    "DAL": (32.85, -96.85),
}

# MVP IATA -> ICAO mapping (used when AeroAPI is unavailable)
IATA_TO_ICAO: Dict[str, str] = {
    "ORD": "KORD", "JFK": "KJFK", "LHR": "EGLL", "LAX": "KLAX",
    "DXB": "OMDB", "HND": "RJTT", "CDG": "LFPG", "AMS": "EHAM",
    "FRA": "EDDF", "SIN": "WSSS", "ATL": "KATL", "DFW": "KDFW",
    "DEN": "KDEN", "SFO": "KSFO", "IAD": "KIAD", "BOS": "KBOS",
    "MIA": "KMIA", "SEA": "KSEA", "EWR": "KEWR", "PHL": "KPHL",
    "LGA": "KLGA",
    "LNK": "KLNK",
    "OMA": "KOMA",
    "DSM": "KDSM",
    "MSN": "KMSN",
    "MKE": "KMKE",
    "CLE": "KCLE",
    "CMH": "KCMH",
    "IND": "KIND",
    "CVG": "KCVG",
    "BNA": "KBNA",
    "MEM": "KMEM",
    "OKC": "KOKC",
    "TUL": "KTUL",
    "ABQ": "KABQ",
    "ELP": "KELP",
    "SAT": "KSAT",
    "AUS": "KAUS",
    "HOU": "KHOU",
    "DAL": "KDAL",
}


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def norm_code(code: str) -> str:
    return (code or "").strip().upper()


def uniq_preserve(items: Sequence[str]) -> List[str]:
    """Deduplicate a list of strings while preserving order."""
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        x = (x or "").strip().upper()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def fmt_time(iso_str: Optional[str], tz_name: Optional[str] = None) -> Optional[str]:
    """Format an ISO-8601 timestamp to 'h:MM AM/PM TZ' in a given timezone."""
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if tz_name:
            try:
                local_dt = dt.astimezone(ZoneInfo(tz_name))
            except (KeyError, Exception):
                local_dt = dt.astimezone()
        else:
            local_dt = dt.astimezone()
        time_str = local_dt.strftime("%-I:%M %p %Z").strip()
        return time_str
    except Exception:
        return None


def category_to_condition(cat: str, precip_label: str = "None") -> str:
    """Map flight category + precipitation to a human-readable condition string."""
    if precip_label and precip_label != "None":
        return precip_label
    cat = (cat or "").upper()
    if cat == "VFR":
        return "Clear"
    if cat == "MVFR":
        return "Cloudy"
    if cat == "IFR":
        return "Low Visibility"
    if cat == "LIFR":
        return "Storm / Severe IFR"
    return "Unknown"


def delay_risk_label(delay_minutes: int) -> str:
    """Convert delay minutes to a human-readable risk label."""
    if delay_minutes <= 5:
        return "Low"
    if delay_minutes <= 20:
        return "Moderate"
    if delay_minutes <= 45:
        return "High"
    return "Very High"


def parse_cors_origins(raw: str) -> list:
    raw = (raw or "").strip()
    if not raw:
        return ["http://localhost:3000"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or ["http://localhost:3000"]

"""
METAR CSV → PyTorch Geometric graph snapshots for ST-GNN training.

Each hourly time slot produces one graph:
  - Nodes:     airports (from the METAR stations in the CSV)
  - Node feat: [congestion_proxy, vis_norm, wind_norm, cat_ordinal, precip_severity]
               (same 5-dim feature vector as the live inference path)
  - Target:    synthetic delay label (minutes) per node, derived from
               FAA-documented weather-to-delay correlations.

Delay-label generation rationale:
  The FAA Air Traffic Organization publishes Ground Delay Program (GDP)
  triggers keyed on flight category, precipitation type, and wind:
    - LIFR/IFR → arrival rate reductions → queuing delays
    - Snow / freezing precip → de-icing hold + runway treatment → 20–75 min
    - Thunderstorms → ground stops → 30–120 min
    - High winds → runway configuration changes → 10–30 min
  We translate these documented thresholds into a deterministic base delay
  and add calibrated noise to prevent the model from memorising a lookup table.

Edge construction:
  All airport pairs are connected (fully-connected directed graph).  Edge
  weight encodes inverse great-circle distance (normalised 0-1) so that
  nearby airports influence each other more (e.g. the JFK-EWR-LGA cluster,
  European hub cluster).
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Station normalisation: IEM uses IATA for US stations, ICAO for others
# ---------------------------------------------------------------------------

_IEM_TO_IATA: Dict[str, str] = {
    "EDDF": "FRA", "EGLL": "LHR", "EHAM": "AMS",
    "JFK": "JFK", "LAX": "LAX", "ORD": "ORD",
    "LFPG": "CDG", "OMDB": "DXB", "RJTT": "HND", "WSSS": "SIN",
}

# Coordinates for each airport (lat, lon) for edge-weight computation.
_AIRPORT_COORDS: Dict[str, Tuple[float, float]] = {
    "FRA": (50.0258, 8.5214), "LHR": (51.4785, -0.4614),
    "AMS": (52.3086, 4.7639), "JFK": (40.6386, -73.7622),
    "LAX": (33.9382, -118.3865), "ORD": (41.9742, -87.9073),
    "CDG": (49.0153, 2.5344), "DXB": (25.2539, 55.3656),
    "HND": (35.5533, 139.7811), "SIN": (1.3502, 103.9940),
}

# Base congestion score per airport (known operational capacity utilisation).
# Source: FAA OPSNET / Eurocontrol CODA average delay-per-movement rankings.
_BASE_CONGESTION: Dict[str, float] = {
    "ORD": 0.75, "JFK": 0.72, "LHR": 0.82, "LAX": 0.55,
    "FRA": 0.62, "CDG": 0.58, "AMS": 0.65, "DXB": 0.50,
    "HND": 0.70, "SIN": 0.45,
}

# Precipitation severity mapping (mirrors api_clients._PRECIP_SEVERITY).
_PRECIP_MAP: Dict[str, float] = {
    "RA": 0.3, "-RA": 0.2, "+RA": 0.5,
    "DZ": 0.15, "-DZ": 0.1, "+DZ": 0.25,
    "SHRA": 0.35, "-SHRA": 0.2, "+SHRA": 0.55,
    "SN": 0.65, "-SN": 0.45, "+SN": 0.85,
    "SHSN": 0.65, "-SHSN": 0.45, "+SHSN": 0.85,
    "FZRA": 0.95, "-FZRA": 0.7, "+FZRA": 1.0,
    "FZDZ": 0.6, "-FZDZ": 0.4, "+FZDZ": 0.75,
    "PL": 0.55, "-PL": 0.4, "+PL": 0.75,
    "IC": 0.5, "GR": 0.85, "GS": 0.55,
    "TS": 0.8, "+TS": 0.95, "TSRA": 0.85, "+TSRA": 1.0,
    "TSSN": 0.9, "TSPL": 0.85,
    "FG": 0.55, "FZFG": 0.7,
    "BR": 0.1, "HZ": 0.08, "FU": 0.1,
    "BLSN": 0.75, "BLDU": 0.4, "BLSA": 0.4,
    "SS": 0.5, "DS": 0.6,
    "SQ": 0.7, "FC": 1.0, "VA": 0.9,
}

# Flight-category ordinal (mirrors data.py CAT_MAP)
_CAT_MAP = {"VFR": 0.0, "MVFR": 0.33, "IFR": 0.66, "LIFR": 1.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dLon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or str(val).strip() in ("", "M", "T"):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _derive_flight_category(vis_miles: float, ceil_ft: Optional[float]) -> str:
    """Derive flight category from visibility and ceiling (FAA AIM 7-1-7)."""
    ceil = ceil_ft if ceil_ft is not None else 99999.0
    if vis_miles < 1.0 or ceil < 500.0:
        return "LIFR"
    if vis_miles < 3.0 or ceil < 1000.0:
        return "IFR"
    if vis_miles < 5.0 or ceil < 3000.0:
        return "MVFR"
    return "VFR"


def _get_ceiling(row: Dict[str, str]) -> Optional[float]:
    """Return lowest BKN/OVC layer height in feet, or None."""
    for i in range(1, 5):
        cover = (row.get(f"skyc{i}") or "").strip().upper()
        height = _safe_float(row.get(f"skyl{i}"), default=-1.0)
        if cover in ("BKN", "OVC") and height >= 0:
            return height
    return None


def _extract_precip_severity(wxcodes_str: str) -> Tuple[float, str]:
    """Parse IEM wxcodes column → (severity 0-1, label)."""
    raw = (wxcodes_str or "").strip()
    if not raw or raw == "M":
        return 0.0, "None"
    # wxcodes can be space-separated
    codes = [c.strip() for c in raw.replace(",", " ").split() if c.strip()]
    best_sev = 0.0
    best_code = "None"
    for code in codes:
        sev = _PRECIP_MAP.get(code, 0.0)
        if sev > best_sev:
            best_sev = sev
            best_code = code
    return best_sev, best_code


def _congestion_proxy(iata: str, hour_utc: int) -> float:
    """
    Estimate congestion from airport identity + time of day.
    Peak hours (local approximation via UTC offset) get a 30% boost.
    """
    base = _BASE_CONGESTION.get(iata, 0.5)

    # Rough UTC-to-local peak detection
    # US airports: peak 12-02 UTC (07-21 EST)
    # European: peak 06-20 UTC
    # Middle East/Asia: peak 00-14 UTC
    us = {"ORD", "JFK", "LAX"}
    eu = {"LHR", "FRA", "CDG", "AMS"}
    asia = {"HND", "SIN", "DXB"}

    if iata in us:
        peak = 12 <= hour_utc <= 2 or hour_utc >= 12
    elif iata in eu:
        peak = 6 <= hour_utc <= 20
    elif iata in asia:
        peak = hour_utc <= 14 or hour_utc >= 22
    else:
        peak = 8 <= hour_utc <= 20

    factor = 1.25 if peak else 0.80
    return _clamp(base * factor, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Delay proxy label (physics-informed, FAA-correlated)
# ---------------------------------------------------------------------------

def compute_delay_label(
    vis_miles: float,
    wind_kts: float,
    gust_kts: float,
    flight_cat: str,
    precip_sev: float,
    wxcodes_str: str,
    snowdepth: float,
    hour_utc: int,
    *,
    noise_rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Estimate delay in minutes using FAA-documented weather-delay correlations.

    Sources:
      - FAA OPSNET: GDP triggers and delay distributions
      - FAA AC 150/5200-30D: Airport Winter Operations
      - Eurocontrol CODA: Weather-delay distributions
    """
    delay = 0.0

    # 1. Visibility-based delay (arrival rate reductions)
    if vis_miles < 1.0:
        delay += 45.0    # LIFR: ~50% arrival rate reduction
    elif vis_miles < 3.0:
        delay += 20.0    # IFR: ~30% reduction
    elif vis_miles < 5.0:
        delay += 8.0     # MVFR: ~15% reduction

    # 2. Wind
    if wind_kts > 35:
        delay += 25.0
    elif wind_kts > 25:
        delay += 12.0
    elif wind_kts > 15:
        delay += 3.0

    # 3. Gust
    if gust_kts > 40:
        delay += 15.0
    elif gust_kts > 30:
        delay += 8.0

    # 4. Precipitation / wx codes (dominant delay driver)
    wx = (wxcodes_str or "").upper()
    if any(c in wx for c in ("+SN", "+FZRA", "+FZDZ")):
        delay += 55.0    # Heavy winter precip: de-icing + runway treatment
    elif any(c in wx for c in ("SN", "FZRA", "FZDZ", "PL", "IC")):
        delay += 35.0
    elif any(c in wx for c in ("-SN", "-FZRA")):
        delay += 18.0
    elif any(c in wx for c in ("TS", "+TS", "TSRA")):
        delay += 40.0    # Thunderstorm: ground stop
    elif any(c in wx for c in ("+RA", "+SHRA")):
        delay += 12.0
    elif any(c in wx for c in ("RA", "SHRA", "-SHRA", "DZ")):
        delay += 5.0
    elif any(c in wx for c in ("FG", "FZFG")):
        delay += 30.0    # Fog: major visibility impact
    elif any(c in wx for c in ("BR", "HZ")):
        delay += 3.0

    # 5. Snow depth (runway contamination → extra clearing time)
    if snowdepth > 0:
        delay += min(snowdepth * 3.0, 30.0)

    # 6. Time-of-day congestion multiplier (peak hours compound delays)
    if 12 <= hour_utc <= 23:
        delay *= 1.25
    elif 6 <= hour_utc < 12:
        delay *= 1.10
    else:
        delay *= 0.80

    # 7. Add heteroscedastic noise (prevent memorisation)
    if noise_rng is not None:
        sigma = max(2.0, delay * 0.15)
        delay += noise_rng.normal(0.0, sigma)

    return max(0.0, delay)


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------

def build_edge_index_and_attr(airports: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sparse route-based directed graph.

    Uses real-world high-traffic route pairs instead of full connectivity.
    This prevents GCN over-smoothing (which collapses all node representations
    to the same value in a small fully-connected graph after 2 hops).

    Edges are bidirectional with inverse-distance weights.
    """
    # Major real-world route pairs (high-traffic connections)
    _ROUTE_PAIRS = [
        # US domestic
        ("ORD", "JFK"), ("ORD", "LAX"), ("JFK", "LAX"),
        # Transatlantic
        ("ORD", "LHR"), ("JFK", "LHR"), ("JFK", "CDG"), ("JFK", "AMS"),
        ("ORD", "FRA"),
        # European intra
        ("LHR", "FRA"), ("LHR", "AMS"), ("LHR", "CDG"),
        ("FRA", "AMS"), ("FRA", "CDG"), ("AMS", "CDG"),
        # Middle East / Asia connections
        ("DXB", "LHR"), ("DXB", "SIN"), ("DXB", "FRA"),
        ("HND", "SIN"), ("LAX", "HND"), ("LAX", "SIN"),
        ("LHR", "HND"), ("LHR", "SIN"),
    ]

    idx = {a: i for i, a in enumerate(airports)}
    src, dst, weights = [], [], []

    for a, b in _ROUTE_PAIRS:
        if a not in idx or b not in idx:
            continue
        ca = _AIRPORT_COORDS.get(a, (0.0, 0.0))
        cb = _AIRPORT_COORDS.get(b, (0.0, 0.0))
        d = max(_haversine_km(ca[0], ca[1], cb[0], cb[1]), 100.0)
        w = _clamp(5000.0 / d, 0.1, 1.0)  # Closer → higher weight, capped

        # Bidirectional
        src.extend([idx[a], idx[b]])
        dst.extend([idx[b], idx[a]])
        weights.extend([w, w])

    if not src:
        # Fallback: self-loops
        for i in range(len(airports)):
            src.append(i)
            dst.append(i)
            weights.append(1.0)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

class METARGraphDataset:
    """
    Parses an IEM METAR CSV and produces hourly PyG graph snapshots.

    Usage:
        ds = METARGraphDataset("backend/data/raw/iem_metar_30d.csv")
        train, val, test = ds.split(train_frac=0.7, val_frac=0.15)
        for data in train:
            out = model(data.x, data.edge_index)
            loss = F.mse_loss(out.squeeze(), data.y)
    """

    def __init__(self, csv_path: str | Path, *, seed: int = 42):
        self.csv_path = Path(csv_path)
        self.rng = np.random.default_rng(seed)

        # Parse CSV → hourly buckets
        self._raw_rows = self._read_csv()
        self._airports = sorted(_IEM_TO_IATA.values())
        self._airport_to_idx = {a: i for i, a in enumerate(self._airports)}

        # Precompute edges (same for all snapshots)
        self._edge_index, self._edge_attr = build_edge_index_and_attr(self._airports)

        # Build graph snapshots
        self.snapshots: List[Data] = self._build_snapshots()

    def _read_csv(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def _bucket_by_hour(self) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        """
        Group rows by (hour_key, iata_code).
        hour_key = "YYYY-MM-DD HH" (truncated to hour).
        Within each (hour, station) we keep the *last* observation (closest to top of hour).
        """
        buckets: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
        for row in self._raw_rows:
            station_raw = (row.get("station") or "").strip().upper()
            iata = _IEM_TO_IATA.get(station_raw)
            if iata is None:
                continue
            valid_str = (row.get("valid") or "").strip()
            if not valid_str:
                continue
            # Truncate to hour
            hour_key = valid_str[:13]  # "2026-01-09 23"
            buckets[hour_key][iata].append(row)
        return buckets

    def _aggregate_hour(self, rows: List[Dict[str, str]]) -> Dict[str, Any]:
        """Pick the last METAR in the hour and extract features."""
        row = rows[-1]  # Last observation (most representative)

        vis = _safe_float(row.get("vsby"), 10.0)
        wind = _safe_float(row.get("sknt"), 0.0)
        gust = _safe_float(row.get("gust"), 0.0)
        snowdepth = _safe_float(row.get("snowdepth"), 0.0)
        wxcodes = (row.get("wxcodes") or "").strip()

        ceiling = _get_ceiling(row)
        flight_cat = _derive_flight_category(vis, ceiling)
        precip_sev, precip_label = _extract_precip_severity(wxcodes)

        return {
            "vis_miles": vis,
            "wind_kts": wind,
            "gust_kts": gust,
            "flight_cat": flight_cat,
            "precip_sev": precip_sev,
            "precip_label": precip_label,
            "wxcodes": wxcodes,
            "snowdepth": snowdepth,
        }

    def _propagate_delays(self, raw_delays: List[float]) -> List[float]:
        """
        Simulate network delay propagation: delays at one hub spill over
        to connected hubs via connecting passengers (weighted by inverse distance).
        This gives the GNN something real to learn: spatial correlation.
        """
        alpha = 0.20  # 20% of delay comes from network neighbours
        airports = self._airports
        propagated = list(raw_delays)

        for i, ap_i in enumerate(airports):
            ci = _AIRPORT_COORDS.get(ap_i, (0.0, 0.0))
            w_sum = 0.0
            w_total = 0.0
            for j, ap_j in enumerate(airports):
                if i == j:
                    continue
                cj = _AIRPORT_COORDS.get(ap_j, (0.0, 0.0))
                dist = max(_haversine_km(ci[0], ci[1], cj[0], cj[1]), 100.0)
                w = 1.0 / dist
                w_sum += w * raw_delays[j]
                w_total += w
            if w_total > 0:
                neighbour_influence = w_sum / w_total
                propagated[i] = (1.0 - alpha) * raw_delays[i] + alpha * neighbour_influence

        return propagated

    def _build_snapshots(self) -> List[Data]:
        buckets = self._bucket_by_hour()
        snapshots: List[Data] = []
        n = len(self._airports)

        # Sort by hour for temporal split later
        sorted_hours = sorted(buckets.keys())

        prev_delays: Optional[List[float]] = None  # For temporal persistence

        for hour_key in sorted_hours:
            station_data = buckets[hour_key]

            # Skip hours where fewer than half the airports have data
            if len(station_data) < n // 2:
                continue

            features = []  # [n_airports, 5]
            raw_labels = []    # [n_airports]
            hour_utc = int(hour_key[-2:])  # last 2 chars = hour

            for iata in self._airports:
                if iata in station_data:
                    agg = self._aggregate_hour(station_data[iata])
                else:
                    # Missing station this hour → default (clear weather, no delay)
                    agg = {
                        "vis_miles": 10.0, "wind_kts": 0.0, "gust_kts": 0.0,
                        "flight_cat": "VFR", "precip_sev": 0.0,
                        "precip_label": "None", "wxcodes": "", "snowdepth": 0.0,
                    }

                congestion = _congestion_proxy(iata, hour_utc)
                vis_norm = _clamp(agg["vis_miles"] / 10.0, 0.0, 1.0)
                wind_norm = _clamp(agg["wind_kts"] / 50.0, 0.0, 1.0)
                cat_val = _CAT_MAP.get(agg["flight_cat"], 0.0)
                precip_sev = agg["precip_sev"]

                features.append([congestion, vis_norm, wind_norm, cat_val, precip_sev])

                delay = compute_delay_label(
                    agg["vis_miles"], agg["wind_kts"], agg["gust_kts"],
                    agg["flight_cat"], precip_sev, agg["wxcodes"],
                    agg["snowdepth"], hour_utc,
                    noise_rng=self.rng,
                )
                raw_labels.append(delay)

            # Spatial propagation: delays at one hub affect neighbours
            labels = self._propagate_delays(raw_labels)

            # Temporal persistence: delays are "sticky" (blend with previous hour)
            if prev_delays is not None:
                temporal_alpha = 0.15  # 15% from previous hour
                labels = [
                    (1.0 - temporal_alpha) * labels[i] + temporal_alpha * prev_delays[i]
                    for i in range(n)
                ]
            prev_delays = list(labels)

            x = torch.tensor(features, dtype=torch.float)        # [N, 5]
            y = torch.tensor(labels, dtype=torch.float)           # [N]
            data = Data(
                x=x,
                edge_index=self._edge_index.clone(),
                edge_attr=self._edge_attr.clone(),
                y=y,
                num_nodes=n,
            )
            # Store hour key as metadata for temporal split
            data.hour_key = hour_key
            snapshots.append(data)

        return snapshots

    def split(
        self,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
    ) -> Tuple[List[Data], List[Data], List[Data]]:
        """Temporal split (no data leakage): first X% train, next Y% val, rest test."""
        n = len(self.snapshots)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train = self.snapshots[:train_end]
        val = self.snapshots[train_end:val_end]
        test = self.snapshots[val_end:]
        return train, val, test

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics for logging."""
        all_y = torch.cat([d.y for d in self.snapshots])
        return {
            "total_snapshots": len(self.snapshots),
            "num_airports": len(self._airports),
            "airports": self._airports,
            "num_edges": self._edge_index.shape[1],
            "delay_mean": float(all_y.mean()),
            "delay_std": float(all_y.std()),
            "delay_median": float(all_y.median()),
            "delay_max": float(all_y.max()),
            "delay_min": float(all_y.min()),
            "pct_nonzero_delay": float((all_y > 0.5).sum() / all_y.numel() * 100),
        }

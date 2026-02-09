from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import torch
from torch_geometric.data import Data

from api_clients import (
    AeroAPIClient,
    CheckWXClient,
    OpenSkyClient,
    _clamp,
    _extract_precip_severity,
)


def _uniq_preserve(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        x = (x or "").strip().upper()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _fmt_time(iso_str: Optional[str], tz_name: Optional[str] = None) -> Optional[str]:
    """Format an ISO-8601 timestamp to 'h:MM AM/PM TZ' in a given timezone.

    If *tz_name* is provided (e.g. 'America/Chicago'), the time is shown in
    that zone.  Otherwise falls back to the system local timezone.
    """
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if tz_name:
            try:
                local_dt = dt.astimezone(ZoneInfo(tz_name))
            except (KeyError, Exception):
                local_dt = dt.astimezone()  # system local
        else:
            local_dt = dt.astimezone()  # system local
        # "3:44 PM EST"
        time_str = local_dt.strftime("%-I:%M %p %Z").strip()
        return time_str
    except Exception:
        return None


def _category_to_condition(cat: str, precip_label: str = "None") -> str:
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAT_MAP = {"VFR": 0.0, "MVFR": 0.33, "IFR": 0.66, "LIFR": 1.0}

# Major hub airports included in every graph for network context.
BASE_AIRPORTS = [
    "ORD", "JFK", "LHR", "LAX", "DXB", "HND", "CDG", "AMS", "FRA", "SIN",
    "ATL", "DFW", "DEN", "SFO", "IAD", "BOS", "MIA", "SEA", "EWR", "PHL",
]

# Approximate lat/lon for known airports (avoids API calls for graph viz).
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
}

# MVP IATA -> ICAO mapping (used when AeroAPI is unavailable).
IATA_TO_ICAO = {
    "ORD": "KORD", "JFK": "KJFK", "LHR": "EGLL", "LAX": "KLAX",
    "DXB": "OMDB", "HND": "RJTT", "CDG": "LFPG", "AMS": "EHAM",
    "FRA": "EDDF", "SIN": "WSSS", "ATL": "KATL", "DFW": "KDFW",
    "DEN": "KDEN", "SFO": "KSFO", "IAD": "KIAD", "BOS": "KBOS",
    "MIA": "KMIA", "SEA": "KSEA", "EWR": "KEWR", "PHL": "KPHL",
}


# ---------------------------------------------------------------------------
# Graph Handler
# ---------------------------------------------------------------------------

class AviationGraphHandler:
    """
    Builds per-request aviation graphs and runs inference.

    Supports two modes:
    1. **Track mode** (`get_prediction_for_flight`): predict delay for a specific booked flight.
    2. **Suggest mode** (`suggest_routes`): given origin/dest, find and rank itineraries by delay risk.
    """

    def __init__(self):
        self.opensky = OpenSkyClient()
        self.aeroapi = AeroAPIClient()
        self.checkwx = CheckWXClient()

    # -- ICAO resolution ----------------------------------------------------

    def _resolve_icao(self, iata_code: str) -> Optional[str]:
        iata = (iata_code or "").strip().upper()
        if not iata:
            return None
        if iata in IATA_TO_ICAO:
            return IATA_TO_ICAO[iata]
        info = self.aeroapi.get_airport_info(iata)
        if info and info.icao:
            return info.icao
        return None

    # -- Graph construction -------------------------------------------------

    def build_graph(
        self,
        origin_iata: str,
        destination_iata: str,
        *,
        airports: Optional[Sequence[str]] = None,
        origin_lat: Optional[float] = None,
        origin_lon: Optional[float] = None,
        airline_prefixes: Optional[Sequence[str]] = None,
        use_taf: bool = False,
    ) -> Tuple[Data, Dict[str, Any]]:
        """
        Build a PyG graph with 5-dim node features:
        [congestion, visibility_norm, wind_norm, flight_category_ordinal, precip_severity]

        When `use_taf=True`, weather features come from TAF (forecast) instead of current METAR.
        """
        origin_iata = (origin_iata or "").strip().upper()
        destination_iata = (destination_iata or "").strip().upper()

        airports_list = _uniq_preserve(list(airports) if airports else list(BASE_AIRPORTS))
        if origin_iata:
            airports_list = _uniq_preserve(airports_list + [origin_iata])
        if destination_iata:
            airports_list = _uniq_preserve(airports_list + [destination_iata])

        airport_to_idx = {code: i for i, code in enumerate(airports_list)}

        metar_by_iata: Dict[str, Dict[str, Any]] = {}
        taf_by_iata: Dict[str, Dict[str, Any]] = {}
        congestion_by_iata: Dict[str, float] = {}
        node_features: List[List[float]] = []

        for iata in airports_list:
            icao = self._resolve_icao(iata) or iata

            # Current METAR (always fetched for UI display)
            metar = self.checkwx.get_metar_data(icao) or {}
            metar_by_iata[iata] = metar

            # TAF (forecast) if requested
            taf: Dict[str, Any] = {}
            if use_taf:
                taf = self.checkwx.get_taf_data(icao) or {}
            taf_by_iata[iata] = taf

            congestion = float(self.aeroapi.get_gate_congestion(icao or iata))
            congestion_by_iata[iata] = congestion

            # Choose weather source: TAF for suggest mode, METAR for track mode
            if use_taf and taf:
                vis_raw = float(taf.get("forecast_visibility_miles", 10.0) or 10.0)
                wind_raw = float(taf.get("forecast_wind_speed_kts", 0.0) or 0.0)
                cat_str = str(taf.get("forecast_flight_category", "VFR") or "VFR").upper()
                precip_sev = float(taf.get("forecast_precip_severity", 0.0) or 0.0)
            else:
                vis_raw = float(metar.get("visibility_miles", 10.0) or 10.0)
                wind_raw = float(metar.get("wind_speed_kts", 0.0) or 0.0)
                cat_str = str(metar.get("flight_category", "VFR") or "VFR").upper()
                precip_sev = float(metar.get("precip_severity", 0.0) or 0.0)

            vis_norm = _clamp(vis_raw / 10.0, 0.0, 1.0)
            wind_norm = _clamp(wind_raw / 50.0, 0.0, 1.0)
            cat_val = float(CAT_MAP.get(cat_str, 0.0))

            node_features.append([congestion, vis_norm, wind_norm, cat_val, precip_sev])

        x = torch.tensor(node_features, dtype=torch.float)

        # --- Edges ---
        src_nodes: List[int] = []
        dst_nodes: List[int] = []
        edge_attrs: List[List[float]] = []

        incoming_count = 0
        prefixes = [p.strip().upper() for p in (airline_prefixes or []) if p and p.strip()]
        if not prefixes:
            prefixes = ["UAL", "AAL"]

        for pref in prefixes:
            incoming = self.opensky.get_incoming_aircraft_distance(
                origin_iata or "ORD", pref,
                airport_lat=origin_lat, airport_lon=origin_lon,
            )
            incoming_count += len(incoming or [])

        traffic_proxy = _clamp(incoming_count / 30.0, 0.0, 1.0)

        if origin_iata in airport_to_idx:
            origin_idx = airport_to_idx[origin_iata]
            for other_iata, other_idx in airport_to_idx.items():
                if other_iata == origin_iata:
                    continue
                src_nodes.append(other_idx)
                dst_nodes.append(origin_idx)
                edge_attrs.append([traffic_proxy])

        if origin_iata in airport_to_idx and destination_iata in airport_to_idx:
            src_nodes.append(airport_to_idx[origin_iata])
            dst_nodes.append(airport_to_idx[destination_iata])
            edge_attrs.append([max(0.1, traffic_proxy)])

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(airports_list))
        ctx = {
            "airports": airports_list,
            "airport_to_idx": airport_to_idx,
            "metar_by_iata": metar_by_iata,
            "taf_by_iata": taf_by_iata,
            "congestion_by_iata": congestion_by_iata,
            "traffic_proxy": traffic_proxy,
        }
        return data, ctx

    # -- Graph visualization data -------------------------------------------

    def _build_graph_viz(
        self,
        model,
        data: Data,
        ctx: Dict[str, Any],
        origin_iata: str,
        destination_iata: str,
    ) -> Dict[str, Any]:
        """Extract graph data for the frontend D3 visualization.

        Returns a dict with 'nodes' and 'edges' suitable for rendering
        the actual GNN network the model reasons over.
        """
        airports = ctx["airports"]
        airport_to_idx = ctx["airport_to_idx"]
        metar_by_iata = ctx.get("metar_by_iata", {})
        congestion_by_iata = ctx.get("congestion_by_iata", {})

        # Run model to get per-node predictions
        model.eval()
        log_space = getattr(model, "_log_space", False)
        with torch.no_grad():
            out = model(data.x, data.edge_index)

        # Build node list
        nodes = []
        for iata in airports:
            idx = airport_to_idx[iata]
            raw = float(out[idx].item())
            delay = math.expm1(raw) if log_space else raw * 100.0
            delay = max(0.0, delay)

            metar = metar_by_iata.get(iata, {})
            congestion = float(congestion_by_iata.get(iata, 0.0))
            precip_sev = float(metar.get("precip_severity", 0.0) or 0.0)
            cat = str(metar.get("flight_category", "VFR") or "VFR").upper()
            condition = str(metar.get("precip_label", "None") or "None")
            wind = float(metar.get("wind_speed_kts", 0.0) or 0.0)
            vis = float(metar.get("visibility_miles", 10.0) or 10.0)

            # Classify node role
            if iata == origin_iata:
                role = "origin"
            elif iata == destination_iata:
                role = "destination"
            else:
                role = "hub"

            # Compute a 0-1 risk score for coloring
            risk = _clamp(delay / 30.0, 0.0, 1.0)

            # Get lat/lon: prefer static map (no API call), fall back to cached API
            if iata in AIRPORT_COORDS:
                lat, lon = AIRPORT_COORDS[iata]
            else:
                info = self.aeroapi.get_airport_info(iata)
                lat = info.latitude if info else None
                lon = info.longitude if info else None

            nodes.append({
                "id": iata,
                "role": role,
                "risk": round(risk, 2),
                "predictedDelay": round(delay, 1),
                "congestion": round(congestion, 2),
                "precipSeverity": round(precip_sev, 2),
                "condition": condition if condition != "None" else cat,
                "wind": round(wind, 0),
                "visibility": round(vis, 1),
                "lat": lat,
                "lon": lon,
            })

        # Build edge list from the PyG edge_index
        edges = []
        edge_index = data.edge_index
        if edge_index.numel() > 0:
            num_edges = edge_index.shape[1]
            for e in range(num_edges):
                src_idx = int(edge_index[0, e].item())
                dst_idx = int(edge_index[1, e].item())
                if src_idx < len(airports) and dst_idx < len(airports):
                    src_iata = airports[src_idx]
                    dst_iata = airports[dst_idx]
                    # Classify the edge
                    if src_iata == origin_iata and dst_iata == destination_iata:
                        edge_type = "flight"
                    elif dst_iata == origin_iata:
                        edge_type = "inbound"
                    else:
                        edge_type = "network"
                    edges.append({
                        "source": src_iata,
                        "target": dst_iata,
                        "type": edge_type,
                    })

        return {"nodes": nodes, "edges": edges}

    # -- Shared helpers -----------------------------------------------------

    def _run_model(self, model, data: Data, ctx: Dict[str, Any], dest_iata: str) -> int:
        """
        Run the ST-GNN and return predicted delay in minutes for the given airport.

        If the model was trained in log-space (model._log_space == True, set by load_model
        when trained weights are loaded), output is interpreted as log(1 + delay_minutes)
        and converted back via expm1().  Otherwise falls back to the legacy scale multiplier.
        """
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
        dest_idx = ctx["airport_to_idx"].get(dest_iata, 0)
        raw_pred = float(out[dest_idx].item())

        if getattr(model, "_log_space", False):
            # Trained model: output is log(1 + delay_minutes)
            delay = math.expm1(raw_pred)
        else:
            # Untrained / legacy: scale the raw output
            scale = float(os.environ.get("STGNN_OUTPUT_SCALE", "100") or 100.0)
            delay = raw_pred * scale

        return max(0, int(delay))

    def _compute_risk_fields(self, metar: Dict, congestion: float, traffic_proxy: float, precip_sev: float) -> Dict[str, Any]:
        cat_sev = float(CAT_MAP.get(str(metar.get("flight_category", "VFR")).upper(), 0.0))
        wind_sev = _clamp(float(metar.get("wind_speed_kts", 0.0) or 0.0) / 50.0, 0.0, 1.0)
        weather_sev = max(cat_sev, wind_sev, precip_sev)

        propagation_risk = int(_clamp(
            0.35 * weather_sev + 0.25 * precip_sev + 0.25 * traffic_proxy + 0.15 * congestion,
            0.0, 1.0
        ) * 100)

        return {
            "weatherSeverity": int(weather_sev * 100),
            "precipSeverity": int(precip_sev * 100),
            "propagationRisk": propagation_risk,
        }

    def _metar_widget(self, metar: Dict[str, Any]) -> Dict[str, Any]:
        cat = str(metar.get("flight_category", "VFR"))
        precip_label = str(metar.get("precip_label", "None") or "None")
        return {
            "temp": int(metar.get("temp_f") or 0),
            "condition": _category_to_condition(cat, precip_label),
            "windSpeed": int(float(metar.get("wind_speed_kts", 0) or 0)),
            "visibility": f"{int(float(metar.get('visibility_miles', 10) or 10))} mi",
            "precipSeverity": int(float(metar.get("precip_severity", 0) or 0) * 100),
            "precipLabel": precip_label,
        }

    # -----------------------------------------------------------------------
    # MODE 1: Track a specific flight (existing behavior, now with precip)
    # -----------------------------------------------------------------------

    def get_prediction_for_flight(self, flight_number: str, model) -> Dict[str, Any]:
        flight_number = (flight_number or "").strip().upper()
        if not flight_number:
            return {"error": "INVALID_FLIGHT", "detail": "Flight number is required.", "_status_code": 400}

        real_status = self.aeroapi.get_flight_status(flight_number)
        if isinstance(real_status, dict) and real_status.get("error"):
            code = str(real_status.get("error"))
            status_code = 429 if code == "API_QUOTA_EXCEEDED" else 503
            return {**real_status, "_status_code": status_code}

        origin_obj = real_status.get("origin") or {}
        dest_obj = real_status.get("destination") or {}
        origin = (origin_obj.get("code") or "").strip().upper()
        destination = (dest_obj.get("code") or "").strip().upper()
        origin_tz = origin_obj.get("timezone")       # e.g. "America/New_York"
        dest_tz = dest_obj.get("timezone")             # e.g. "America/Chicago"
        origin_iata = (origin_obj.get("code_iata") or "").strip().upper()
        dest_iata = (dest_obj.get("code_iata") or "").strip().upper()
        if not origin or not destination:
            return {
                "error": "FLIGHT_NOT_FOUND",
                "detail": f"No live route data for {flight_number}. Flight may be outside tracking window.",
                "_status_code": 404,
            }

        prefixes: List[str] = []
        operator_icao = (real_status.get("operator_icao") or "").strip().upper()
        if len(operator_icao) == 3:
            prefixes.append(operator_icao)

        origin_info = self.aeroapi.get_airport_info(origin)
        origin_lat = origin_info.latitude if origin_info else None
        origin_lon = origin_info.longitude if origin_info else None

        # Use IATA codes for graph nodes (deduplicates with BASE_AIRPORTS)
        o_graph = origin_iata or origin
        d_graph = dest_iata or destination
        airports = _uniq_preserve(list(BASE_AIRPORTS) + [o_graph, d_graph])

        data, ctx = self.build_graph(
            o_graph, d_graph,
            airports=airports,
            origin_lat=origin_lat, origin_lon=origin_lon,
            airline_prefixes=prefixes,
            use_taf=False,
        )

        predicted_delay_minutes = self._run_model(model, data, ctx, d_graph)

        # Graph visualization data for the frontend
        graph_viz = self._build_graph_viz(model, data, ctx, o_graph, d_graph)

        # Live position
        position_data = self.aeroapi.get_flight_position(flight_number) or {}
        live_lat = position_data.get("latitude")
        live_lon = position_data.get("longitude")
        live_heading = position_data.get("heading")
        live_alt = position_data.get("altitude")

        # Terminal / gate
        terminal_origin = real_status.get("terminal_origin", "-") or "-"
        gate_origin = real_status.get("gate_origin", "-") or "-"
        terminal_dest = real_status.get("terminal_destination", "-") or "-"
        gate_dest = real_status.get("gate_destination", "-") or "-"
        baggage = real_status.get("baggage_claim", "-") or "-"

        # Times
        scheduled_out_raw = real_status.get("scheduled_out")
        actual_out_raw = real_status.get("actual_out")
        estimated_out_raw = real_status.get("estimated_out")
        scheduled_in_raw = real_status.get("scheduled_in")
        actual_in_raw = real_status.get("actual_in")
        estimated_in_raw = real_status.get("estimated_in")

        sched_dep = _fmt_time(scheduled_out_raw, origin_tz) or "TBD"
        actual_dep = _fmt_time(actual_out_raw, origin_tz) or _fmt_time(estimated_out_raw, origin_tz) or "TBD"
        sched_arr = _fmt_time(scheduled_in_raw, dest_tz) or "TBD"
        actual_arr = _fmt_time(actual_in_raw, dest_tz) or _fmt_time(estimated_in_raw, dest_tz) or "TBD"
        dep_time_kind = "actual" if actual_out_raw else ("estimated" if estimated_out_raw else "unknown")
        arr_time_kind = "actual" if actual_in_raw else ("estimated" if estimated_in_raw else "unknown")

        # ---------------------------------------------------------------
        # Compute OBSERVED delay from AeroAPI times (factual, not predicted)
        # ---------------------------------------------------------------
        observed_dep_delay = 0
        # Method 1: actual_out vs scheduled_out
        best_dep = actual_out_raw or estimated_out_raw
        if best_dep and scheduled_out_raw:
            try:
                dep_dt = datetime.fromisoformat(best_dep.replace("Z", "+00:00"))
                sched_dep_dt = datetime.fromisoformat(scheduled_out_raw.replace("Z", "+00:00"))
                observed_dep_delay = max(0, int((dep_dt - sched_dep_dt).total_seconds() / 60))
            except Exception:
                pass
        # Method 2: AeroAPI's direct departure_delay field (in seconds)
        aero_dep_delay = real_status.get("departure_delay")
        if aero_dep_delay is not None:
            try:
                observed_dep_delay = max(observed_dep_delay, max(0, int(float(aero_dep_delay) / 60)))
            except (ValueError, TypeError):
                pass

        observed_arr_delay = 0
        best_arr = actual_in_raw or estimated_in_raw
        if best_arr and scheduled_in_raw:
            try:
                arr_dt = datetime.fromisoformat(best_arr.replace("Z", "+00:00"))
                sched_arr_dt = datetime.fromisoformat(scheduled_in_raw.replace("Z", "+00:00"))
                observed_arr_delay = max(0, int((arr_dt - sched_arr_dt).total_seconds() / 60))
            except Exception:
                pass
        aero_arr_delay = real_status.get("arrival_delay")
        if aero_arr_delay is not None:
            try:
                observed_arr_delay = max(observed_arr_delay, max(0, int(float(aero_arr_delay) / 60)))
            except (ValueError, TypeError):
                pass

        # Final delay = max(observed, model-predicted)
        # If the plane already departed late, that's a FACT, not a prediction.
        observed_delay = max(observed_dep_delay, observed_arr_delay)
        final_delay_minutes = max(observed_delay, predicted_delay_minutes)

        # ---------------------------------------------------------------
        # Status badge — uses AeroAPI status, observed delay, and model
        # ---------------------------------------------------------------
        aero_status_raw = str(real_status.get("status") or "").lower()
        cancelled = bool(real_status.get("cancelled") or "cancelled" in aero_status_raw)
        aero_says_delayed = "delayed" in aero_status_raw
        progress_pct = real_status.get("progress_percent")

        # Determine if the flight has arrived
        aero_has_arrived = (
            "arrived" in aero_status_raw
            or "landed" in aero_status_raw
            or "gate arrival" in aero_status_raw
            or (isinstance(progress_pct, (int, float)) and progress_pct >= 100)
        )
        aero_en_route = (
            "en route" in aero_status_raw
            or (isinstance(progress_pct, (int, float)) and 0 < progress_pct < 100)
        )

        status_text = "On Time"
        if cancelled:
            status_text = "Cancelled"
        elif aero_has_arrived:
            # Flight has LANDED — show arrival-centric status
            if final_delay_minutes > 45:
                status_text = "Arrived - Severely Delayed"
            elif final_delay_minutes > 15:
                status_text = "Arrived - Delayed"
            elif final_delay_minutes > 0:
                status_text = "Arrived - Slight Delay"
            elif observed_dep_delay == 0 and best_dep and scheduled_out_raw and str(best_dep) < str(scheduled_out_raw):
                status_text = "Arrived - Early"
            else:
                status_text = "Arrived"
        elif final_delay_minutes > 60:
            status_text = "Severely Delayed"
        elif final_delay_minutes > 15 or aero_says_delayed:
            status_text = "Delayed"
        elif final_delay_minutes > 0 and final_delay_minutes <= 15:
            status_text = "Slight Delay"
        elif observed_dep_delay == 0 and best_dep and scheduled_out_raw and str(best_dep) < str(scheduled_out_raw):
            status_text = "Early"

        # Suffix for en-route flights
        if aero_en_route and not aero_has_arrived:
            if "Delayed" in status_text or "Severely" in status_text:
                status_text += " - En Route"

        metar_o = ctx["metar_by_iata"].get(origin) or {}
        metar_d = ctx["metar_by_iata"].get(destination) or {}
        congestion_origin = float(ctx["congestion_by_iata"].get(origin, 0.0) or 0.0)
        traffic_proxy = float(ctx.get("traffic_proxy", 0.0) or 0.0)
        precip_sev_origin = float(metar_o.get("precip_severity", 0.0) or 0.0)

        risk = self._compute_risk_fields(metar_o, congestion_origin, traffic_proxy, precip_sev_origin)
        delay_probability = int(_clamp(final_delay_minutes / 120.0, 0.0, 1.0) * 90 + 5)

        # Determine aircraft status from multiple sources
        incoming_aircraft_status = "Unknown"
        if aero_has_arrived:
            incoming_aircraft_status = "Landed"
        elif aero_en_route:
            incoming_aircraft_status = "In Air"
        else:
            incoming_aircraft_status = "At Gate"
        # Override with live altitude if available (more precise)
        try:
            if live_alt is not None:
                if float(live_alt) <= 0:
                    incoming_aircraft_status = "Landed"
                else:
                    incoming_aircraft_status = "In Air"
        except Exception:
            pass

        airline = (
            (real_status.get("operator") or "").strip()
            or (real_status.get("operator_iata") or "").strip()
            or (real_status.get("operator_icao") or "").strip()
            or "Unknown Airline"
        )

        return {
            "flightNumber": flight_number,
            "origin": origin_iata or origin,            # prefer IATA (ORD) over ICAO (KORD)
            "destination": dest_iata or destination,
            "originIcao": origin,
            "destinationIcao": destination,
            "status": status_text,
            "predictedDelayMinutes": final_delay_minutes,
            "observedDelayMinutes": observed_delay,
            "modelPredictedDelay": predicted_delay_minutes,
            "scheduledDep": sched_dep,
            "actualDep": actual_dep,
            "depTimeKind": dep_time_kind,
            "predictedTakeoff": actual_dep,
            "scheduledArr": sched_arr,
            "actualArr": actual_arr,
            "arrTimeKind": arr_time_kind,
            "terminalOrigin": terminal_origin,
            "gateOrigin": gate_origin,
            "terminalDest": terminal_dest,
            "gateDest": gate_dest,
            "baggageClaim": baggage,
            "delayProbability": delay_probability,
            "networkCongestion": int(congestion_origin * 100),
            "propagationRisk": risk["propagationRisk"],
            "precipSeverity": risk["precipSeverity"],
            "incomingAircraftStatus": incoming_aircraft_status,
            "note": "Live route data. Delay = max(observed from AeroAPI, model prediction).",
            "airline": airline,
            "weatherOrigin": self._metar_widget(metar_o),
            "weatherDest": self._metar_widget(metar_d),
            "livePosition": {
                "lat": live_lat, "lon": live_lon,
                "heading": live_heading, "altitude": live_alt,
            } if live_lat is not None and live_lon is not None else None,
            "graphData": graph_viz,
            "sources": [
                {"title": "FlightAware AeroAPI", "uri": "https://flightaware.com/commercial/aeroapi/"},
                {"title": "OpenSky Network", "uri": "https://opensky-network.org/"},
                {"title": "CheckWX", "uri": "https://www.checkwxapi.com/"},
            ],
        }

    # -----------------------------------------------------------------------
    # MODE 2: Suggest alternative routes (pre-departure, TAF-based)
    # -----------------------------------------------------------------------

    def suggest_routes(self, origin: str, destination: str, model, *, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Given origin/destination, enumerate candidate itineraries and rank by predicted delay risk.

        Returns a response with:
        - origin/destination weather (current + forecast)
        - ranked list of itineraries (direct + 1-stop via connection hubs)
        """
        origin = (origin or "").strip().upper()
        destination = (destination or "").strip().upper()
        if not origin or not destination:
            return {"error": "INVALID_ROUTE", "detail": "Both origin and destination are required.", "_status_code": 400}
        if origin == destination:
            return {"error": "INVALID_ROUTE", "detail": "Origin and destination must be different.", "_status_code": 400}

        # Resolve airport info
        origin_info = self.aeroapi.get_airport_info(origin)
        dest_info = self.aeroapi.get_airport_info(destination)
        origin_lat = origin_info.latitude if origin_info else None
        origin_lon = origin_info.longitude if origin_info else None
        origin_name = (origin_info.name if origin_info else None) or origin
        dest_name = (dest_info.name if dest_info else None) or destination
        suggest_origin_tz = origin_info.timezone if origin_info else None
        suggest_dest_tz = dest_info.timezone if dest_info else None

        # Parse optional date
        start_iso = None
        end_iso = None
        if date_str:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                start_iso = dt.isoformat()
                end_iso = (dt + timedelta(days=1)).isoformat()
            except Exception:
                pass

        # --- Build graph using TAF (forecast) for pre-departure prediction ---
        airports = _uniq_preserve(list(BASE_AIRPORTS) + [origin, destination])
        data, ctx = self.build_graph(
            origin, destination,
            airports=airports,
            origin_lat=origin_lat, origin_lon=origin_lon,
            use_taf=True,
        )

        # --- Enumerate candidate itineraries ---
        itineraries: List[Dict[str, Any]] = []

        # 1) Direct flights
        direct_flights = self.aeroapi.get_scheduled_flights(origin, destination, start_iso=start_iso, end_iso=end_iso)
        predicted_delay_direct = self._run_model(model, data, ctx, destination)

        metar_o = ctx["metar_by_iata"].get(origin) or {}
        metar_d = ctx["metar_by_iata"].get(destination) or {}
        taf_o = ctx["taf_by_iata"].get(origin) or {}
        taf_d = ctx["taf_by_iata"].get(destination) or {}

        congestion_origin = float(ctx["congestion_by_iata"].get(origin, 0.0) or 0.0)
        traffic_proxy = float(ctx.get("traffic_proxy", 0.0) or 0.0)
        precip_sev_o = float(taf_o.get("forecast_precip_severity", 0.0) or metar_o.get("precip_severity", 0.0) or 0.0)

        risk_direct = self._compute_risk_fields(metar_o, congestion_origin, traffic_proxy, precip_sev_o)

        if direct_flights:
            for fl in direct_flights[:5]:
                ident = fl.get("ident") or fl.get("fa_flight_id", "")
                airline = (fl.get("operator") or fl.get("operator_iata") or fl.get("operator_icao") or "").strip() or "Unknown"
                sched_dep = _fmt_time(fl.get("scheduled_out") or fl.get("scheduled_off"), suggest_origin_tz) or "TBD"
                sched_arr = _fmt_time(fl.get("scheduled_in") or fl.get("scheduled_on"), suggest_dest_tz) or "TBD"

                itineraries.append({
                    "type": "direct",
                    "flightNumber": ident,
                    "airline": airline,
                    "legs": [{
                        "origin": origin,
                        "destination": destination,
                        "scheduledDep": sched_dep,
                        "scheduledArr": sched_arr,
                        "flightNumber": ident,
                    }],
                    "predictedDelayMinutes": predicted_delay_direct,
                    "delayRisk": _delay_risk_label(predicted_delay_direct),
                    "propagationRisk": risk_direct["propagationRisk"],
                    "precipSeverity": risk_direct["precipSeverity"],
                    "stops": 0,
                })
        else:
            # No scheduled flights found -- add a synthetic "direct" entry if the route is feasible
            itineraries.append({
                "type": "direct",
                "flightNumber": f"{origin}-{destination}",
                "airline": "Multiple carriers",
                "legs": [{"origin": origin, "destination": destination, "scheduledDep": "TBD", "scheduledArr": "TBD", "flightNumber": "N/A"}],
                "predictedDelayMinutes": predicted_delay_direct,
                "delayRisk": _delay_risk_label(predicted_delay_direct),
                "propagationRisk": risk_direct["propagationRisk"],
                "precipSeverity": risk_direct["precipSeverity"],
                "stops": 0,
            })

        # 2) 1-stop connections via major hubs
        connection_hubs = [h for h in ["ORD", "JFK", "ATL", "DFW", "DEN", "LAX", "SFO", "LHR", "FRA", "AMS"]
                          if h != origin and h != destination]

        for hub in connection_hubs[:6]:
            hub_icao = self._resolve_icao(hub) or hub
            hub_taf = ctx["taf_by_iata"].get(hub) or {}
            hub_metar = ctx["metar_by_iata"].get(hub) or {}
            hub_congestion = float(ctx["congestion_by_iata"].get(hub, 0.0) or 0.0)
            hub_precip = float(hub_taf.get("forecast_precip_severity", 0.0) or hub_metar.get("precip_severity", 0.0) or 0.0)

            # Predict delay for leg 1 (origin -> hub) and leg 2 (hub -> dest)
            hub_idx = ctx["airport_to_idx"].get(hub)
            dest_idx = ctx["airport_to_idx"].get(destination)
            if hub_idx is None or dest_idx is None:
                continue

            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)

            if getattr(model, "_log_space", False):
                delay_leg1 = max(0, int(math.expm1(float(out[hub_idx].item()))))
                delay_leg2 = max(0, int(math.expm1(float(out[dest_idx].item()))))
            else:
                scale = float(os.environ.get("STGNN_OUTPUT_SCALE", "100") or 100.0)
                delay_leg1 = max(0, int(float(out[hub_idx].item()) * scale))
                delay_leg2 = max(0, int(float(out[dest_idx].item()) * scale))
            total_delay = delay_leg1 + delay_leg2

            risk_hub = self._compute_risk_fields(hub_metar, hub_congestion, traffic_proxy, hub_precip)
            combined_risk = max(risk_direct["propagationRisk"], risk_hub["propagationRisk"])
            combined_precip = max(risk_direct["precipSeverity"], risk_hub["precipSeverity"])

            hub_name = hub
            hub_info = self.aeroapi.get_airport_info(hub)
            if hub_info and hub_info.name:
                hub_name = f"{hub} ({hub_info.name})"

            itineraries.append({
                "type": "connection",
                "flightNumber": f"via {hub}",
                "airline": "Multiple carriers",
                "legs": [
                    {"origin": origin, "destination": hub, "scheduledDep": "TBD", "scheduledArr": "TBD", "flightNumber": "N/A"},
                    {"origin": hub, "destination": destination, "scheduledDep": "TBD", "scheduledArr": "TBD", "flightNumber": "N/A"},
                ],
                "connectionHub": hub,
                "connectionHubName": hub_name,
                "predictedDelayMinutes": total_delay,
                "delayRisk": _delay_risk_label(total_delay),
                "propagationRisk": combined_risk,
                "precipSeverity": combined_precip,
                "stops": 1,
            })

        # Sort by predicted delay (lower = better)
        itineraries.sort(key=lambda it: (it["predictedDelayMinutes"], it["stops"]))

        # Assign rank and "recommended" flag
        for i, it in enumerate(itineraries):
            it["rank"] = i + 1
            it["recommended"] = i == 0

        return {
            "origin": origin,
            "originName": origin_name,
            "destination": destination,
            "destinationName": dest_name,
            "date": date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "itineraryCount": len(itineraries),
            "itineraries": itineraries,
            "weatherOrigin": self._metar_widget(metar_o),
            "weatherDest": self._metar_widget(metar_d),
            "forecastOrigin": {
                "visibility": f"{int(float(taf_o.get('forecast_visibility_miles', 10) or 10))} mi",
                "windSpeed": int(float(taf_o.get('forecast_wind_speed_kts', 0) or 0)),
                "flightCategory": str(taf_o.get("forecast_flight_category", "VFR")),
                "precipSeverity": int(float(taf_o.get("forecast_precip_severity", 0) or 0) * 100),
                "precipLabel": str(taf_o.get("forecast_precip_label", "None") or "None"),
            },
            "forecastDest": {
                "visibility": f"{int(float(taf_d.get('forecast_visibility_miles', 10) or 10))} mi",
                "windSpeed": int(float(taf_d.get('forecast_wind_speed_kts', 0) or 0)),
                "flightCategory": str(taf_d.get("forecast_flight_category", "VFR")),
                "precipSeverity": int(float(taf_d.get("forecast_precip_severity", 0) or 0) * 100),
                "precipLabel": str(taf_d.get("forecast_precip_label", "None") or "None"),
            },
            "sources": [
                {"title": "FlightAware AeroAPI", "uri": "https://flightaware.com/commercial/aeroapi/"},
                {"title": "OpenSky Network", "uri": "https://opensky-network.org/"},
                {"title": "CheckWX (METAR + TAF)", "uri": "https://www.checkwxapi.com/"},
            ],
        }


def _delay_risk_label(delay_minutes: int) -> str:
    if delay_minutes <= 5:
        return "Low"
    if delay_minutes <= 20:
        return "Moderate"
    if delay_minutes <= 45:
        return "High"
    return "Very High"

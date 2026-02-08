from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data

from api_clients import AeroAPIClient, CheckWXClient, OpenSkyClient


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _uniq_preserve(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        x = (x or "").strip().upper()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _fmt_time(iso_str: Optional[str]) -> Optional[str]:
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%H:%M")
    except Exception:
        return None


def _category_to_condition(cat: str) -> str:
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


class AviationGraphHandler:
    """
    Builds a per-request aviation graph (nodes=airports, edges=connections into origin + the target route)
    and runs inference for the requested flight.

    Key guarantees (P1):
    - We resolve the live flight's origin/destination FIRST (AeroAPI), then build the graph for that route,
      then run the model.
    - Airport code normalization is consistent: nodes use IATA; weather uses ICAO when available.
    """

    def __init__(self):
        self.opensky = OpenSkyClient()
        self.aeroapi = AeroAPIClient()
        self.checkwx = CheckWXClient()

        # Base hubs for the demo graph; request-specific origin/dest are added dynamically.
        self.base_airports = ["ORD", "JFK", "LHR", "LAX", "DXB", "HND", "CDG", "AMS", "FRA", "SIN"]

        # MVP mapping for common hubs (IATA -> ICAO), used when AeroAPI airport lookup is unavailable.
        self.iata_to_icao = {
            "ORD": "KORD",
            "JFK": "KJFK",
            "LHR": "EGLL",
            "LAX": "KLAX",
            "DXB": "OMDB",
            "HND": "RJTT",
            "CDG": "LFPG",
            "AMS": "EHAM",
            "FRA": "EDDF",
            "SIN": "WSSS",
        }

    def _resolve_icao(self, iata_code: str) -> Optional[str]:
        iata = (iata_code or "").strip().upper()
        if not iata:
            return None
        if iata in self.iata_to_icao:
            return self.iata_to_icao[iata]
        info = self.aeroapi.get_airport_info(iata)
        if info and info.icao:
            return info.icao
        return None

    def build_graph(
        self,
        target_flight: str,
        origin_iata: str,
        destination_iata: str,
        *,
        airports: Optional[Sequence[str]] = None,
        origin_lat: Optional[float] = None,
        origin_lon: Optional[float] = None,
        airline_prefixes: Optional[Sequence[str]] = None,
    ) -> Tuple[Data, Dict[str, Any]]:
        """
        Returns (PyG Data, ctx dict) where ctx contains mappings + raw METAR/congestion used.
        """
        origin_iata = (origin_iata or "").strip().upper()
        destination_iata = (destination_iata or "").strip().upper()

        airports_list = _uniq_preserve(list(airports) if airports else list(self.base_airports))
        if origin_iata:
            airports_list = _uniq_preserve(airports_list + [origin_iata])
        if destination_iata:
            airports_list = _uniq_preserve(airports_list + [destination_iata])

        airport_to_idx = {code: i for i, code in enumerate(airports_list)}

        # --- Node Features ---
        # Features: [Congestion (0-1), Visibility (norm), Wind (norm), Flight Category (ordinal)]
        cat_map = {"VFR": 0.0, "MVFR": 0.33, "IFR": 0.66, "LIFR": 1.0}

        metar_by_iata: Dict[str, Dict[str, Any]] = {}
        congestion_by_iata: Dict[str, float] = {}
        node_features: List[List[float]] = []

        for iata in airports_list:
            icao = self._resolve_icao(iata) or iata  # last resort: pass through
            metar = self.checkwx.get_metar_data(icao) or {}
            metar_by_iata[iata] = metar

            # Prefer ICAO for AeroAPI if we have it; fall back to IATA.
            congestion = float(self.aeroapi.get_gate_congestion(icao or iata))
            congestion_by_iata[iata] = congestion

            vis_norm = _clamp(float(metar.get("visibility_miles", 10.0) or 10.0) / 10.0, 0.0, 1.0)
            wind_norm = _clamp(float(metar.get("wind_speed_kts", 0.0) or 0.0) / 50.0, 0.0, 1.0)
            cat_val = float(cat_map.get(str(metar.get("flight_category", "VFR")).upper(), 0.0))

            node_features.append([congestion, vis_norm, wind_norm, cat_val])

        x = torch.tensor(node_features, dtype=torch.float)

        # --- Edge Index & (optional) edge_attr ---
        src_nodes: List[int] = []
        dst_nodes: List[int] = []
        edge_attrs: List[List[float]] = []

        incoming_count = 0
        prefixes = [p.strip().upper() for p in (airline_prefixes or []) if p and p.strip()]
        if not prefixes:
            prefixes = ["UAL", "AAL"]  # default heuristic

        # Traffic proxy around the true origin airport (if we have lat/lon).
        for pref in prefixes:
            incoming = self.opensky.get_incoming_aircraft_distance(
                origin_iata or "ORD",
                pref,
                airport_lat=origin_lat,
                airport_lon=origin_lon,
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

        # Ensure target flight path edge exists
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
            "congestion_by_iata": congestion_by_iata,
            "traffic_proxy": traffic_proxy,
        }
        return data, ctx

    def get_prediction_for_flight(self, flight_number: str, model) -> Dict[str, Any]:
        """
        Full pipeline (P1):
        1) Resolve live flight metadata (AeroAPI)
        2) Build graph using the live route (origin/destination)
        3) Run model inference
        4) Return a frontend-friendly response
        """
        flight_number = (flight_number or "").strip().upper()
        if not flight_number:
            return {"error": "INVALID_FLIGHT", "detail": "Flight number is required.", "_status_code": 400}

        real_status = self.aeroapi.get_flight_status(flight_number)
        if isinstance(real_status, dict) and real_status.get("error"):
            code = str(real_status.get("error"))
            status_code = 429 if code == "API_QUOTA_EXCEEDED" else 503
            return {**real_status, "_status_code": status_code}

        origin = ((real_status.get("origin") or {}).get("code") or "").strip().upper()
        destination = ((real_status.get("destination") or {}).get("code") or "").strip().upper()
        if not origin or not destination:
            return {
                "error": "FLIGHT_NOT_FOUND",
                "detail": f"No live route data available for {flight_number}. The flight may be outside the live tracking window.",
                "_status_code": 404,
            }

        # Airline prefix for OpenSky callsign filtering (best effort)
        prefixes: List[str] = []
        operator_icao = (real_status.get("operator_icao") or "").strip().upper()
        if len(operator_icao) == 3:
            prefixes.append(operator_icao)

        # Airport lat/lon for OpenSky bbox (best effort)
        origin_info = self.aeroapi.get_airport_info(origin)
        origin_lat = origin_info.latitude if origin_info else None
        origin_lon = origin_info.longitude if origin_info else None

        airports = _uniq_preserve(list(self.base_airports) + [origin, destination])

        data, ctx = self.build_graph(
            flight_number,
            origin,
            destination,
            airports=airports,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            airline_prefixes=prefixes,
        )

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)

        dest_idx = ctx["airport_to_idx"].get(destination, 0)
        raw_pred = float(out[dest_idx].item())
        scale = float(os.environ.get("STGNN_OUTPUT_SCALE", "100") or 100.0)
        predicted_delay_minutes = max(0, int(raw_pred * scale))

        # Live position (best effort)
        position_data = self.aeroapi.get_flight_position(flight_number) or {}
        live_lat = position_data.get("latitude")
        live_lon = position_data.get("longitude")
        live_heading = position_data.get("heading")
        live_alt = position_data.get("altitude")

        # Terminals / gates
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

        sched_dep = _fmt_time(scheduled_out_raw) or "TBD"
        actual_dep = _fmt_time(actual_out_raw) or _fmt_time(estimated_out_raw) or "TBD"

        sched_arr = _fmt_time(scheduled_in_raw) or "TBD"
        actual_arr = _fmt_time(actual_in_raw) or _fmt_time(estimated_in_raw) or "TBD"

        # Status badge (hybrid: cancellation from API, delay from model)
        cancelled = bool(real_status.get("cancelled") or str(real_status.get("status", "")).lower() == "cancelled")
        status_text = "On Time"
        if cancelled:
            status_text = "Cancelled"
        elif predicted_delay_minutes > 15:
            status_text = "Delayed"
        elif predicted_delay_minutes == 0 and actual_out_raw and scheduled_out_raw and str(actual_out_raw) < str(scheduled_out_raw):
            status_text = "Early"

        # METAR → UI widgets
        metar_o = (ctx["metar_by_iata"].get(origin) or {})
        metar_d = (ctx["metar_by_iata"].get(destination) or {})

        def metar_widget(metar: Dict[str, Any]) -> Dict[str, Any]:
            cat = str(metar.get("flight_category", "VFR"))
            return {
                "temp": int(metar.get("temp_f") or 0),
                "condition": _category_to_condition(cat),
                "windSpeed": int(float(metar.get("wind_speed_kts", 0) or 0)),
                "visibility": f"{int(float(metar.get('visibility_miles', 10) or 10))} mi",
            }

        weather_origin = metar_widget(metar_o)
        weather_dest = metar_widget(metar_d)

        congestion_origin = float(ctx["congestion_by_iata"].get(origin, 0.0) or 0.0)
        traffic_proxy = float(ctx.get("traffic_proxy", 0.0) or 0.0)

        # Heuristic risk aggregation (explicitly heuristic until P2 model calibration exists)
        cat_map = {"VFR": 0.0, "MVFR": 0.33, "IFR": 0.66, "LIFR": 1.0}
        weather_sev = max(
            float(cat_map.get(str(metar_o.get("flight_category", "VFR")).upper(), 0.0)),
            _clamp(float(metar_o.get("wind_speed_kts", 0.0) or 0.0) / 50.0, 0.0, 1.0),
        )
        propagation_risk = int(_clamp(0.5 * weather_sev + 0.3 * traffic_proxy + 0.2 * congestion_origin, 0.0, 1.0) * 100)

        delay_probability = int(_clamp((predicted_delay_minutes / 120.0), 0.0, 1.0) * 90 + 5)

        incoming_aircraft_status = "In Air"
        try:
            if live_alt is not None and float(live_alt) <= 0:
                incoming_aircraft_status = "Landed"
        except Exception:
            incoming_aircraft_status = "In Air"

        airline = (
            (real_status.get("operator") or "").strip()
            or (real_status.get("operator_iata") or "").strip()
            or (real_status.get("operator_icao") or "").strip()
            or "Unknown Airline"
        )

        return {
            "flightNumber": flight_number,
            "origin": origin,
            "destination": destination,
            "status": status_text,
            "predictedDelayMinutes": predicted_delay_minutes,

            # Times
            "scheduledDep": sched_dep,
            "actualDep": actual_dep,
            "predictedTakeoff": actual_dep,
            "scheduledArr": sched_arr,
            "actualArr": actual_arr,

            # Gate/Terminal
            "terminalOrigin": terminal_origin,
            "gateOrigin": gate_origin,
            "terminalDest": terminal_dest,
            "gateDest": gate_dest,
            "baggageClaim": baggage,

            # Heuristic fields (until a calibrated P2 model exists)
            "delayProbability": delay_probability,
            "networkCongestion": int(congestion_origin * 100),
            "propagationRisk": propagation_risk,
            "incomingAircraftStatus": incoming_aircraft_status,

            "note": "Live route data. Probability/risk fields are heuristics until model calibration (P2).",
            "airline": airline,

            "weatherOrigin": weather_origin,
            "weatherDest": weather_dest,

            "livePosition": {
                "lat": live_lat,
                "lon": live_lon,
                "heading": live_heading,
                "altitude": live_alt,
            }
            if live_lat is not None and live_lon is not None
            else None,
            "sources": [
                {"title": "FlightAware AeroAPI", "uri": "https://flightaware.com/commercial/aeroapi/"},
                {"title": "OpenSky Network", "uri": "https://opensky-network.org/"},
                {"title": "CheckWX", "uri": "https://www.checkwxapi.com/"},
            ],
        }

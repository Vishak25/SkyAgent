
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from typing import Tuple, Dict
from datetime import datetime

from api_clients import OpenSkyClient, AeroAPIClient, CheckWXClient

class AviationGraphHandler:
    """
    Handles the construction of the aviation graph (Nodes=Airports, Edges=Flights).
    Integrates with API Clients to fetch real-time features.
    """
    def __init__(self):
        self.opensky = OpenSkyClient()
        self.aeroapi = AeroAPIClient()
        self.checkwx = CheckWXClient()
        
        # Define a static set of major hubs for the MVP
        self.airports = ['ORD', 'JFK', 'LHR', 'LAX', 'DXB', 'HND', 'CDG', 'AMS', 'FRA', 'SIN']
        self.airport_to_idx = {code: i for i, code in enumerate(self.airports)}
        self.iata_to_icao = {
            'ORD': 'KORD', 'JFK': 'KJFK', 'LHR': 'EGLL', 'LAX': 'KLAX',
            'DXB': 'OMDB', 'HND': 'RJTT', 'CDG': 'LFPG', 'AMS': 'EHAM',
            'FRA': 'EDDF', 'SIN': 'WSSS'
        }

    def build_graph(self, target_flight: str, origin: str, destination: str) -> Data:
        """
        Constructs a PyTorch Geometric Data object representing the current state of the aviation network.
        Now uses live Data Fusion from OpenSky, AeroAPI, and CheckWX.
        """
        num_nodes = len(self.airports)
        
        # --- 1. Node Features (Data Fusion) ---
        # Features: [Congestion (0-1), Visibility (norm), Wind (norm), Flight Category (ordinal)]
        node_features = []
        
        # Flight Category Mapping: VFR=0, MVFR=0.33, IFR=0.66, LIFR=1.0 (Higher is worse)
        cat_map = {"VFR": 0.0, "MVFR": 0.33, "IFR": 0.66, "LIFR": 1.0}
        
        for code in self.airports:
            # API Calls (Live or Stubbed if key missing)
            congestion = self.aeroapi.get_gate_congestion(code) # Already 0-1
            
            # CheckWX requires ICAO
            icao = self.iata_to_icao.get(code, code)
            weather = self.checkwx.get_metar_data(icao)
            
            # Feature Scaling / Normalization
            vis_norm = min(weather.get('visibility_miles', 10) / 10.0, 1.0)
            wind_norm = min(weather.get('wind_speed_kts', 0) / 50.0, 1.0)
            cat_val = cat_map.get(weather.get('flight_category', 'VFR'), 0.0)
            # Ceiling is nice to have but we'll stick to 4 dims for now or add it? 
            # Model expects 4 dims based on initialization in main.py. 
            
            feat = [congestion, vis_norm, wind_norm, cat_val]
            node_features.append(feat)
            
        x = torch.tensor(node_features, dtype=torch.float)
        
        # --- 2. Edge Index & Features (OpenSky Integration) ---
        # We define edges dynamically based on active flight routes found by OpenSky
        
        src_nodes = []
        dst_nodes = []
        edge_attrs = []
        
        # For the MVP, we focus on the central hub (ORD/Origin) and see what's coming in
        if origin in self.airport_to_idx:
            origin_idx = self.airport_to_idx[origin]
            
            # Get actual incoming flights for United (UAL) AND American (AAL) as requested
            # Note: get_incoming_aircraft_distance in api_clients.py currently takes one airline.
            # We will make two calls or modify the client. Making two calls is safer for now.
            incoming_ual = self.opensky.get_incoming_aircraft_distance(origin, "UAL")
            incoming_aal = self.opensky.get_incoming_aircraft_distance(origin, "AAL")
            incoming_flights = incoming_ual + incoming_aal
            
            # If we find live flights, they form edges into the origin
            # For this MVP graph, since we don't have the full "Previous Airport" of every live plane,
            # we simply connect them from a "Neighbor" node or self-loops to represent traffic volume.
            
            for other_code, other_idx in self.airport_to_idx.items():
                if other_code == origin: continue
                
                # Heuristic: If we don't have full route info from OpenSky free tier, 
                # we assume some traffic from major hubs exists.
                
                # Feature: Distance between airports (static) + Active Flights (dynamic)
                # Here we mock the number of active flights on this route based on the 'incoming_flights' pool size
                # This is a proxy for "Incoming Flow"
                active_traffic_proxy = len(incoming_flights) / 50.0 # Normalize count
                
                src_nodes.append(other_idx)
                dst_nodes.append(origin_idx)
                edge_attrs.append([active_traffic_proxy])
                
        # Ensure at least the target flight path exists
        if origin in self.airport_to_idx and destination in self.airport_to_idx:
            src = self.airport_to_idx[origin]
            dst = self.airport_to_idx[destination]
            src_nodes.append(src)
            dst_nodes.append(dst)
            edge_attrs.append([0.5]) # Average load
                    
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # Resizing fix if no edges found (rare)
        if edge_index.numel() == 0:
             edge_index = torch.empty((2, 0), dtype=torch.long)
             edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

    def get_prediction_for_flight(self, flight_number: str, model) -> Dict:
        """
        Runs the full pipeline: Fetch Data -> Build Graph -> Run Model -> Return Result
        """
        # 1. Resolve Flight Metadata (Mock for now, would typically come from AeroAPI)
        origin = 'ORD'
        destination = 'LHR'
        
        # 2. Build Graph
        # print("Building Aviation Graph...")
        data = self.build_graph(flight_number, origin, destination)
        
        # 3. Run Inference
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            
        # Extract prediction for the destination node (representing arrival delay)
        dest_idx = self.airport_to_idx.get(destination, 0)
        predicted_delay_timesteps = out[dest_idx].item()
        
        # Convert abstract model output to minutes (scaling factor)
        predicted_delay_minutes = max(0, int(predicted_delay_timesteps * 100))
        
        # Get Real Status from AeroAPI (or stub)
        real_status = self.aeroapi.get_flight_status(flight_number)
        
        # Check for explicit API errors (e.g. Quota Limit)
        if real_status.get("error"):
             return {
                 "error": real_status["error"],
                 "detail": real_status.get("detail", "External API Error")
             }
        
        # Validation: If flight not found in live window, DO NOT use MVP defaults.
        if not real_status or not real_status.get('origin'):
             return {
                 "error": "Flight Not Found",
                 "detail": f"Simulation data unavailable for {flight_number}. Flight is likely outside the live tracking window."
             }

        origin = real_status.get('origin', {}).get('code', origin)
        destination = real_status.get('destination', {}).get('code', destination)
        note = "Live Route Data"

        # --- Live Position Tracking (AeroAPI) ---
        position_data = self.aeroapi.get_flight_position(flight_number)
        live_lat = position_data.get('latitude')
        live_lon = position_data.get('longitude')
        live_heading = position_data.get('heading')
        live_alt = position_data.get('altitude')

        # --- Data Extraction for Detailed Cards ---
        terminal_origin = real_status.get('terminal_origin', '-') or '-'
        gate_origin = real_status.get('gate_origin', '-') or '-'
        terminal_dest = real_status.get('terminal_destination', '-') or '-'
        gate_dest = real_status.get('gate_destination', '-') or '-'
        baggage = real_status.get('baggage_claim', '-') or '-'

        # Time Helper
        def fmt_time(iso_str):
            if not iso_str: return None
            try:
                dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
                return dt.strftime("%H:%M")
            except:
                return None

        scheduled_out_raw = real_status.get('scheduled_out')
        actual_out_raw = real_status.get('actual_out')
        estimated_out_raw = real_status.get('estimated_out')
        
        scheduled_in_raw = real_status.get('scheduled_in')
        actual_in_raw = real_status.get('actual_in')
        estimated_in_raw = real_status.get('estimated_in')

        sched_dep = fmt_time(scheduled_out_raw) or "TBD"
        actual_dep = fmt_time(actual_out_raw) or fmt_time(estimated_out_raw) or "TBD"
        
        sched_arr = fmt_time(scheduled_in_raw) or "TBD"
        actual_arr = fmt_time(actual_in_raw) or fmt_time(estimated_in_raw) or "TBD"

        # Logic for "Early/On Time/Delayed" Badge
        status_text = "On Time"
        if predicted_delay_minutes > 15:
            status_text = "Delayed"
        elif predicted_delay_minutes == 0:
             status_text = "Early" if (actual_out_raw and actual_out_raw < scheduled_out_raw) else "On Time"

        return {
            "flightNumber": flight_number,
            "origin": origin,
            "destination": destination,
            "status": status_text,
            "predictedDelayMinutes": predicted_delay_minutes,
            
            # Times
            "scheduledDep": sched_dep,
            "actualDep": actual_dep,
            "scheduledArr": sched_arr,
            "actualArr": actual_arr,
            "predictedTakeoff": actual_dep, # Rename or keep specific usage

            # Gate/Terminal
            "terminalOrigin": terminal_origin,
            "gateOrigin": gate_origin,
            "terminalDest": terminal_dest,
            "gateDest": gate_dest,
            "baggageClaim": baggage,

            "delayProbability": min(99, int(predicted_delay_minutes / 2) + 10),
            "networkCongestion": int(data.x[self.airport_to_idx.get(origin, 0)][0] * 100),
            "propagationRisk": int(data.x[self.airport_to_idx.get(origin, 0)][2] * 100),
            "note": note,
            "airline": "United Airlines", # TODO: Extract carrier
            "weatherOrigin": {
                 "temp": 15, 
                 "condition": "Cloudy", # TODO: Map from data.x category
                 "windSpeed": int(data.x[self.airport_to_idx.get(origin, 0)][2] * 50),
                 "visibility": f"{int(data.x[self.airport_to_idx.get(origin, 0)][1] * 10)}mi"
            },
            "weatherDest": {
                 "temp": 10, "condition": "Rain", "windSpeed": 18, "visibility": "5km"
            },
            "incomingAircraftStatus": "In Air",
            "livePosition": {
                "lat": live_lat,
                "lon": live_lon,
                "heading": live_heading,
                "altitude": live_alt
            } if live_lat else None
        }

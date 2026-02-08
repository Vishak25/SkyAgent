
import requests
import random
from typing import Dict, Any, List, Optional

import os

class OpenSkyClient:
    """
    Client for OpenSky Network API.
    Used for: Edges (Flights) - Tracking active aircraft positions.
    Now supports OAuth2 Client Credentials Flow (required for modern accounts).
    """
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        # Map env vars: USER -> Client ID, PASSWORD -> Client Secret
        self.client_id = client_id or os.environ.get("OPENSKY_USER")
        self.client_secret = client_secret or os.environ.get("OPENSKY_PASSWORD")
        self.token = None
        self.base_url = "https://opensky-network.org/api/states/all"
        self.token_url = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
        
        # Authenticate immediately
        self._authenticate()

    def _authenticate(self):
        """
        Fetches a Bearer token using Client Credentials.
        """
        if not self.client_id or not self.client_secret:
            print("OpenSky: Missing credentials. Access will be limited/anonymous.")
            return

        try:
            payload = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            # Note: Content-Type application/x-www-form-urlencoded is default for data=...
            response = requests.post(self.token_url, data=payload, timeout=10)
            
            if response.status_code == 200:
                self.token = response.json().get('access_token')
                print("OpenSky: Authentication successful.")
            else:
                print(f"OpenSky Auth Failed {response.status_code}: {response.text}")
        except Exception as e:
            print(f"OpenSky Auth Error: {e}")

    def get_incoming_aircraft_distance(self, airport_iata: str = "ORD", airline_icao: str = "UAL") -> List[Dict]:
        """
        Retrieves live state vectors for an airline around a bounding box (ORD area).
        ORD Bounding Box (approx): lat [41.9, 42.1], lon [-88.0, -87.8]
        """
        # Extended box for better visibility
        params = {
            "lamin": 41.0000, 
            "lomin": -89.0000, 
            "lamax": 43.0000, 
            "lomax": -87.0000
        }
        
        headers = {}
        if self.token:
            headers['Authorization'] = f"Bearer {self.token}"
            
        try:
            # print(f"OpenSky Query: {self.base_url} with {params} Auth={bool(self.token)}")
            response = requests.get(self.base_url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                states = data.get("states", [])
                if not states:
                    print("OpenSky: No states found in region.")
                    return []
                
                # Filter by airline ICAO if provided (e.g. UAL, AAL)
                relevant_flights = [s for s in states if s[1] and s[1].strip().upper().startswith(airline_icao)]
                print(f"OpenSky: Found {len(relevant_flights)} flights for {airline_icao}")
                return relevant_flights 
            else:
                print(f"OpenSky Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"OpenSky API Exception: {e}")
        return []

# ...

class AeroAPIClient:
    """
    Client for FlightAware AeroAPI v4.
    Used for: Airport status and precise flight differences.
    """
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("FLIGHTAWARE_API_KEY")
        self.headers = {"x-apikey": key} if key else {}
        self.base_url = "https://aeroapi.flightaware.com/aeroapi/"

    def get_gate_congestion(self, airport_iata: str = "KORD") -> float:
        """
        Uses airport delay endpoints to infer congestion.
        """
        if not self.headers: return 0.5 # Fallback if no key
        
        url = f"{self.base_url}airports/{airport_iata}/delays"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                # Logic: Higher delay counts = higher congestion feature for your GNN
                data = response.json()
                return min(data.get('departure_delay_count', 0) / 100.0, 1.0) # Normalized
        except Exception as e:
            print(f"AeroAPI Error: {e}")
        return 0.1

    def get_flight_status(self, flight_ident: str) -> Dict[str, Any]:
        """
        Retrieves the 'canonical' status of a flight.
        """
        if not self.headers: return {}
        
        url = f"{self.base_url}flights/{flight_ident}"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # AeroAPI v4 returns {'flights': [ ... ]}
                flights = data.get('flights', [])
                if flights:
                    return flights[0]
                return data # Fallback if structure is different
            elif response.status_code == 429:
                 print("AeroAPI Limit Reached.")
                 return {"error": "API_QUOTA_EXCEEDED", "detail": "FlightAware API Quota Limit Reached."}
            return {}
            return {}
        except Exception:
            return {}

    def get_flight_position(self, flight_ident: str) -> Dict[str, Any]:
        """
        Retrieves the latest position (Lat/Lon/Alt) for a specific flight.
        Endpoint: /flights/{id}/position
        """
        if not self.headers: return {}

        url = f"{self.base_url}flights/{flight_ident}/position"
        try:
            # Request just the last 1 position for efficiency
            response = requests.get(url, headers=self.headers, params={"max_pages": 1}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Check directly for 'last_position' (some endpoints) or 'positions' list
                if 'last_position' in data:
                    return data['last_position']
                
                positions = data.get('positions', [])
                if positions:
                    return positions[0] # Newest first typically, but we should check documentation sort. 
                    # Docs say: "Returns a list of positions... ordered by timestamp descending" usually.
                    
                return {}
            return {}
        except Exception as e:
            print(f"AeroAPI Position Error: {e}")
            return {}

class CheckWXClient:
    """
    Client for CheckWX API.
    Used for: Node Features - Aviation weather (METAR).
    """
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("CHECKWX_API_KEY")
        self.headers = {"X-API-Key": key} if key else {}
        self.base_url = "https://api.checkwx.com/metar/"

    def get_metar_data(self, airport_iata: str = "KORD") -> Dict[str, Any]:
        """
        Retrieves decoded METAR data for KORD.
        """
        if not self.headers:
            print("CheckWX: No API Key found, using defaults.")
            return {
                "visibility_miles": 10,
                "wind_speed_kts": 5,
                "ceiling_ft": 20000,
                "flight_category": "VFR"
            }
            
        url = f"{self.base_url}{airport_iata}/decoded"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                json_data = response.json()
                # CheckWX returns { "data": [ ... ] }
                data_list = json_data.get("data", [])
                if not data_list:
                    print(f"CheckWX: No data found for {airport_iata}")
                    return {}
                    
                data = data_list[0]
                return {
                    "visibility_miles": data.get("visibility", {}).get("miles_float", 10.0),
                    "wind_speed_kts": data.get("wind", {}).get("speed_kts", 0),
                    "ceiling_ft": data.get("ceiling", {}).get("feet", 10000),
                    "flight_category": data.get("flight_category", "VFR")
                }
            else:
                print(f"CheckWX Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"CheckWX Exception: {e}")
        return {}

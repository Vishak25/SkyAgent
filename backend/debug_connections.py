
import os
import sys
from api_clients import OpenSkyClient, AeroAPIClient, CheckWXClient
from dotenv import load_dotenv

# Force load .env
load_dotenv()

def test_opensky():
    print("\n--- Testing OpenSky ---")
    user = os.environ.get("OPENSKY_USER")
    print(f"User loaded: {'Yes (length ' + str(len(user)) + ')' if user else 'No'}")
    
    client = OpenSkyClient()
    # Test getting incoming aircraft for ORD (Chicago)
    flights = client.get_incoming_aircraft_distance("ORD")
    
    if flights:
        print(f"✅ Success! Found {len(flights)} live aircraft.")
        print(f"Sample: {flights[0]}")
    else:
        print("⚠️  No flights found, or API call failed. (Check logs above)")

def test_checkwx():
    print("\n--- Testing CheckWX ---")
    key = os.environ.get("CHECKWX_API_KEY")
    print(f"Key loaded: {'Yes' if key else 'No'}")
    
    client = CheckWXClient()
    metar = client.get_metar_data("KORD")
    print(f"METAR Data: {metar}")
    
    if metar.get("visibility_miles") and not key:
        print("⚠️  Using default/fallback because key is missing.")
    elif metar:
         print("✅ Live data received.")
    else:
         print("❌ Request failed.")

if __name__ == "__main__":
    print(f"Testing Connectivity from: {os.getcwd()}")
    test_opensky()
    test_checkwx()

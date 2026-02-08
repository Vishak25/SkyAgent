import os
from dotenv import load_dotenv
from api_clients import AeroAPIClient
import json

load_dotenv()

def test_aero():
    print("--- Testing AeroAPI ---")
    key = os.environ.get("FLIGHTAWARE_API_KEY")
    print(f"Key loaded: {'Yes' if key else 'No'}")
    
    if not key:
        print("Skipping AeroAPI test (No Key).")
        return

    client = AeroAPIClient()
    flight_id = "UA1989"
    
    print(f"Fetching status for {flight_id}...")
    try:
        data = client.get_flight_status(flight_id)
        print(f"Raw Status Result type: {type(data)}")
        print(json.dumps(data, indent=2, default=str))
        
        if not data:
            print("❌ Result is empty/None.")
        else:
             origin = data.get('origin', {}).get('code')
             print(f"Origin Found: {origin}")
             
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_aero()

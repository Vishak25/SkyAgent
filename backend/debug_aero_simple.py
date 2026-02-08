
from api_clients import AeroAPIClient
import os

# You might need to set the API key if it's not in the environment or hardcoded in the client
# Assuming it's picked up or hardcoded in api_clients.py based on previous edits
# If it needs .env, I'll assume the environment has it or I need to find where it is.
# Checking api_clients.py, it takes optional api_key in init.
# python backend/data.py uses os.getenv typically.
# Let's check how main.py or data.py initializes it.

from data import AviationGraphHandler

handler = AviationGraphHandler()
# The handler initializes clients. 
# Let's verify how AeroAPIClient gets its key.
# In api_clients.py: headers = {"x-apikey": api_key} if api_key else {}
# So data.py must pass it or it defaults to None.
# Let's check backend/data.py again to see how it calls AeroAPIClient().

print("Testing AA3679...")
res_aa = handler.aeroapi.get_flight_status("AA3679")
print(f"Result for AA3679: {res_aa}")

print("\nTesting AAL3679...")
res_aal = handler.aeroapi.get_flight_status("AAL3679")
print(f"Result for AAL3679: {res_aal}")

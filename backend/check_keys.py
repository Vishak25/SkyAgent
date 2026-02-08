    
import os
import os

# Try loading from .env and .env.local
# load_dotenv()
# load_dotenv('.env.local')

keys = [
    "FLIGHTAWARE_API_KEY",
    "CHECKWX_API_KEY",
    "OPENSKY_USER",
    "OPENSKY_PASSWORD"
]

print("--- API Key Check ---")
for key in keys:
    value = os.environ.get(key)
    if value:
        # Mask the key for security in logs
        masked = value[:4] + "*" * (len(value) - 4) if len(value) > 4 else "****"
        print(f"✅ {key}: Found ({masked})")
    else:
        print(f"❌ {key}: MISSING")

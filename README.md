# SkyCast AI: Spatio-Temporal Flight Network Predictor ✈️

SkyCast AI is an advanced aviation intelligence platform that uses a **Graph Neural Network (ST-GNN)** to predict flight delays and provide passenger-centric insights. It models the aviation network as a dynamic graph, fusing real-time data from multiple sources to understand how congestion and weather propagate delays across the system.

## 🚀 Key Features

*   **Spatio-Temporal Graph Modeling:** Uses PyTorch Geometric to represent airports as nodes and flights as edges.
*   **Real-Time Data Fusion:** Integrates three live data streams:
    *   **OpenSky Network (OAuth2):** Tracks live aircraft movement to estimate incoming traffic load.
    *   **FlightAware AeroAPI:** Provides precise flight status, gate info, and position.
    *   **CheckWX API:** Supplies real-time METAR weather data for airport nodes.
*   **Passenger-Centric AI Insights:** Translates complex graph data into actionable advice (e.g., "Gate congestion may cause delays," "Your plane is incoming").
*   **Robust Architecture:** FastAPI backend with Python-Dotenv security + Angular frontend.

## 🏗️ Architecture

```mermaid
graph TD
    User[Passive Passenger] -->|Interacts| UI[Angular Frontend]
    UI -->|GET /predict/{flight}| API[FastAPI Backend]
    
    subgraph "Backend Intelligence"
        API --> Handler[AviationGraphHandler]
        Handler -->|Fetch| OpenSky[OpenSky Network API]
        Handler -->|Fetch| CheckWX[CheckWX Weather API]
        Handler -->|Fetch| AeroAPI[FlightAware AeroAPI]
        
        OpenSky -->|Traffic Flow| STGNN[ST-GNN Model]
        CheckWX -->|Node Features| STGNN
        AeroAPI -->|Flight State| STGNN
        
        STGNN -->|Prediction| Insight[AI Insight Engine]
    end
    
    Insight -->|JSON Response| UI
```

## 🛠️ Setup & Installation

### Prerequisites
*   Node.js & npm
*   Python 3.11+ (recommended)
*   API Keys: OpenSky (Account), FlightAware (Free Tier), CheckWX (Free Tier).

### 1. Backend Setup
1.  Create & activate the shared virtualenv (from the project root):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2.  Navigate to `backend/`:
    ```bash
    cd backend
    ```
3.  Install dependencies:
    ```bash
    # Fully pinned backend dependencies
    pip install -r requirements.lock
    ```
4.  Create `.env` file (start from `backend/.env.example`):
    ```env
    # OpenSky OAuth2 Client Credentials (client_id / client_secret)
    OPENSKY_USER=your_client_id
    OPENSKY_PASSWORD=your_client_secret
    FLIGHTAWARE_API_KEY=your_key
    CHECKWX_API_KEY=your_key
    ```
5.  Run the server:
    ```bash
    python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
    ```

### 2. Frontend Setup
1.  Navigate to root:
    ```bash
    cd ..
    ```
2.  Install packages:
    ```bash
    npm install
    ```
3.  Run the development server:
    ```bash
    npm run dev
    ```
4.  Open `http://localhost:3000`.

### 3. Install Historical Weather Dataset (Recommended)
For training/evaluation beyond live API responses, install historical METAR from Iowa Environmental Mesonet (IEM):

```bash
# From repository root
python3 backend/scripts/install_iem_metar.py \
  --days-back 30 \
  --stations KORD,KJFK,EGLL,KLAX,OMDB,RJTT,LFPG,EHAM,EDDF,WSSS \
  --output backend/data/raw/iem_metar_30d.csv
```

This creates a join-friendly CSV keyed by `station` and `valid` (UTC timestamp), saved under `backend/data/raw/`.

## ✅ Validation Status
*   **APIs:** Connected & Verified. (Note: AeroAPI Free Tier limits may trigger "Quota Exceeded").
*   **Model:** Validated with live O'Hare departures (e.g., UA1989).
*   **UI:** Validated for passenger UX responsiveness.

# SkyAgent — Agentic Aviation Delay Prediction

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.128-009688?logo=fastapi&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-0.1-1C3A57">
  <img alt="React" src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white">
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-Qwen2.5--7B-5A67D8">
</p>

> Multi-agent LangGraph system for real-time flight delay prediction, backed by a Spatio-Temporal Graph Neural Network (ST-GNN) and LLM-generated risk assessments.

**🚀 Live demo:** [huggingface.co/spaces/Vishak25/SkyAgent](https://huggingface.co/spaces/Vishak25/SkyAgent)

Look up any flight by number and get a full delay risk assessment in seconds — live flight status, weather at both airports, an ST-GNN delay prediction, a natural-language summary, and automatic rerouting suggestions when the predicted delay exceeds 30 minutes.

## Results

ST-GNN test-set performance (held-out snapshots from a 30-day METAR window):

| Metric | Value |
|--------|-------|
| **MAE** | **4.64 min** |
| Predictions within ±15 min | **90.6%** |
| Predictions within ±30 min | 97.1% |
| Model size | **1,793 parameters** |

Fewer than 2K parameters, yet 9 out of 10 delay predictions land within 15 minutes of the actual value. Full details in [`backend/model_weights/training_report.json`](backend/model_weights/training_report.json).

## How It Works

```
React Frontend  →  FastAPI Backend  →  LangGraph Agent Pipeline  →  vLLM (Qwen2.5-7B)
                                       flight_monitor → weather → delay_risk → summary / rerouting
```

Three layers:

1. **Data** — live flight status (FlightAware AeroAPI), decoded METAR weather (CheckWX), and ADS-B traffic density (OpenSky)
2. **Prediction** — a spatio-temporal GNN (GCNConv ×3 + temporal attention, PyTorch Geometric) models delay propagation across major hub airports as a dynamic graph
3. **Intelligence** — a LangGraph multi-agent pipeline orchestrates data collection, GNN inference, and LLM-generated assessments, with conditional routing: flights with ≥30 min predicted delay trigger the rerouting agent

| Agent | Responsibility |
|-------|---------------|
| FlightMonitorAgent | Live flight status and route via AeroAPI |
| WeatherAgent | METAR for origin + destination (VFR/MVFR/IFR/LIFR) |
| DelayRiskAgent | ST-GNN forward pass → predicted delay in minutes |
| ReroutingAgent | Ranks direct + 1-stop alternatives by delay risk |

All API clients share a thread-safe TTL cache, with JSON fixtures for offline development (`USE_FIXTURES=1`).

## API

| Endpoint | Description |
|----------|-------------|
| `GET /analyze/{flight}` | Full pipeline — flight + weather + GNN + LLM summary |
| `GET /predict/{flight}` | ST-GNN inference only (fastest path) |
| `GET /suggest?origin=X&destination=Y` | Rank itineraries by predicted delay |

## Quick Start

```bash
# Backend (Python 3.11+, uv)
cd backend
cp .env.example .env          # FLIGHTAWARE_API_KEY, CHECKWX_API_KEY
uv run uvicorn src.api.main:app --port 8080 --reload

# Frontend (Node 18+)
cd frontend
npm install && npm run dev    # http://localhost:3000

# Tests
cd backend && uv run pytest tests/ -x -q
```

The LLM layer works with any OpenAI-compatible endpoint — set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`. The reference deployment serves Qwen2.5-7B-Instruct via vLLM.

## License

MIT © [Vishak Nandakumar](https://github.com/Vishak25)

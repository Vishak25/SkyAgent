---
title: SkyAgent
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SkyAgent — Agentic Aviation Delay Propagation System

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.128-009688?logo=fastapi&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-0.1-1C3A57">
  <img alt="React" src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white">
  <img alt="Vite" src="https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white">
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-Qwen2.5--7B-5A67D8">
</p>

> Multi-agent LangGraph system for real-time flight delay prediction and propagation analysis, backed by a Spatio-Temporal Graph Neural Network (ST-GNN) and large language model inference on the GMU Hopper HPC cluster.

**🚀 Live demo:** [huggingface.co/spaces/Vishak25/SkyAgent](https://huggingface.co/spaces/Vishak25/SkyAgent)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Results](#2-results)
3. [System Architecture](#3-system-architecture)
4. [Backend](#4-backend)
5. [Frontend](#5-frontend)
6. [HPC Deployment (GMU Hopper)](#6-hpc-deployment-gmu-hopper)
7. [Data Sources](#7-data-sources)
8. [Development Guide](#8-development-guide)
9. [Project Phases](#9-project-phases)

---

## 1. Project Overview

SkyAgent predicts flight delays and propagates risk through the aviation network using a three-layer approach:

1. **Data layer** — real-time flight status (FlightAware AeroAPI), live METAR weather (CheckWX), and ADS-B traffic (OpenSky)
2. **Prediction layer** — a Spatio-Temporal GNN that models delay propagation across 20+ major hub airports as a dynamic graph
3. **Intelligence layer** — a LangGraph multi-agent pipeline that orchestrates data collection, GNN inference, and LLM-generated natural language assessment via Qwen2.5-7B-Instruct on GMU Hopper's A100 GPU nodes

### Key capabilities

- Look up any flight by number and get a full delay risk assessment in seconds
- Weather analysis at origin and destination (VFR/MVFR/IFR/LIFR categorization)
- ST-GNN delay prediction in minutes, trained on 30 days of METAR observations
- vLLM-powered natural language summary (Qwen2.5-7B on GMU Hopper)
- Automatic rerouting suggestions when predicted delay exceeds 30 minutes
- Step-by-step agent activity trace showing exactly what each agent did

---

## 2. Results

ST-GNN test-set performance (held-out snapshots from the 30-day IEM METAR window):

| Metric | Value |
|--------|-------|
| **MAE** | **4.64 min** |
| RMSE | 9.71 min |
| Predictions within ±15 min | **90.6%** |
| Predictions within ±30 min | 97.1% |
| Best validation MAE | 3.22 min (epoch 296/300) |
| Model size | **1,793 parameters** |

The model is deliberately tiny — fewer than 2K parameters — yet lands 9 out of 10 delay predictions within 15 minutes of the actual value. Full details in [`backend/model_weights/training_report.json`](backend/model_weights/training_report.json).

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR LAPTOP                              │
│                                                                 │
│   Browser (localhost:3000)                                      │
│       │  Vite proxy /api → :8080                               │
│       ▼                                                         │
│   React + Vite Frontend                                         │
│   ├── FlightSearch      (flight number input)                   │
│   ├── FlightHeader      (route, times, delays)                  │
│   ├── WeatherCard ×2    (origin + destination METAR)            │
│   ├── DelayRiskCard     (ST-GNN prediction + bar)               │
│   ├── AgentActivityFeed (pipeline step trace)                   │
│   ├── LLMSummary        (Qwen narrative)                        │
│   └── RouteComparison   (alternatives if delay ≥ 30min)        │
│                                                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP GET /analyze/{flight}
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              FastAPI Backend (localhost:8080)                   │
│                                                                 │
│   GET /analyze/{flight}  →  LangGraph Pipeline                 │
│   GET /predict/{flight}  →  ST-GNN only (no LLM)              │
│   GET /suggest           →  Route ranking by delay risk        │
│                                                                 │
│   LangGraph StateGraph:                                         │
│   flight_monitor → weather → delay_risk → [conditional]        │
│                                    ↓ delay < 30min             │
│                                 summary → END                  │
│                                    ↓ delay ≥ 30min             │
│                                 rerouting → END                │
│                                                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │ OpenAI-compatible API
                            │ via SSH tunnel localhost:8000
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 GMU Hopper HPC — GPU Node                       │
│                                                                 │
│   vLLM Server (Singularity container)                           │
│   Model:  Qwen/Qwen2.5-7B-Instruct                             │
│   GPU:    MIG 3g.40gb slice (A100)                             │
│   Port:   8000  │  Auth: Bearer skyagent-dev                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Backend

**Location:** `backend/` | **Runtime:** Python 3.11 via `uv`

### Directory layout

```
backend/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py       # LangGraph StateGraph (entry point)
│   │   ├── flight_monitor.py     # AeroAPI flight status node
│   │   ├── weather_agent.py      # CheckWX METAR/TAF node
│   │   ├── delay_risk_agent.py   # ST-GNN inference node
│   │   └── rerouting_agent.py    # Alternative route node
│   ├── tools/
│   │   ├── _cache.py             # Thread-safe TTLCache (5-min TTL)
│   │   ├── _fixtures.py          # JSON fixture loader (offline dev)
│   │   ├── flight_tools.py       # AeroAPIClient
│   │   ├── weather_tools.py      # CheckWXClient
│   │   └── opensky_tools.py      # OpenSkyClient
│   ├── models/
│   │   └── delay_gnn/
│   │       ├── model.py          # STGNN (PyTorch Geometric)
│   │       ├── graph_builder.py  # AviationGraphHandler
│   │       ├── dataset.py        # METARGraphDataset
│   │       └── train.py          # Training script
│   ├── api/
│   │   ├── main.py               # FastAPI app, startup model load
│   │   ├── routes.py             # Route definitions
│   │   └── websocket.py          # WebSocket manager (Phase 4)
│   └── config/
│       └── settings.py           # All env vars, airport data, helpers
├── slurm/
│   ├── vllm_qwen9b.slurm         # MIG 3g.40gb, Qwen2.5-7B-Instruct
│   ├── vllm_mistral.slurm        # 4× A100, Mistral Small 4 (demo)
│   ├── agent_service.slurm       # CPU node FastAPI service
│   └── gnn_training.slurm        # A100 GNN training job
├── fixtures/                     # AeroAPI, CheckWX, OpenSky JSON fixtures
├── data/raw/                     # IEM METAR CSV (30-day window)
├── model_weights/                # stgnn_best.pt + training_report.json
├── tests/                        # pytest suite (52 pass, 1 skip)
└── requirements.txt
```

### API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Root info, lists all endpoints |
| `GET /health` | Health check |
| `GET /predict/{flight}` | ST-GNN inference only — no LLM, fastest path |
| `GET /analyze/{flight}` | Full LangGraph pipeline — flight + weather + GNN + LLM |
| `GET /suggest?origin=X&destination=Y` | Rank direct + 1-stop itineraries by predicted delay |

### Agent system (LangGraph)

**Orchestrator** (`src/agents/orchestrator.py`) builds a `StateGraph` with shared state: `flight_number`, `origin`, `destination`, `flight_status`, `weather_origin`, `weather_destination`, `predicted_delay`, `alternative_routes`, `llm_summary`, `agent_log`.

| Agent | File | Responsibility |
|-------|------|---------------|
| FlightMonitorAgent | `flight_monitor.py` | Fetches live flight status and position via AeroAPI; extracts origin/destination IATA |
| WeatherAgent | `weather_agent.py` | Fetches decoded METAR for origin + destination via CheckWX; IATA→ICAO conversion |
| DelayRiskAgent | `delay_risk_agent.py` | Builds airport graph, runs ST-GNN forward pass, returns predicted delay in minutes |
| ReroutingAgent | `rerouting_agent.py` | Enumerates direct + 1-stop alternatives scored by ST-GNN; uses LLM for recommendation. Only runs when `predicted_delay >= 30` |

### ST-GNN model

3-layer Graph Convolutional Network with temporal attention (`src/models/delay_gnn/model.py`):

- **Input:** 5 features per airport node — visibility, wind, ceiling, precipitation severity, flight category ordinal
- **Architecture:** GCNConv(5→32) → GCNConv(32→32) → GCNConv(32→32) + temporal attention head → Linear(32→1)
- **Output:** log-space delay minutes, inverted at inference via `expm1()`
- **Training:** MSE loss, Adam optimizer on 720 hourly IEM METAR graph snapshots (30-day window across 10 international hubs)
- **Weights:** `model_weights/stgnn_best.pt` (loaded at FastAPI startup)

### Tool clients

| Client | Source | Key methods |
|--------|--------|-------------|
| `AeroAPIClient` | FlightAware AeroAPI v4 | `get_flight_status()`, `get_flight_position()`, `get_airport_info()` |
| `CheckWXClient` | CheckWX | `get_metar_data()` → decoded dict with VFR/MVFR/IFR/LIFR, `get_taf_data()` |
| `OpenSkyClient` | OpenSky Network | `get_traffic_density()` (ADS-B congestion), `get_flight_position()` |

All clients use a thread-safe `TTLCache` (5-min TTL). Set `USE_FIXTURES=1` to serve from local JSON files instead of live APIs.

### Configuration (`src/config/settings.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `FLIGHTAWARE_API_KEY` | — | FlightAware AeroAPI v4 key |
| `CHECKWX_API_KEY` | — | CheckWX API key |
| `LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint (vLLM, Gemini, etc.) |
| `LLM_API_KEY` | `skyagent-dev` | Bearer token for the endpoint |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Model identifier |
| `MODEL_WEIGHTS_PATH` | — | Path to `stgnn_best.pt` |
| `USE_FIXTURES` | `0` | Enable offline fixture mode |
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |

---

## 5. Frontend

**Location:** `frontend/` | **Stack:** React 18 + TypeScript + Vite  
**Dev server:** `npm run dev` → `http://localhost:3000`  
**API proxy:** Vite proxies `/api/*` → `http://localhost:8080`

### Components

| Component | Description |
|-----------|-------------|
| `FlightSearch` | Flight number input with spinner; proxies to `/api/analyze/{flight}` |
| `FlightHeader` | Route (IATA codes, cities, times in local timezone), aircraft badge, status badge, departure/arrival delay pills |
| `WeatherCard` | VFR/MVFR/IFR/LIFR badge, visibility, wind, ceiling, temperature — rendered for both origin and destination |
| `DelayRiskCard` | Predicted delay (minutes) with color-coded risk label (Low/Moderate/High/Very High) and progress bar |
| `AgentActivityFeed` | Ordered step trace from `agent_log` with icons per agent type |
| `LLMSummary` | Qwen2.5-7B narrative with "GMU Hopper" badge; only renders when `llm_summary` is non-null |
| `RouteComparison` | Up to 5 alternative itineraries with delay scores; only renders when `alternative_routes` is non-empty (delay ≥ 30 min) |

---

## 6. HPC Deployment (GMU Hopper)

### SLURM scripts

| Script | Partition | Resource | Purpose |
|--------|-----------|----------|---------|
| `vllm_qwen9b.slurm` | gpuq | MIG 3g.40gb (A100) | Serve Qwen2.5-7B-Instruct, dev |
| `vllm_mistral.slurm` | gpuq | 4× A100 80GB | Serve Mistral Small 4 119B, demo |
| `agent_service.slurm` | normal | 8 CPU, 32GB | Run FastAPI backend on CPU node |
| `gnn_training.slurm` | gpuq | A100 | Train ST-GNN, output stgnn_best.pt |

### Deployment workflow

```bash
# 1. Submit vLLM job on Hopper
cd /scratch/$USER/skyagent && sbatch slurm/vllm_qwen9b.slurm

# 2. Note the GPU node from the job log
squeue -u $USER   # e.g. gpu020

# 3. Open SSH tunnel from laptop
ssh -L 8000:gpu020.orc.gmu.edu:8000 vnandak@hopper.orc.gmu.edu

# 4. Start backend on laptop
cd backend && uv run uvicorn src.api.main:app --port 8080 --reload

# 5. Start frontend on laptop
cd frontend && npm run dev

# 6. Open http://localhost:3000
```

---

## 7. Data Sources

| Source | Used for |
|--------|---------|
| FlightAware AeroAPI v4 | Real-time flight status, position, schedules, airport info |
| CheckWX | Decoded METAR observations and TAF forecasts |
| OpenSky Network | Live ADS-B traffic density (congestion features) |
| IEM METAR Archive | Historical METAR for GNN training |

---

## 8. Development Guide

### Prerequisites

- Python 3.11+ with `uv` (`pip install uv`)
- Node.js 18+ with `npm`
- API keys: `FLIGHTAWARE_API_KEY`, `CHECKWX_API_KEY`

### Backend

```bash
cd backend
cp .env.example .env        # fill in API keys

uv run uvicorn src.api.main:app --port 8080 --reload   # live mode
USE_FIXTURES=1 uv run uvicorn src.api.main:app --port 8080 --reload  # offline

uv run pytest tests/ -x -q  # 52 pass, 1 skip

uv run python -m src.models.delay_gnn.train \
  --data data/raw/iem_metar_30d.csv --epochs 200
```

### Frontend

```bash
cd frontend
npm install
npm run dev      # :3000 with API proxy
npm run build    # production build
```

---

## 9. Project Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Hopper + vLLM | **Complete** | pyenv + venv on Hopper, Qwen weights downloaded, Singularity image, vLLM serving via SLURM |
| 2 — Agent Pipeline | **Complete** | LangGraph StateGraph, all 4 agents, `/analyze` endpoint, end-to-end verified |
| 3 — Frontend | **Complete** | React + Vite UI, 7 components, Vite proxy |
| 4 — WebSocket Streaming | Pending | Real-time agent updates pushed to browser (`websocket.py` skeleton exists) |
| 5 — Cloud Deployment | Skipped | Deferred |

---

## License

MIT © Vishak Nandakumar

## Author

**Vishak Nandakumar** — [@Vishak25](https://github.com/Vishak25)
Built for the GMU Hopper HPC cluster as part of research into agentic systems over real-time aviation data.

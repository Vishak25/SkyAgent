# SkyCast AI — Project Progress Log

> This document tracks the evolution of SkyCast AI from initial MVP to its current state.
> It is updated every time a significant change is made.

---

## Table of Contents

1. [Phase 0: Initial MVP](#phase-0-initial-mvp)
2. [Phase 1: Expert Review & P0/P1 Hardening](#phase-1-expert-review--p0p1-hardening)
3. [Phase 2: Feature Expansion (Winter Ops, TAF, Route Suggestions)](#phase-2-feature-expansion)
4. [Phase 3: Model Training on IEM METAR Data](#phase-3-model-training-on-iem-metar-data)
5. [Phase 4: Flight Selection & Observed Delay Fix](#phase-4-flight-selection--observed-delay-fix)
6. [Current Architecture](#current-architecture)
7. [What's Next](#whats-next)
7. [File Change Log](#file-change-log)

---

## Phase 0: Initial MVP

**Date:** 2026-02-08  
**Commit:** `1df8d84` — "MVP"

### What existed

- **Backend (FastAPI + Python)**
  - `backend/main.py` — FastAPI server with a single `GET /predict/{flight_number}` endpoint.
  - `backend/model.py` — 4-feature ST-GNN (GCNConv, 16 hidden channels). No trained weights; random initialization only.
  - `backend/data.py` — `AviationGraphHandler` that hard-coded origin/destination to `ORD → LHR`, built the graph, ran inference, *then* fetched the real flight's route from AeroAPI and overwrote the response fields (so the prediction didn't actually match the flight).
  - `backend/api_clients.py` — Three API clients (OpenSky, AeroAPI, CheckWX) with no caching, no retries, and no consistent IATA/ICAO normalization.
  - `backend/requirements.txt` — Unpinned deps; `python-dotenv` was missing even though it was imported.
  - Debug scripts: `check_keys.py`, `debug_aero.py`, `debug_aero_simple.py`, `debug_connections.py`.

- **Frontend (Angular 21 + Tailwind)**
  - Single-page app with a flight-number search bar.
  - Components: `flight-leg-card`, `network-graph` (D3 force layout), `stat-card`, `weather-widget`.
  - `FlightPredictorService` with `getFlightStatus()`, `generateDummyScenario()`, and `analyzeScenario()`.
  - Hardcoded `http://localhost:8000` backend URL.

- **Repo hygiene issues**
  - `credentials.json` committed to the repo (potential secret leak).
  - `.angular/` build cache (19 large files) tracked in git.
  - No `.env.example`, no `.gitignore` for Python caches.
  - CORS set to `allow_origins=["*"]` with `allow_credentials=True`.

### Known problems at this stage

| Area | Issue |
|------|-------|
| ML correctness | Model ran inference on a hard-coded ORD→LHR graph, not the actual flight's route |
| ML validity | No training data, no trained weights — predictions were random noise × 100 |
| API efficiency | ~20 external API calls per request (10 airports × congestion + weather), no caching |
| Security | `credentials.json` in repo, CORS wildcard, no `.env.example` |
| Reproducibility | Missing `python-dotenv` in requirements, unpinned versions |
| Weather features | Only 4 features (congestion, visibility, wind, flight category) — no precipitation/snow |
| UX | Reactive only — user must already know their flight number |

---

## Phase 1: Expert Review & P0/P1 Hardening

**Date:** 2026-02-08  
**Commit:** `cb894bf` — "bug fixes" + uncommitted P0/P1 changes

### What was done

#### P0: Repo Safety & Reproducibility

- **Removed `credentials.json`** from the repo and added it to `.gitignore`.
- **Purged `.angular/` build cache** (19 files, ~5MB) from git tracking; added `.angular/` to `.gitignore`.
- **Added `.env.example`** at repo root and `backend/.env.example` with all required env vars documented.
- **Added Python ignores** to `.gitignore`: `__pycache__/`, `*.py[cod]`, `.venv/`, `.pytest_cache/`, etc.
- **Fixed `backend/requirements.txt`**: added missing `python-dotenv==1.0.1`, pinned all direct deps.
- **Created `backend/requirements.lock`**: fully pinned transitive dependency lockfile (`pip freeze`).
- **Updated README**: fixed non-portable absolute image path, clarified OpenSky creds are OAuth2, added lockfile install option.
- **Made CORS configurable** via `CORS_ALLOW_ORIGINS` env var (no longer hardcoded `*`).

#### P1: Inference Correctness

- **Fixed prediction pipeline order**: now fetches live flight metadata (origin/destination) from AeroAPI **first**, then builds the graph for that actual route, then runs inference. Previously the graph was built for ORD→LHR regardless of the flight.
- **Added IATA/ICAO normalization**: consistent code resolution using `_resolve_icao()` with a static lookup table + AeroAPI airport info fallback with 7-day TTL cache.
- **Added aggressive TTL caching** to all API clients:
  - OpenSky states: 15s cache
  - AeroAPI congestion: 2 min cache
  - AeroAPI flight status: 15s cache
  - AeroAPI flight position: 10s cache
  - AeroAPI airport info: 7-day cache
  - CheckWX METAR: 2 min cache
- **Fixed duplicate `return` statement** in `AeroAPIClient.get_flight_status()`.
- **Made error responses use proper HTTP status codes** (429 for quota, 404 for not found, 503 for key missing) instead of always returning 200 with an error body.
- **Frontend error handling**: now parses JSON from non-2xx responses so users see useful messages.
- **Heuristic fields explicitly labeled**: response `note` field clarifies which fields are heuristics vs model output.

#### P1: Dataset Research (for future P2 training)

Identified and documented six datasets for training a real delay model:

| Dataset | What it provides | Source |
|---------|-----------------|--------|
| BTS On-Time Performance | Flight-level delay labels + causes | transtats.bts.gov |
| NOAA ISD / Global Hourly | Historical hourly surface weather | ncei.noaa.gov |
| Iowa Environmental Mesonet | Historical METAR archive | mesonet.agron.iastate.edu |
| FAA OPSNET | Ground delay programs, system constraints | aspm.faa.gov |
| OurAirports | Airport metadata (IATA/ICAO, lat/lon) | ourairports.com |
| OpenSky Historical | ADS-B traffic state vectors | opensky-network.org |

---

## Phase 2: Feature Expansion

**Date:** 2026-02-08  
**Scope:** +1,120 lines added, −423 lines removed across 9 files

### Feature 2a: Winter Operations / Precipitation Awareness

**Problem:** The model had no awareness of snow, ice, freezing rain, or de-icing — the #1 delay driver at winter hubs like ORD, JFK, DEN.

**What was implemented:**

- **`api_clients.py`** — Added a comprehensive precipitation severity map (`_PRECIP_SEVERITY`) covering 35+ METAR weather codes: rain (RA), snow (SN), freezing rain (FZRA), ice pellets (PL), thunderstorms (TS), fog (FG), blowing snow (BLSN), etc.
- **`CheckWXClient.get_metar_data()`** — Now extracts the `conditions` array from decoded METAR and returns two new fields:
  - `precip_severity` (float 0..1): normalized severity score
  - `precip_label` (string): human-readable condition (e.g., "SN", "FZRA", "TS")
- **`model.py`** — ST-GNN upgraded from **4 → 5 input features** and hidden dim from 16 → 32. The 5th feature encodes precipitation severity.
- **`data.py`** — Node feature vector is now `[congestion, vis_norm, wind_norm, flight_category, precip_severity]`. Risk heuristics weight precipitation at 25%.
- **Frontend** — Track mode shows a "Precip Impact" stat card and a "Winter Ops" progress bar. AI insight engine warns about de-icing and runway cleaning when severity exceeds thresholds.

### Feature 2b: TAF (Weather Forecast) Integration

**Problem:** Current METAR only tells you what the weather *is now*. To predict delays for a flight departing in 6 hours, you need a *forecast*.

**What was implemented:**

- **`api_clients.py`** — Added `CheckWXClient.get_taf_data()`:
  - Fetches decoded TAF (Terminal Aerodrome Forecast) from CheckWX API
  - Aggregates **worst-case** visibility, wind, flight category, and precipitation across all forecast periods
  - Returns `forecast_visibility_miles`, `forecast_wind_speed_kts`, `forecast_flight_category`, `forecast_precip_severity`, `forecast_precip_label`
  - 10-minute TTL cache
- **`data.py`** — `build_graph()` accepts `use_taf=True` parameter. When set, node features are derived from TAF instead of METAR, enabling forward-looking prediction.

### Feature 2c: Multi-Itinerary Route Suggestion

**Problem:** The system was purely *reactive* — you type a flight number you're already booked on. The much higher value is: "Given ORD → LHR, which route should I book/rebook to minimize delay risk?"

**What was implemented:**

- **`api_clients.py`** — Added `AeroAPIClient.get_scheduled_flights()` to look up scheduled flights between two airports (with optional date filter, 5-min TTL cache).
- **`data.py`** — New `suggest_routes()` method:
  1. Builds graph using **TAF** (forecast) features
  2. Enumerates **direct flights** from AeroAPI schedule data
  3. Enumerates **1-stop connections** via up to 6 major hubs (ORD, JFK, ATL, DFW, DEN, LAX, SFO, LHR, FRA, AMS)
  4. Scores each itinerary with the GNN (per-node delay at destination/hub)
  5. Ranks by predicted total delay (lower = better)
  6. Returns weather + forecast for origin/destination + ranked itinerary list
- **`main.py`** — New `GET /suggest?origin=ORD&destination=LHR&date=2026-02-10` endpoint.
- **Frontend** — New **"Find Best Route"** mode:
  - Mode toggle in header: "Track Flight" vs "Find Best Route"
  - Route search bar with origin, destination, and optional date fields
  - Results view:
    - Origin/destination header with airport names
    - Current weather + TAF forecast cards for both airports
    - AI route analysis (rule-based: warns about winter weather, compares direct vs connection savings)
    - Ranked itinerary cards showing: route type (direct/1-stop), delay prediction, risk bars, "Recommended" badge
  - New `ItineraryCardComponent` with delay color coding, risk progress bars, and connection hub display
  - `FlightPredictorService` — Added `suggestRoutes()` API call and `analyzeRoutes()` rule-based insight generator

### Files changed in Phase 2

| File | Change |
|------|--------|
| `backend/api_clients.py` | +precipitation extraction, +TAF client, +scheduled flights lookup, +caching |
| `backend/data.py` | +5-dim features, +`suggest_routes()`, +TAF graph mode, +risk heuristics with precip |
| `backend/main.py` | +`/suggest` endpoint, model 4→5 dims, hidden 16→32 |
| `backend/model.py` | 5-dim input, 32 hidden channels |
| `src/services/flight-predictor.service.ts` | +`RouteSuggestion` types, +`suggestRoutes()`, +`analyzeRoutes()`, +precip in `analyzeScenario()` |
| `src/app.component.ts` | +search mode toggle, +suggest mode state, +`suggestRoutes()` handler |
| `src/app.component.html` | +mode toggle UI, +suggest mode results panel, +precip stats in track mode |
| `src/components/itinerary-card.component.ts` | **New file** — itinerary comparison card |
| `.gitignore` | +`.venv/` |
| `README.md` | +historical weather install instructions |

---

## Phase 3: Model Training on IEM METAR Data

**Date:** 2026-02-08  
**Scope:** New training pipeline + trained model weights  
**Dataset:** `iem_metar_30d.csv` — 37,047 METAR observations, 10 major airports, 30 days (Jan 9 – Feb 8, 2026)

### The problem

The ST-GNN had **no trained weights** — predictions were random noise × a scale factor. The model needed to learn the relationship between weather features and flight delays.

### The challenge

We have weather data (METAR) but **no actual flight delay labels**. Supervised training requires delay targets.

### The approach: physics-informed proxy labels

Instead of using real delay data (which would require BTS On-Time Performance data), we generate **synthetic delay labels** from well-documented FAA weather-to-delay correlations:

| Weather condition | Delay contribution | Source |
|---|---|---|
| Visibility < 1 mi (LIFR) | +45 min | FAA OPSNET: ~50% arrival rate reduction |
| Visibility 1–3 mi (IFR) | +20 min | FAA OPSNET: ~30% arrival rate reduction |
| Heavy snow (+SN) | +55 min | FAA AC 150/5200-30D: de-icing + runway treatment |
| Moderate snow (SN) | +35 min | FAA AC 150/5200-30D |
| Freezing rain (FZRA) | +55 min | Highest ops impact (holdover time) |
| Thunderstorm (TS) | +40 min | Ground stop typical duration |
| Fog (FG) | +30 min | Major visibility impact on ILS approaches |
| Wind > 35 kt | +25 min | Possible runway closure, crosswind limits |
| Snow depth | +3 min/inch | Runway contamination clearing |

Additional label refinements:
- **Spatial propagation**: 20% of each airport's delay comes from network neighbors (weighted by inverse distance), simulating how delay cascades through connecting traffic
- **Temporal persistence**: 15% blended from the previous hour (delays are "sticky" in real operations)
- **Time-of-day modulation**: peak hours (12–23 UTC) get a 1.25× multiplier; off-peak gets 0.80×
- **Heteroscedastic noise**: σ = max(2.0, delay × 0.15) prevents the model from memorizing a lookup table

### Training pipeline

**`backend/training/dataset.py`** — METAR CSV → PyG graph snapshots:
- Parses 37,047 rows → 720 hourly graph snapshots
- Each snapshot: 10-node graph with 44 sparse route-based edges (not fully connected — prevents GCN over-smoothing)
- Node features: `[congestion_proxy, vis_norm, wind_norm, cat_ordinal, precip_severity]` (same 5-dim as inference)
- Flight category derived from ceiling + visibility per FAA AIM 7-1-7
- Temporal split: 503 train / 109 val / 108 test (70/15/15 by time, no data leakage)

**`backend/training/train.py`** — Training loop:
- **Loss**: Weighted Huber loss in log-space — `log(1 + delay)` targets to handle right-skewed distribution; sample weights = `√(1 + target)` to upweight delayed observations
- **Optimizer**: AdamW (lr=0.003, weight_decay=1e-4)
- **Scheduler**: Cosine annealing with warm restarts (T₀=30, T_mult=2)
- **Regularization**: Dropout 0.2 (in model), gradient clipping at 5.0
- **Early stopping**: patience=50 on validation MAE

### Graph structure: why sparse edges matter

Initial attempts with a fully-connected graph produced identical predictions for all airports (the GCN over-smoothed in a 10-node graph after 2 hops, every node saw every other node's features). The fix: route-based sparse edges representing real airline connections:
- US domestic: ORD↔JFK, ORD↔LAX, JFK↔LAX
- Transatlantic: JFK↔LHR, JFK↔CDG, ORD↔LHR, ORD↔FRA
- European intra: LHR↔FRA, LHR↔AMS, FRA↔CDG, etc.
- Asia/ME: DXB↔LHR, LAX↔HND, HND↔SIN, etc.

This gives each node a distinct 2-hop neighborhood, allowing the GCN to learn differentiated representations.

### Training iterations and results

| Run | Key change | R² | Test MAE | Delay recall |
|-----|-----------|-----|----------|-------------|
| v1 | Vanilla MSE, full-connect graph | −0.15 | 6.80 min | 0% (mode collapse) |
| v2 | Log-space targets + weighted Huber | −0.03 | 5.31 min | 0% (over-smoothing) |
| v3 | Sparse route-based edges | 0.32 | 4.87 min | 47.5% |
| **v4 (final)** | **300 epochs, cosine anneal** | **0.38** | **4.64 min** | **44.9%** |

### Final model metrics (test set: Feb 5–8, 2026)

| Metric | Value |
|--------|-------|
| MAE | 4.64 min |
| RMSE | 9.71 min |
| R² | 0.379 |
| Within ±15 min | 90.6% |
| Within ±30 min | 97.1% |
| Delay (>15 min) recall | 44.9% |
| Delay (>15 min) precision | 54.6% |
| Val MAE (best) | 3.22 min (epoch 296) |
| Training time | 9.8 seconds (CPU) |

### Qualitative validation (worst-weather test snapshot)

| Airport | Weather | Predicted | Actual |
|---------|---------|-----------|--------|
| FRA | Fog + snow (LIFR, precip=0.55) | **27.4 min** | 82.9 min |
| LHR | Low vis (LIFR) | **4.5 min** | 14.2 min |
| ORD | Clear (VFR) | **2.2 min** | 2.5 min |
| JFK | Clear (VFR) | **2.9 min** | 2.4 min |

The model correctly identifies Frankfurt as highest-delay (fog + snow) and ranks clear-weather airports near zero. It compresses the upper range (27 vs 83 for FRA) — expected with log-space training and only 1,793 parameters.

### Inference integration

- **`data.py`** — `_run_model()` updated: if `model._log_space == True` (set when trained weights are loaded), applies `expm1()` to convert log-space output back to minutes. Falls back to legacy `STGNN_OUTPUT_SCALE` for untrained/random weights.
- **`main.py`** — `load_model()` now auto-discovers `backend/model_weights/stgnn_best.pt` if `MODEL_WEIGHTS_PATH` env var is not set. Sets `model._log_space = True` when trained weights are loaded.

### Files added/changed in Phase 3

| File | Change |
|------|--------|
| `backend/training/__init__.py` | **New** — Training package |
| `backend/training/dataset.py` | **New** — METAR CSV parser, proxy label generator, graph snapshot builder |
| `backend/training/train.py` | **New** — Training loop, evaluation, weights saving |
| `backend/model_weights/stgnn_best.pt` | **New** — Trained model weights (1,793 params) |
| `backend/model_weights/training_report.json` | **New** — Full training metrics report |
| `backend/data/raw/iem_metar_30d.csv` | **New** — 37K-row IEM METAR dataset |
| `backend/data.py` | Updated `_run_model()` for log-space output; updated `suggest_routes()` |
| `backend/main.py` | Auto-discover trained weights, set `_log_space` flag |
| `backend/requirements.txt` | Added `numpy>=1.26.0` |

---

## Current Architecture

```
User
 ├─ [Track Mode] ──> GET /predict/{flight}
 │    └─ AeroAPI (flight status) → resolve origin/dest
 │    └─ Build graph (METAR features, 5-dim) → TRAINED ST-GNN → delay prediction
 │    └─ AeroAPI (position) → live lat/lon
 │    └─ Return: delay, risk, weather, gate/terminal, live position
 │
 └─ [Suggest Mode] ──> GET /suggest?origin=X&destination=Y&date=Z
      └─ AeroAPI (scheduled flights) → enumerate direct options
      └─ Build graph (TAF forecast features, 5-dim) → TRAINED ST-GNN
      └─ Score direct flights + 1-stop connections via 6 hubs
      └─ Rank by predicted delay → return itinerary list

Training Pipeline:
  IEM METAR CSV (37K rows, 30 days, 10 airports)
    → Hourly graph snapshots (720 graphs, 44 sparse edges each)
    → Physics-informed delay labels (FAA-correlated weather rules)
    → Log-space weighted Huber loss + AdamW + cosine anneal
    → Best weights saved to model_weights/stgnn_best.pt
```

**Model:** ST-GNN (2-hop GCN, 32 hidden, MLP readout) — 1,793 trainable parameters  
**Node features:** `[congestion, visibility, wind, flight_category, precip_severity]`  
**Output:** `log(1 + delay_minutes)` → converted via `expm1()` at inference  
**Graph (training):** 10 airports, 44 route-based edges  
**Graph (inference):** 20+ base airports + dynamic origin/dest; edges = traffic proxy into origin + route edge

---

## Phase 4: Flight Selection & Observed Delay Fix

> **Date:** 2026-02-08  
> **Trigger:** Live testing revealed SkyCast showed AA3162 as "On Time" with 2 min delay, while the actual flight was 67+ minutes delayed and en route.

### Root Causes Found

1. **Wrong flight selected from AeroAPI:** The `/flights/{ident}` endpoint returns ~15 flights (past, present, and future). Our code blindly took `flights[0]` (tomorrow's scheduled flight) instead of the one actually in the air today.

2. **Status badge only checked model prediction:** The code set status = "Delayed" only when `predicted_delay_minutes > 15`, ignoring AeroAPI's own delay data. The GNN model predicted 2 min (correct for weather — it was clear at both JFK and ORD), but the real delay was operational (crew, mechanical, ATC).

3. **No observed delay calculation:** We had `scheduled_out`, `actual_out`, and `departure_delay` from AeroAPI but never used them.

### Fixes Implemented

| File | Change |
|------|--------|
| `backend/api_clients.py` | New `_pick_best_flight()` function: prioritises en-route flights, then most-recently-arrived, then next scheduled. Prevents selecting tomorrow's flight over today's active one. |
| `backend/data.py` | Computes `observed_delay` from `actual_out − scheduled_out` AND AeroAPI's `departure_delay` / `arrival_delay` fields. `final_delay = max(observed, model_predicted)`. Status badge now uses both observed delay and AeroAPI's status string. New statuses: "Severely Delayed", "Slight Delay", "- En Route" suffix. |
| `src/services/flight-predictor.service.ts` | Status type widened to `string` to handle compound statuses. `analyzeScenario()` reports observed delay separately from model prediction. New fields: `observedDelayMinutes`, `modelPredictedDelay`. |
| `src/components/flight-leg-card.component.ts` | Status badge uses `computed()` with `includes()` for flexible matching: rose for severe/cancelled, amber for delayed, emerald for on time. |

### Result (AA3162)

| Field | Before | After | Real |
|-------|--------|-------|------|
| Status | On Time | **Severely Delayed - En Route** | Delayed - EnRoute |
| Delay | 2 min | **67 min** | ~67-96 min |
| Actual Dep | 20:44 (=sched) | **21:51** | 21:51 UTC |
| Gate (dest) | K12 | **K5** | K5 |

---

## What's Next

| Priority | Item | Status |
|----------|------|--------|
| P2 | Incorporate BTS On-Time Performance data (real delay labels) | Not started |
| P2 | Retrain with real labels + METAR features (supervised) | Not started |
| P2 | Add NOTAM / FICON data for runway contamination | Not started |
| P2 | Add de-icing holdover time estimation | Not started |
| P2 | Align training/inference graph structure (same edge pattern) | Not started |
| P3 | Replace heuristic delay probability with calibrated model output | Blocked on real labels |
| P3 | Add WebSocket for real-time flight tracking updates | Not started |
| P3 | Deploy (Docker, CI/CD, production CORS) | Not started |
| P3 | Add unit/integration tests | Not started |

---

## File Change Log

| Date | Files | Summary |
|------|-------|---------|
| 2026-02-08 | Initial commit (all files) | MVP: 4-feature GNN, 3 API clients, Angular dashboard |
| 2026-02-08 | `.gitignore`, `credentials.json`, `.angular/`, `.env.example`, `requirements.txt`, `requirements.lock`, `README.md`, `main.py`, `api_clients.py`, `data.py`, `flight-predictor.service.ts` | P0/P1: repo hygiene, inference correctness, caching, error handling |
| 2026-02-08 | `api_clients.py`, `data.py`, `main.py`, `model.py`, `app.component.ts`, `app.component.html`, `flight-predictor.service.ts`, `itinerary-card.component.ts` | Phase 2: winter ops (precip), TAF forecasts, route suggestion mode |
| 2026-02-08 | `training/dataset.py`, `training/train.py`, `model_weights/stgnn_best.pt`, `data.py`, `main.py`, `requirements.txt` | Phase 3: training pipeline, trained model (R²=0.38, MAE=4.64 min), log-space inference |
| 2026-02-08 | `api_clients.py`, `data.py`, `flight-predictor.service.ts`, `flight-leg-card.component.ts` | Phase 4: Fix flight selection (pick en-route over next-scheduled), compute observed delay from AeroAPI, status badge uses real delay data |
| 2026-02-09 | `api_clients.py`, `data.py`, `flight-leg-card.component.ts` | Phase 4b: Local airport timezone display (departure in origin TZ, arrival in dest TZ), IATA codes instead of ICAO, improved flight selection scoring with recency, AirportInfo.timezone |
| 2026-02-09 | `api_clients.py`, `data.py`, `app.component.html`, `flight-leg-card.component.ts`, `flight-predictor.service.ts` | Phase 4c: Fixed stale-leg selection (prefer active/pending flight over old arrivals), corrected delay severity threshold, and added Actual vs Est time labeling for UI accuracy |

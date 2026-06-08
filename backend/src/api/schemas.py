"""
Canonical Pydantic response models for SkyAgent API.

Both /predict and /analyze return FlightAnalysis (same camelCase shape).
/analyze adds llmSummary, alternativeRoutes, and agentLog on top.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class WeatherWidget(BaseModel):
    flightCategory: str = "VFR"
    temp: int = 0
    condition: str = "Unknown"
    windSpeed: int = 0
    visibility: str = "10 mi"
    precipSeverity: int = 0
    precipLabel: str = "None"


class LivePosition(BaseModel):
    lat: float
    lon: float
    heading: Optional[float] = None
    altitude: Optional[float] = None


class GraphNode(BaseModel):
    id: str
    role: str
    risk: float
    predictedDelay: float
    congestion: float
    precipSeverity: float
    condition: str
    wind: float
    visibility: float
    lat: Optional[float] = None
    lon: Optional[float] = None


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str


class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class ItineraryLeg(BaseModel):
    origin: str
    destination: str
    scheduledDep: str
    scheduledArr: str
    flightNumber: str


class Itinerary(BaseModel):
    type: str
    flightNumber: str
    airline: str
    legs: List[ItineraryLeg]
    predictedDelayMinutes: int
    delayRisk: str
    propagationRisk: int
    precipSeverity: int
    stops: int
    rank: Optional[int] = None
    recommended: Optional[bool] = None
    connectionHub: Optional[str] = None
    connectionHubName: Optional[str] = None


class FlightAnalysis(BaseModel):
    # Identity
    flightNumber: str
    origin: str
    destination: str
    originIcao: Optional[str] = None
    destinationIcao: Optional[str] = None
    status: str
    airline: str

    # Timing (pre-formatted strings)
    scheduledDep: str
    actualDep: str
    depTimeKind: str
    predictedTakeoff: str
    scheduledArr: str
    actualArr: str
    arrTimeKind: str

    # Gate/terminal
    terminalOrigin: str = "-"
    gateOrigin: str = "-"
    terminalDest: str = "-"
    gateDest: str = "-"
    baggageClaim: str = "-"

    # Delay
    predictedDelayMinutes: int
    observedDelayMinutes: int
    modelPredictedDelay: int
    inboundDelayMinutes: Optional[int] = None
    inboundFlightId: Optional[str] = None
    delayProbability: int

    # Risk metrics (0-100)
    networkCongestion: int
    propagationRisk: int
    precipSeverity: int
    incomingAircraftStatus: str

    # Weather
    weatherOrigin: Optional[WeatherWidget] = None
    weatherDest: Optional[WeatherWidget] = None

    # Live position
    livePosition: Optional[LivePosition] = None

    # Graph visualization
    graphData: Optional[GraphData] = None

    # Sources
    sources: List[Dict[str, str]] = []
    note: Optional[str] = None

    # Pipeline-only extras (populated by /analyze, None for /predict)
    llmSummary: Optional[str] = None
    alternativeRoutes: Optional[List[Itinerary]] = None
    agentLog: Optional[List[str]] = None

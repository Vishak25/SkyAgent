"""
LangGraph Orchestrator — coordinates the SkyAgent multi-agent pipeline.

Manages a StateGraph that routes work between:
  - FlightMonitorAgent  (live flight tracking)
  - WeatherAgent        (METAR/TAF analysis)
  - DelayRiskAgent      (ST-GNN inference + risk scoring)
  - ReroutingAgent      (alternative route recommendation, conditional)

Usage:
  from src.agents.orchestrator import run_pipeline
  result = await run_pipeline(flight_number="UAL123")
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.delay_risk_agent import DelayRiskAgent
from src.agents.flight_monitor import FlightMonitorAgent
from src.agents.rerouting_agent import ReroutingAgent
from src.agents.weather_agent import WeatherAgent
from src.config.settings import VLLM_API_KEY, VLLM_BASE_URL, VLLM_MODEL


# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    flight_number: str
    origin: Optional[str]
    destination: Optional[str]
    flight_status: Optional[Dict[str, Any]]
    weather_origin: Optional[Dict[str, Any]]
    weather_destination: Optional[Dict[str, Any]]
    predicted_delay: Optional[int]
    risk_score: Optional[float]
    alternative_routes: Optional[list]
    llm_summary: Optional[str]
    agent_log: List[str]


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        model=VLLM_MODEL,
        temperature=0.1,
        max_tokens=512,
    )


def build_pipeline() -> Any:
    """Build and compile the LangGraph agent pipeline."""
    llm = _build_llm()
    flight_agent = FlightMonitorAgent()
    weather_agent = WeatherAgent()
    delay_agent = DelayRiskAgent()
    rerouting_agent = ReroutingAgent()

    # ── Node: Flight Monitor ────────────────────────────────────────────────
    async def flight_monitor_node(state: AgentState) -> Dict[str, Any]:
        result = await flight_agent.run(state["flight_number"])
        status = result.get("status") or {}
        origin_obj = status.get("origin") or {}
        dest_obj = status.get("destination") or {}
        origin = (origin_obj.get("code_iata") or origin_obj.get("code") or "").upper()
        destination = (dest_obj.get("code_iata") or dest_obj.get("code") or "").upper()
        log = f"FlightMonitor: {state['flight_number']} {origin}→{destination} [{status.get('status', 'unknown')}]"
        return {
            "flight_status": result,
            "origin": origin,
            "destination": destination,
            "agent_log": state["agent_log"] + [log],
        }

    # ── Node: Weather ───────────────────────────────────────────────────────
    async def weather_node(state: AgentState) -> Dict[str, Any]:
        if not state.get("origin") or not state.get("destination"):
            return {"agent_log": state["agent_log"] + ["Weather: skipped (no origin/dest)"]}
        result = await weather_agent.run(state["origin"], state["destination"])
        return {
            "weather_origin": result.get("weatherOrigin"),
            "weather_destination": result.get("weatherDest"),
            "agent_log": state["agent_log"] + [f"Weather: fetched for {state['origin']}/{state['destination']}"],
        }

    # ── Node: Delay Risk ────────────────────────────────────────────────────
    async def delay_risk_node(state: AgentState) -> Dict[str, Any]:
        if not state.get("origin") or not state.get("destination"):
            return {"agent_log": state["agent_log"] + ["DelayRisk: skipped (no origin/dest)"]}
        result = await delay_agent.run(state["origin"], state["destination"])
        delay_mins = result.get("predictedDelayMinutes", 0)
        return {
            "predicted_delay": delay_mins,
            "agent_log": state["agent_log"] + [f"DelayRisk: {delay_mins}min predicted"],
        }

    # ── Node: Rerouting (conditional — only when delay ≥ threshold) ─────────
    async def rerouting_node(state: AgentState) -> Dict[str, Any]:
        result = await rerouting_agent.run(state["origin"], state["destination"])
        routes = result.get("itineraries") or []
        prompt = (
            f"Flight {state['flight_number']} from {state['origin']} to {state['destination']} "
            f"is predicted to be delayed {state['predicted_delay']} minutes. "
            f"Top alternative routes: {routes[:3]}. "
            "Recommend the best alternative in 2-3 sentences."
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return {
            "alternative_routes": routes,
            "llm_summary": response.content,
            "agent_log": state["agent_log"] + [f"Rerouting: {len(routes)} alternatives found"],
        }

    # ── Node: Summary (low/moderate delay path) ─────────────────────────────
    async def summary_node(state: AgentState) -> Dict[str, Any]:
        delay = state.get("predicted_delay") or 0
        w_origin = (state.get("weather_origin") or {}).get("raw_text", "N/A")
        w_dest = (state.get("weather_destination") or {}).get("raw_text", "N/A")
        prompt = (
            f"Flight {state['flight_number']} from {state.get('origin')} to {state.get('destination')}. "
            f"Predicted delay: {delay} minutes. "
            f"Weather at origin: {w_origin}. "
            f"Weather at destination: {w_dest}. "
            "Give a brief 2-3 sentence delay risk assessment for a passenger."
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return {
            "llm_summary": response.content,
            "agent_log": state["agent_log"] + ["Summary: generated"],
        }

    # ── Routing logic ────────────────────────────────────────────────────────
    def should_reroute(state: AgentState) -> str:
        delay = state.get("predicted_delay") or 0
        return "rerouting" if delay >= ReroutingAgent.DELAY_THRESHOLD_MINUTES else "summary"

    # ── Build graph ──────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("flight_monitor", flight_monitor_node)
    graph.add_node("weather", weather_node)
    graph.add_node("delay_risk", delay_risk_node)
    graph.add_node("rerouting", rerouting_node)
    graph.add_node("summary", summary_node)

    graph.set_entry_point("flight_monitor")
    graph.add_edge("flight_monitor", "weather")
    graph.add_edge("weather", "delay_risk")
    graph.add_conditional_edges(
        "delay_risk",
        should_reroute,
        {"rerouting": "rerouting", "summary": "summary"},
    )
    graph.add_edge("rerouting", END)
    graph.add_edge("summary", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Module-level pipeline (lazy init)
# ---------------------------------------------------------------------------

_pipeline = None


def get_pipeline() -> Any:
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


async def run_pipeline(flight_number: str) -> Dict[str, Any]:
    """Entry point for the agentic delay propagation pipeline."""
    pipeline = get_pipeline()
    initial_state: AgentState = {
        "flight_number": flight_number,
        "origin": None,
        "destination": None,
        "flight_status": None,
        "weather_origin": None,
        "weather_destination": None,
        "predicted_delay": None,
        "risk_score": None,
        "alternative_routes": None,
        "llm_summary": None,
        "agent_log": [],
    }
    return await pipeline.ainvoke(initial_state)

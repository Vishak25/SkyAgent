"""
LangGraph Orchestrator — coordinates the SkyAgent multi-agent pipeline.

Architecture:
  1. track_node     — calls get_prediction_for_flight() (the authoritative flight-grounded
                       predictor) using the trained ST-GNN singleton.  All flight data,
                       observed delay, inbound-aircraft propagation, weather, and position
                       come from this single source.
  2. summary_node   — LLM generates a 2–3 sentence passenger-facing risk narrative.
  3. rerouting_node — only when predicted_delay >= 30 min; fetches alternatives via
                       suggest_routes() and asks the LLM to recommend the best option.

Both LLM nodes have try/except with deterministic fallbacks so a Gemini timeout or quota
error never returns a 500.

Usage:
  from src.agents.orchestrator import init_pipeline, run_pipeline
  init_pipeline(graph_handler, model)        # called once at startup
  result = await run_pipeline("UAL123")
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.config.settings import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

_DELAY_THRESHOLD = 30  # minutes — triggers rerouting path


# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    flight_number: str
    origin: Optional[str]
    destination: Optional[str]
    predicted_delay: Optional[int]
    alternative_routes: Optional[list]
    llm_summary: Optional[str]
    agent_log: List[str]
    track_data: Optional[Dict[str, Any]]
    pipeline_error: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# LLM builder
# ---------------------------------------------------------------------------

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=512,
    )


# ---------------------------------------------------------------------------
# Pipeline builder — takes trained singletons, never creates a random STGNN
# ---------------------------------------------------------------------------

def build_pipeline(graph_handler, model) -> Any:
    """Build and compile the LangGraph agent pipeline."""
    llm = _build_llm()

    # ── Node: Track (flight-grounded core) ──────────────────────────────────
    async def track_node(state: AgentState) -> Dict[str, Any]:
        try:
            result = await asyncio.to_thread(
                graph_handler.get_prediction_for_flight,
                state["flight_number"],
                model,
            )
        except Exception as exc:
            err = {"error": "TRACK_ERROR", "detail": str(exc), "_status_code": 500}
            return {"pipeline_error": err, "agent_log": state["agent_log"] + [f"Track: error — {exc}"]}

        if isinstance(result, dict) and result.get("error"):
            return {
                "pipeline_error": result,
                "agent_log": state["agent_log"] + [f"Track: error — {result.get('error')}"],
            }

        origin = result.get("origin") or ""
        destination = result.get("destination") or ""
        delay = result.get("predictedDelayMinutes", 0)
        inbound = result.get("inboundDelayMinutes")
        inbound_note = f" (inbound +{inbound} min)" if inbound else ""
        log = (
            f"FlightMonitor: {state['flight_number']} {origin}→{destination} "
            f"[{result.get('status', 'unknown')}] — {delay} min predicted{inbound_note}"
        )
        return {
            "track_data": result,
            "origin": origin,
            "destination": destination,
            "predicted_delay": delay,
            "agent_log": state["agent_log"] + [log],
        }

    # ── Node: Summary (low/moderate delay path) ─────────────────────────────
    async def summary_node(state: AgentState) -> Dict[str, Any]:
        data = state.get("track_data") or {}
        delay = data.get("predictedDelayMinutes", 0)
        inbound = data.get("inboundDelayMinutes")
        weather_o = data.get("weatherOrigin") or {}
        weather_d = data.get("weatherDest") or {}
        origin = state.get("origin", "origin")
        dest = state.get("destination", "destination")

        inbound_clause = f" The inbound aircraft arrived {inbound} minutes late." if inbound else ""
        prompt = (
            f"Flight {state['flight_number']} from {origin} to {dest}. "
            f"Status: {data.get('status', 'Unknown')}. "
            f"Predicted delay: {delay} minutes.{inbound_clause} "
            f"Origin weather: {weather_o.get('condition', 'N/A')}, "
            f"wind {weather_o.get('windSpeed', 0)} kt, {weather_o.get('flightCategory', 'VFR')}. "
            f"Destination weather: {weather_d.get('condition', 'N/A')}, "
            f"wind {weather_d.get('windSpeed', 0)} kt, {weather_d.get('flightCategory', 'VFR')}. "
            "Give a brief 2–3 sentence delay risk assessment for a passenger."
        )

        try:
            response = await asyncio.wait_for(
                llm.ainvoke([HumanMessage(content=prompt)]),
                timeout=20.0,
            )
            summary = response.content
        except Exception:
            summary = (
                f"Flight {state['flight_number']} from {origin} to {dest} "
                f"shows a predicted delay of {delay} minute{'s' if delay != 1 else ''}. "
                f"Current conditions at {origin} are {weather_o.get('condition', 'unknown')} "
                f"and at {dest} are {weather_d.get('condition', 'unknown')}. "
                "Check with your airline for the latest updates."
            )

        return {
            "llm_summary": summary,
            "agent_log": state["agent_log"] + ["Summary: generated"],
        }

    # ── Node: Rerouting (delay ≥ threshold) ─────────────────────────────────
    async def rerouting_node(state: AgentState) -> Dict[str, Any]:
        origin = state.get("origin", "")
        dest = state.get("destination", "")
        delay = state.get("predicted_delay", 0)

        try:
            routes_result = await asyncio.to_thread(
                graph_handler.suggest_routes, origin, dest, model
            )
            routes = routes_result.get("itineraries") or []
        except Exception:
            routes = []

        if routes:
            top3 = [
                f"{r.get('flightNumber')} via {r.get('connectionHub') or 'direct'} "
                f"(~{r.get('predictedDelayMinutes', '?')} min, {r.get('delayRisk', '?')} risk)"
                for r in routes[:3]
            ]
            prompt = (
                f"Flight {state['flight_number']} from {origin} to {dest} "
                f"is predicted delayed {delay} minutes. "
                f"Top alternatives: {'; '.join(top3)}. "
                "Recommend the best alternative in 2–3 sentences."
            )
            try:
                response = await asyncio.wait_for(
                    llm.ainvoke([HumanMessage(content=prompt)]),
                    timeout=20.0,
                )
                summary = response.content
            except Exception:
                best = routes[0]
                via = f"via {best.get('connectionHub')}" if best.get("connectionHub") else "direct"
                summary = (
                    f"The best available alternative is {best.get('flightNumber', 'an alternate flight')} "
                    f"{via} with approximately {best.get('predictedDelayMinutes', '?')} minutes of predicted delay "
                    f"({best.get('delayRisk', 'unknown')} risk). "
                    "Please check availability with your airline directly."
                )
        else:
            summary = (
                f"With a {delay}-minute predicted delay on {state['flight_number']}, "
                "no alternative itineraries were found at this time. "
                "Please contact your airline for rebooking options."
            )

        return {
            "alternative_routes": routes,
            "llm_summary": summary,
            "agent_log": state["agent_log"] + [f"Rerouting: {len(routes)} alternatives found"],
        }

    # ── Routing logic ────────────────────────────────────────────────────────
    def route_after_track(state: AgentState) -> str:
        if state.get("pipeline_error"):
            return END
        delay = state.get("predicted_delay") or 0
        return "rerouting" if delay >= _DELAY_THRESHOLD else "summary"

    # ── Build graph ──────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("track", track_node)
    graph.add_node("summary", summary_node)
    graph.add_node("rerouting", rerouting_node)

    graph.set_entry_point("track")
    graph.add_conditional_edges(
        "track",
        route_after_track,
        {"rerouting": "rerouting", "summary": "summary", END: END},
    )
    graph.add_edge("rerouting", END)
    graph.add_edge("summary", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Module-level singletons — set by init_pipeline() at FastAPI startup
# ---------------------------------------------------------------------------

_pipeline = None
_graph_handler = None
_model = None


def init_pipeline(graph_handler, model) -> None:
    """Called once at startup with the trained graph_handler and model."""
    global _pipeline, _graph_handler, _model
    _graph_handler = graph_handler
    _model = model
    _pipeline = build_pipeline(graph_handler, model)


async def run_pipeline(flight_number: str) -> Dict[str, Any]:
    """Entry point for the agentic delay propagation pipeline."""
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized — call init_pipeline() at startup.")

    initial_state: AgentState = {
        "flight_number": flight_number,
        "origin": None,
        "destination": None,
        "predicted_delay": None,
        "alternative_routes": None,
        "llm_summary": None,
        "agent_log": [],
        "track_data": None,
        "pipeline_error": None,
    }

    final_state = await _pipeline.ainvoke(initial_state)

    # Surface track-level errors with their proper HTTP status code
    if final_state.get("pipeline_error"):
        return final_state["pipeline_error"]

    # Merge: rich flight dict + pipeline extras → one flat camelCase response
    result = dict(final_state.get("track_data") or {})
    result["llmSummary"] = final_state.get("llm_summary")
    result["alternativeRoutes"] = final_state.get("alternative_routes")
    result["agentLog"] = final_state.get("agent_log", [])
    return result

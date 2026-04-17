"""
WebSocket endpoint for real-time agent activity streaming (Phase 4).

Provides live updates from the LangGraph agent orchestrator to the
React frontend dashboard. Clients subscribe to flight events and
receive agent reasoning steps + delay updates in real time.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        data = json.dumps(message)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(data)
            except Exception:
                self.disconnect(connection)

    async def send_personal(self, message: Dict[str, Any], websocket: WebSocket) -> None:
        await websocket.send_text(json.dumps(message))


manager = ConnectionManager()


# NOTE: Register this router in main.py after Phase 4 agents are wired up.
# Example usage:
#
#   from fastapi import APIRouter
#   ws_router = APIRouter()
#
#   @ws_router.websocket("/ws/flights")
#   async def websocket_endpoint(websocket: WebSocket):
#       await manager.connect(websocket)
#       try:
#           while True:
#               data = await websocket.receive_text()
#               event = json.loads(data)
#               # Trigger agent pipeline and stream results back
#               await manager.send_personal({"type": "ack", "event": event}, websocket)
#       except WebSocketDisconnect:
#           manager.disconnect(websocket)

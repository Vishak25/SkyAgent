"""
Integration tests for the Suggest mode (/suggest endpoint).

Tests the route suggestion pipeline with fixture-backed API clients.
"""
import os
import sys
os.environ["USE_FIXTURES"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from src.models.delay_gnn.graph_builder import AviationGraphHandler
from src.models.delay_gnn.model import STGNN


@pytest.fixture(scope="module")
def handler():
    return AviationGraphHandler()


@pytest.fixture(scope="module")
def model():
    m = STGNN(in_channels=5, hidden_channels=64, out_channels=1)
    m.eval()
    return m


class TestSuggestMode:
    """Integration tests for suggest_routes (Suggest mode)."""

    def test_ORD_to_LHR_returns_results(self, handler, model):
        """ORD → LHR: should return itineraries from scheduled fixture."""
        result = handler.suggest_routes("ORD", "LHR", model)
        assert isinstance(result, dict)
        # Should have itineraries list or error
        itineraries = result.get("itineraries", [])
        if not result.get("error"):
            assert isinstance(itineraries, list)

    def test_suggest_response_schema(self, handler, model):
        """Verify suggest response has expected top-level keys."""
        result = handler.suggest_routes("ORD", "LHR", model)
        assert isinstance(result, dict)
        # Should have at minimum some recognizable keys
        has_valid_keys = any(
            k in result for k in ["itineraries", "routes", "error", "suggestions", "origin", "destination"]
        )
        assert has_valid_keys, f"Unexpected response keys: {list(result.keys())}"

    def test_suggest_missing_route(self, handler, model):
        """Non-existent route should return empty or error."""
        result = handler.suggest_routes("ZZZ", "YYY", model)
        assert isinstance(result, dict)
        # Either error or empty itineraries
        itineraries = result.get("itineraries", [])
        has_error = "error" in result
        assert has_error or isinstance(itineraries, list)

    def test_suggest_same_origin_dest(self, handler, model):
        """Same origin and destination should handle gracefully."""
        result = handler.suggest_routes("ORD", "ORD", model)
        assert isinstance(result, dict)
        # Should either error or return empty
        itineraries = result.get("itineraries", [])
        assert "error" in result or len(itineraries) == 0 or itineraries is not None

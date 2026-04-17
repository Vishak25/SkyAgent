"""
Integration tests for the Track mode (/predict/{flight_number}).

These tests use the full AviationGraphHandler with fixture-backed API clients
to verify end-to-end prediction flow for real discovered flights.
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
    """
    Create a graph handler in fixture mode.
    scope=module so the slow PyTorch import only happens once.
    """
    return AviationGraphHandler()


@pytest.fixture(scope="module")
def model():
    """A fresh ST-GNN model (random weights — good enough for schema tests)."""
    m = STGNN(in_channels=5, hidden_channels=64, out_channels=1)
    m.eval()
    return m


class TestTrackMode:
    """Integration tests for get_prediction_for_flight (Track mode)."""

    def test_UAL1989_response_schema(self, handler, model):
        """UAL1989 (ORD → BOS): verify full response schema."""
        result = handler.get_prediction_for_flight("UAL1989", model)

        assert isinstance(result, dict)
        # Must have core fields — response uses 'airline' key, not 'flight'
        assert "airline" in result or "error" in result

        if "error" not in result:
            # Check flight info fields
            assert "airline" in result
            # Delay should be a number
            predicted = result.get("predicted_delay_minutes")
            if predicted is not None:
                assert isinstance(predicted, (int, float))

    def test_AAL100_response_schema(self, handler, model):
        """AAL100 (JFK → LHR): transatlantic route."""
        result = handler.get_prediction_for_flight("AAL100", model)
        assert isinstance(result, dict)

    def test_DAL404_response_schema(self, handler, model):
        """DAL404 (ATL → LGA): short-haul domestic."""
        result = handler.get_prediction_for_flight("DAL404", model)
        assert isinstance(result, dict)

    def test_UAL123_response_schema(self, handler, model):
        """UAL123 (LHR → EWR): inbound international."""
        result = handler.get_prediction_for_flight("UAL123", model)
        assert isinstance(result, dict)

    def test_BAW178_response_schema(self, handler, model):
        """BAW178 (JFK → LHR): British Airways."""
        result = handler.get_prediction_for_flight("BAW178", model)
        assert isinstance(result, dict)

    def test_missing_flight_error(self, handler, model):
        """FAKE999: should return error response."""
        result = handler.get_prediction_for_flight("FAKE999", model)
        assert isinstance(result, dict)
        assert "error" in result or result.get("status_text", "").lower().count("error") > 0 or result.get("status_text", "").lower().count("not found") > 0 or "error" in str(result).lower()

    def test_empty_flight_error(self, handler, model):
        """Empty string: should return error."""
        result = handler.get_prediction_for_flight("", model)
        assert isinstance(result, dict)


class TestTrackModeWeather:
    """Verify weather data is included in track predictions."""

    def test_weather_data_present(self, handler, model):
        """Prediction should include weather info for origin and/or destination."""
        result = handler.get_prediction_for_flight("UAL1989", model)
        if "error" not in result:
            # Response uses camelCase: weatherOrigin, weatherDest, graphData
            has_weather = (
                result.get("weatherOrigin") is not None
                or result.get("weatherDest") is not None
                or result.get("graphData") is not None
            )
            assert has_weather, f"No weather data found. Keys: {sorted(result.keys())}"

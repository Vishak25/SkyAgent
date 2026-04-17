"""
Tests for API clients in fixture mode.

Verifies that all three clients (AeroAPI, CheckWX, OpenSky) correctly load
from JSON fixture files and return properly-structured data.
"""
import os
import sys
os.environ["USE_FIXTURES"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.tools.flight_tools import AeroAPIClient
from src.tools.weather_tools import CheckWXClient
from src.tools.opensky_tools import OpenSkyClient


# ---------------------------------------------------------------------------
# AeroAPI Client Tests
# ---------------------------------------------------------------------------

class TestAeroAPIClient:
    """Tests for the AeroAPI client in fixture mode."""

    def test_fixture_mode_enabled(self, aero_client):
        assert aero_client._fixture_mode is True
        assert aero_client.enabled is True

    def test_flight_status_UAL1989(self, aero_client):
        """UAL1989: ORD → BOS, real captured data."""
        status = aero_client.get_flight_status("UAL1989")
        assert isinstance(status, dict)
        assert "error" not in status
        origin = status.get("origin", {})
        dest = status.get("destination", {})
        assert origin.get("code_iata") == "ORD"
        assert dest.get("code_iata") == "BOS"
        assert status.get("operator") == "UAL"

    def test_flight_status_AAL100(self, aero_client):
        """AAL100: JFK → LHR, transatlantic."""
        status = aero_client.get_flight_status("AAL100")
        assert isinstance(status, dict)
        assert "error" not in status
        assert status.get("origin", {}).get("code_iata") == "JFK"
        assert status.get("destination", {}).get("code_iata") == "LHR"

    def test_flight_status_DAL404(self, aero_client):
        """DAL404: ATL → LGA, short-haul domestic."""
        status = aero_client.get_flight_status("DAL404")
        assert isinstance(status, dict)
        assert "error" not in status
        assert status.get("origin", {}).get("code_iata") == "ATL"
        assert status.get("destination", {}).get("code_iata") == "LGA"

    def test_flight_status_UAL123(self, aero_client):
        """UAL123: LHR → EWR, inbound transatlantic."""
        status = aero_client.get_flight_status("UAL123")
        assert isinstance(status, dict)
        assert "error" not in status
        assert status.get("origin", {}).get("code_iata") == "LHR"
        assert status.get("destination", {}).get("code_iata") == "EWR"

    def test_flight_status_BAW178(self, aero_client):
        """BAW178: JFK → LHR, British Airways."""
        status = aero_client.get_flight_status("BAW178")
        assert isinstance(status, dict)
        assert "error" not in status
        assert status.get("origin", {}).get("code_iata") == "JFK"
        assert status.get("destination", {}).get("code_iata") == "LHR"
        assert status.get("operator") == "BAW"

    def test_flight_status_missing_flight(self, aero_client):
        """FAKE999 should return an error dict."""
        status = aero_client.get_flight_status("FAKE999")
        assert isinstance(status, dict)
        assert status.get("error") == "FLIGHT_LOOKUP_FAILED"

    def test_flight_status_empty_ident(self, aero_client):
        """Empty string should return INVALID_FLIGHT error."""
        status = aero_client.get_flight_status("")
        assert status.get("error") == "INVALID_FLIGHT"

    def test_airport_info_KORD(self, aero_client):
        """Airport info for O'Hare."""
        info = aero_client.get_airport_info("KORD")
        assert info is not None
        assert info.name == "Chicago O'Hare Intl"
        assert info.icao == "KORD"

    def test_airport_info_KJFK(self, aero_client):
        info = aero_client.get_airport_info("KJFK")
        assert info is not None
        assert "Kennedy" in (info.name or "")

    def test_airport_info_EGLL(self, aero_client):
        info = aero_client.get_airport_info("EGLL")
        assert info is not None
        assert "Heathrow" in (info.name or "")

    def test_airport_info_missing(self, aero_client):
        """Unknown airport returns None."""
        info = aero_client.get_airport_info("ZZZZ")
        assert info is None

    def test_gate_congestion_KORD(self, aero_client):
        """Congestion for KORD from fixture (12 dep + 8 arr delays)."""
        cong = aero_client.get_gate_congestion("KORD")
        assert isinstance(cong, float)
        assert 0.0 <= cong <= 1.0
        # (12 + 8) / 200 = 0.1
        assert cong == 0.1

    def test_gate_congestion_missing(self, aero_client):
        """Missing congestion fixture returns 0.1 default."""
        cong = aero_client.get_gate_congestion("ZZZZ")
        assert cong == 0.1

    def test_scheduled_flights_KORD_EGLL(self, aero_client):
        """Scheduled flights ORD → LHR from synthetic fixture."""
        flights = aero_client.get_scheduled_flights("KORD", "EGLL")
        assert isinstance(flights, list)
        assert len(flights) == 2
        assert flights[0].get("ident") == "UAL123"

    def test_scheduled_flights_missing_route(self, aero_client):
        """Missing route returns empty list."""
        flights = aero_client.get_scheduled_flights("ZZZZ", "YYYY")
        assert flights == []

    def test_flight_position_fixture(self, aero_client):
        """Position returns empty dict in fixture mode."""
        pos = aero_client.get_flight_position("UAL1989")
        assert pos == {}

    def test_cache_hit(self, aero_client):
        """Second call should hit cache, not re-read fixture."""
        status1 = aero_client.get_flight_status("UAL1989")
        status2 = aero_client.get_flight_status("UAL1989")
        # Both should return identical data
        assert status1.get("origin", {}).get("code_iata") == status2.get("origin", {}).get("code_iata")

    def test_pick_best_flight(self, aero_client):
        """UAL1989 fixture has 15 entries — _pick_best_flight should select the most relevant."""
        status = aero_client.get_flight_status("UAL1989")
        # Should have origin and destination populated
        assert status.get("origin") is not None
        assert status.get("destination") is not None
        # Should have a scheduled_out time
        assert status.get("scheduled_out") is not None


# ---------------------------------------------------------------------------
# CheckWX Client Tests
# ---------------------------------------------------------------------------

class TestCheckWXClient:
    """Tests for the CheckWX client in fixture mode."""

    def test_fixture_mode_enabled(self, checkwx_client):
        assert checkwx_client._fixture_mode is True
        assert checkwx_client.enabled is True

    def test_metar_KORD(self, checkwx_client):
        """METAR for O'Hare from real captured data."""
        metar = checkwx_client.get_metar_data("KORD")
        assert isinstance(metar, dict)
        assert "flight_category" in metar
        assert metar["flight_category"] in ("VFR", "MVFR", "IFR", "LIFR")
        assert isinstance(metar.get("visibility_miles"), (int, float))
        assert isinstance(metar.get("wind_speed_kts"), (int, float))

    def test_metar_KJFK(self, checkwx_client):
        metar = checkwx_client.get_metar_data("KJFK")
        assert isinstance(metar, dict)
        assert "flight_category" in metar

    def test_metar_EGLL(self, checkwx_client):
        metar = checkwx_client.get_metar_data("EGLL")
        assert isinstance(metar, dict)
        assert "flight_category" in metar

    def test_metar_KBOS(self, checkwx_client):
        metar = checkwx_client.get_metar_data("KBOS")
        assert isinstance(metar, dict)
        assert "flight_category" in metar

    def test_metar_missing_airport(self, checkwx_client):
        """Missing METAR fixture returns VFR fallback."""
        metar = checkwx_client.get_metar_data("ZZZZ")
        assert metar.get("flight_category") == "VFR"
        assert metar.get("visibility_miles") == 10.0

    def test_metar_schema(self, checkwx_client):
        """Verify METAR response has all expected keys."""
        metar = checkwx_client.get_metar_data("KORD")
        expected_keys = [
            "visibility_miles", "wind_speed_kts", "ceiling_ft",
            "flight_category", "precip_severity", "precip_label",
        ]
        for key in expected_keys:
            assert key in metar, f"Missing key: {key}"

    def test_taf_KORD(self, checkwx_client):
        """TAF for O'Hare from real captured data."""
        taf = checkwx_client.get_taf_data("KORD")
        assert isinstance(taf, dict)
        assert "forecast_flight_category" in taf
        assert taf["forecast_flight_category"] in ("VFR", "MVFR", "IFR", "LIFR")
        assert isinstance(taf.get("forecast_periods"), int)
        assert taf["forecast_periods"] > 0

    def test_taf_KJFK(self, checkwx_client):
        taf = checkwx_client.get_taf_data("KJFK")
        assert isinstance(taf, dict)
        assert taf.get("forecast_periods", 0) > 0

    def test_taf_missing_airport(self, checkwx_client):
        """Missing TAF fixture returns VFR fallback."""
        taf = checkwx_client.get_taf_data("ZZZZ")
        assert taf.get("forecast_flight_category") == "VFR"

    def test_taf_schema(self, checkwx_client):
        """Verify TAF response has all expected keys."""
        taf = checkwx_client.get_taf_data("KORD")
        expected_keys = [
            "forecast_visibility_miles", "forecast_wind_speed_kts",
            "forecast_flight_category", "forecast_precip_severity",
            "forecast_precip_label", "forecast_periods",
        ]
        for key in expected_keys:
            assert key in taf, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# OpenSky Client Tests
# ---------------------------------------------------------------------------

class TestOpenSkyClient:
    """Tests for the OpenSky client in fixture mode."""

    def test_fixture_mode_enabled(self, opensky_client):
        assert opensky_client._fixture_mode is True

    def test_incoming_aircraft(self, opensky_client):
        """get_incoming_aircraft_distance returns a list of states."""
        states = opensky_client.get_incoming_aircraft_distance("KORD")
        assert isinstance(states, list)
        # Our fixture has 4 states, but filtering by distance removes 1
        assert len(states) >= 1

    def test_incoming_unknown_airport(self, opensky_client):
        """Unknown airport still returns the ORD fixture states (as fallback)."""
        states = opensky_client.get_incoming_aircraft_distance("ZZZZ")
        # Returns states from ORD fixture (all fixtures map to states_ORD.json)
        assert isinstance(states, list)

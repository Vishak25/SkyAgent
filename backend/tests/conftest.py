"""
Shared pytest fixtures for SkyAgent backend tests.

All tests run with USE_FIXTURES=1 so API clients load from JSON instead of live APIs.
"""
import os
import sys

# Ensure fixture mode is on for all tests
os.environ["USE_FIXTURES"] = "1"

# Add backend root to path so src.* imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.tools.flight_tools import AeroAPIClient
from src.tools.weather_tools import CheckWXClient
from src.tools.opensky_tools import OpenSkyClient


@pytest.fixture
def aero_client():
    """AeroAPI client backed by fixture JSON files."""
    return AeroAPIClient()


@pytest.fixture
def checkwx_client():
    """CheckWX client backed by fixture JSON files."""
    return CheckWXClient()


@pytest.fixture
def opensky_client():
    """OpenSky client backed by fixture JSON files."""
    return OpenSkyClient()

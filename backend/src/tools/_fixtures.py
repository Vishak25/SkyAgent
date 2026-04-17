"""Fixture loading for test/dev mode — shared by all tool modules."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"


def load_fixture(subdir: str, name: str) -> Optional[Any]:
    """Load a JSON fixture file, or return None if it doesn't exist."""
    path = _FIXTURES_DIR / subdir / f"{name}.json"
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

#!/usr/bin/env python3
"""
Install historical METAR weather data from Iowa Environmental Mesonet (IEM).

This script downloads CSV data joinable by station and UTC timestamp.
"""

from __future__ import annotations

import argparse
import csv
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import requests

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DEFAULT_STATIONS = [
    "KORD",
    "KJFK",
    "EGLL",
    "KLAX",
    "OMDB",
    "RJTT",
    "LFPG",
    "EHAM",
    "EDDF",
    "WSSS",
]


def _parse_time(value: str) -> datetime:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Empty datetime string.")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _norm_stations(stations: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for code in stations:
        c = (code or "").strip().upper()
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _default_output_path(repo_backend_dir: Path, start: datetime, end: datetime) -> Path:
    raw_dir = repo_backend_dir / "data" / "raw"
    stem = f"iem_metar_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}.csv"
    return raw_dir / stem


def _build_query(stations: Iterable[str], start: datetime, end: datetime) -> List[tuple]:
    params: List[tuple] = [
        ("data", "all"),
        ("tz", "UTC"),
        ("format", "onlycomma"),
        ("latlon", "yes"),
        ("elev", "yes"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "no"),
        ("report_type", "1"),
        ("report_type", "2"),
        ("sts", start.strftime("%Y-%m-%dT%H:%M:%SZ")),
        ("ets", end.strftime("%Y-%m-%dT%H:%M:%SZ")),
    ]
    for station in stations:
        params.append(("station", station))
    return params


def _row_count(csv_text: str) -> int:
    reader = csv.reader(io.StringIO(csv_text))
    # Subtract one for header if present.
    count = -1
    for _ in reader:
        count += 1
    return max(0, count)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download historical METAR from IEM.")
    parser.add_argument(
        "--stations",
        default=",".join(DEFAULT_STATIONS),
        help="Comma-separated station codes (typically ICAO), e.g. KORD,KJFK,EGLL",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="If --start/--end are omitted, download this many days ending now UTC.",
    )
    parser.add_argument(
        "--start",
        help="UTC start timestamp, ISO8601. Example: 2026-01-01T00:00:00Z",
    )
    parser.add_argument(
        "--end",
        help="UTC end timestamp, ISO8601. Example: 2026-02-01T00:00:00Z",
    )
    parser.add_argument(
        "--output",
        help="Output CSV file path. Defaults to backend/data/raw/iem_metar_<window>.csv",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="HTTP timeout in seconds.",
    )
    args = parser.parse_args()

    stations = _norm_stations(args.stations.split(","))
    if not stations:
        raise SystemExit("No valid stations provided.")

    if bool(args.start) ^ bool(args.end):
        raise SystemExit("Provide both --start and --end, or neither.")

    if args.start and args.end:
        start = _parse_time(args.start)
        end = _parse_time(args.end)
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(1, args.days_back))

    if start >= end:
        raise SystemExit("--start must be earlier than --end.")

    backend_dir = Path(__file__).resolve().parents[1]
    output_path = Path(args.output) if args.output else _default_output_path(backend_dir, start, end)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params = _build_query(stations, start, end)
    response = requests.get(IEM_ASOS_URL, params=params, timeout=args.timeout)
    response.raise_for_status()

    text = response.text
    if not text.startswith("station,valid"):
        sample = text[:200].replace("\n", " ")
        raise RuntimeError(f"Unexpected IEM response format: {sample}")

    output_path.write_text(text, encoding="utf-8")
    rows = _row_count(text)

    print(f"Downloaded IEM METAR: {rows} rows")
    print(f"Stations: {', '.join(stations)}")
    print(f"Window: {start.isoformat()} -> {end.isoformat()}")
    print(f"Saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

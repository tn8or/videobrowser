from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, time, timezone, timedelta
from typing import Dict, List, Tuple
import re

import numpy as np


@dataclass
class VboSession:
    path: str
    track_name: str
    session_start_utc: datetime
    start_line: Tuple[float, float, float, float] | None  # lat1, lon1, lat2, lon2
    columns: List[str]
    data: Dict[str, np.ndarray]


_TIME_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})\.(\d{2})$")


def _parse_time_field(t: str) -> Tuple[int, int, int, int]:
    match = _TIME_RE.match(t.strip())
    if not match:
        raise ValueError(f"Invalid time field: {t}")
    hh, mm, ss, cs = match.groups()
    return int(hh), int(mm), int(ss), int(cs)


def _parse_datetime_utc(date_str: str, time_str: str) -> datetime:
    # date_str: dd/mm/yyyy, time_str: HH:MM
    dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
    return dt.replace(tzinfo=timezone.utc)


def parse_vbo(path: str) -> VboSession:
    section = None
    comments: Dict[str, str] = {}
    columns: List[str] = []
    column_parts: List[str] = []
    data_rows: List[List[str]] = []
    start_line: Tuple[float, float, float, float] | None = None
    session_start_utc: datetime | None = None
    track_name = "Unknown Track"

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip("\n")
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("[") and stripped.endswith("]"):
                section = stripped.lower()
                continue

            if section == "[comments]":
                if ":" in stripped:
                    key, value = stripped.split(":", 1)
                    comments[key.strip()] = value.strip()
                continue

            if section == "[laptiming]":
                if stripped.lower().startswith("start"):
                    parts = stripped.split()
                    if len(parts) >= 5:
                        lat1 = float(parts[1])
                        lon1 = float(parts[2])
                        lat2 = float(parts[3])
                        lon2 = float(parts[4])
                        start_line = (lat1, lon1, lat2, lon2)
                continue

            if section == "[column names]":
                if "[data]" in stripped.lower():
                    cleaned = stripped.replace("[data]", "").strip()
                    if cleaned:
                        column_parts.append(cleaned)
                    joined = " ".join(column_parts).strip()
                    joined = re.sub(r"-\s+", "-", joined)
                    columns = [c for c in joined.split() if c]
                    section = "[data]"
                else:
                    column_parts.append(stripped)
                continue

            if section == "[data]" or "[data]" in stripped.lower():
                if "[data]" in stripped.lower():
                    if not columns and column_parts:
                        joined = " ".join(column_parts).strip()
                        joined = re.sub(r"-\s+", "-", joined)
                        columns = [c for c in joined.split() if c]
                    continue
                if not columns and column_parts:
                    joined = " ".join(column_parts).strip()
                    joined = re.sub(r"-\s+", "-", joined)
                    columns = [c for c in joined.split() if c]
                data_rows.append(stripped.split())

    if "UTC Date Started" in comments:
        date_time = comments["UTC Date Started"].split()
        if len(date_time) >= 2:
            session_start_utc = _parse_datetime_utc(date_time[0], date_time[1])

    if "Venue" in comments:
        track_name = comments["Venue"]

    if session_start_utc is None:
        raise ValueError("Missing UTC Date Started in VBO comments")

    if not columns:
        raise ValueError("Missing column names in VBO")

    data_matrix = list(zip(*data_rows)) if data_rows else []
    data: Dict[str, np.ndarray] = {}

    for idx, col in enumerate(columns):
        values = data_matrix[idx] if data_matrix else []
        if col == "time":
            times = []
            for t in values:
                hh, mm, ss, cs = _parse_time_field(t)
                dt = datetime.combine(session_start_utc.date(), time(hh, mm, ss, cs * 10000), tzinfo=timezone.utc)
                if dt < session_start_utc:
                    dt = dt + timedelta(days=1)
                times.append(dt.timestamp())
            data[col] = np.array(times, dtype=np.float64)
        else:
            data[col] = np.array([float(v) for v in values], dtype=np.float64)

    return VboSession(
        path=path,
        track_name=track_name,
        session_start_utc=session_start_utc,
        start_line=start_line,
        columns=columns,
        data=data,
    )

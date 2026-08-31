from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    type: str
    t_start: float
    t_end: float
    score: float
    recording_id: str
    track_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def t_peak(self) -> float:
        return 0.5 * (self.t_start + self.t_end)


@dataclass
class ClipWindow:
    t_start: float
    t_end: float
    recording_id: str
    types: list[str]
    score: float
    events: list[Event]


def merge_same_type(events: list[Event], gap: float = 1.0) -> list[Event]:
    grouped: dict[tuple[str, str, str | None], list[Event]] = {}
    for event in events:
        grouped.setdefault((event.recording_id, event.type, event.track_id), []).append(event)
    merged: list[Event] = []
    for items in grouped.values():
        items.sort(key=lambda e: e.t_start)
        current = items[0]
        for event in items[1:]:
            if event.t_start <= current.t_end + gap:
                current = Event(
                    type=current.type,
                    t_start=current.t_start,
                    t_end=max(current.t_end, event.t_end),
                    score=max(current.score, event.score),
                    recording_id=current.recording_id,
                    track_id=current.track_id,
                    extra={**current.extra, **event.extra},
                )
            else:
                merged.append(current)
                current = event
        merged.append(current)
    merged.sort(key=lambda e: (e.recording_id, e.t_start, e.type))
    return merged


def clip_windows(
    events: list[Event],
    pad: float,
    duration: float,
    max_len: float = 24.0,
    max_clips: int = 12,
) -> list[ClipWindow]:
    """Cut short clips around the strongest peaks instead of one file of the whole session."""
    if not events:
        return []
    half = min(pad, max(1.0, max_len / 2.0))
    ranked = sorted(events, key=lambda e: (-e.score, e.t_start))
    windows: list[ClipWindow] = []
    for event in ranked:
        if len(windows) >= max_clips:
            break
        peak = event.t_peak
        lo = max(0.0, peak - half)
        hi = min(duration, max(peak + half, lo + 1.0))
        if hi - lo > max_len:
            extra = (hi - lo - max_len) / 2.0
            lo += extra
            hi -= extra
        if any(min(hi, w.t_end) - max(lo, w.t_start) > 4.0 for w in windows):
            continue
        nearby = [e for e in events if e.t_end >= lo and e.t_start <= hi]
        if not nearby:
            nearby = [event]
        types: list[str] = []
        for item in nearby:
            if item.type not in types:
                types.append(item.type)
        windows.append(
            ClipWindow(
                t_start=lo,
                t_end=hi,
                recording_id=event.recording_id,
                types=types,
                score=max(e.score for e in nearby),
                events=nearby,
            )
        )
    windows.sort(key=lambda w: w.t_start)
    return windows


def events_to_json(events: list[Event], clips: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "events": [
            {
                "type": e.type,
                "t_start": round(e.t_start, 3),
                "t_end": round(e.t_end, 3),
                "score": round(e.score, 4),
                "recording_id": e.recording_id,
                "track_id": e.track_id,
                "extra": e.extra,
            }
            for e in events
        ],
        "clips": clips,
    }


def dumps_extra(extra: dict[str, Any]) -> str:
    return json.dumps(extra, default=str)

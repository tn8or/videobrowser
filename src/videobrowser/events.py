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
    merge_gap: float = 8.0,
) -> list[ClipWindow]:
    """Cut clips around the strongest peaks, merging near-duplicates into one file."""
    if not events:
        return []
    half = min(pad, max(1.0, max_len / 2.0))
    max_merge = max(max_len, 40.0)
    ranked = sorted(events, key=lambda e: (-e.score, e.t_start))
    windows: list[ClipWindow] = []

    def _types(bucket: list[Event]) -> list[str]:
        types: list[str] = []
        for item in bucket:
            if item.type not in types:
                types.append(item.type)
        return types

    def _make(lo: float, hi: float, bucket: list[Event]) -> ClipWindow:
        return ClipWindow(
            t_start=lo,
            t_end=hi,
            recording_id=bucket[0].recording_id,
            types=_types(bucket),
            score=max(e.score for e in bucket),
            events=bucket,
        )

    for event in ranked:
        peak = event.t_peak
        lo = max(0.0, peak - half)
        hi = min(duration, max(peak + half, lo + 1.0))
        if hi - lo > max_len:
            extra = (hi - lo - max_len) / 2.0
            lo += extra
            hi -= extra
        nearby = [e for e in events if e.t_end >= lo and e.t_start <= hi] or [event]
        host: int | None = None
        for i, existing in enumerate(windows):
            overlap = min(hi, existing.t_end) - max(lo, existing.t_start)
            gap = 0.0 if overlap >= 0 else -overlap
            union_lo = min(lo, existing.t_start)
            union_hi = max(hi, existing.t_end)
            if gap <= merge_gap and union_hi - union_lo <= max_merge:
                host = i
                break
        if host is not None:
            prev = windows[host]
            bucket = list(prev.events)
            for item in nearby:
                if item not in bucket:
                    bucket.append(item)
            windows[host] = _make(
                min(lo, prev.t_start),
                max(hi, prev.t_end),
                bucket,
            )
            continue
        if len(windows) >= max_clips:
            continue
        windows.append(_make(lo, hi, nearby))
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

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from videobrowser.probe import ProbeError, VideoInfo, probe

VIDEO_EXTS = {".mp4", ".mov", ".m4v"}
GOPRO_RE = re.compile(
    r"^(?P<prefix>G[HX])(?P<chapter>\d{2})(?P<recording>\d{4})$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chapter:
    path: Path
    prefix: str
    chapter: int
    recording: str
    duration: float
    offset: float
    info: VideoInfo


@dataclass
class Recording:
    recording_id: str
    prefix: str
    recording: str
    chapters: list[Chapter] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(ch.duration for ch in self.chapters)

    @property
    def paths(self) -> list[Path]:
        return [ch.path for ch in self.chapters]


def stem_match(path: Path) -> re.Match[str] | None:
    return GOPRO_RE.match(path.stem)


def recording_id_for(path: Path) -> str:
    match = stem_match(path)
    if match:
        return f"{match.group('prefix').upper()}-{match.group('recording')}"
    return path.stem


def iter_video_paths(inputs: list[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        item = item.expanduser().resolve()
        candidates: list[Path]
        if item.is_dir():
            candidates = sorted(
                p
                for p in item.iterdir()
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS
            )
        elif item.is_file():
            candidates = [item]
        else:
            raise FileNotFoundError(item)
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
    return found


def group_recordings(paths: list[Path]) -> list[Recording]:
    buckets: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    loners: list[Path] = []
    for path in paths:
        match = stem_match(path)
        if not match:
            loners.append(path)
            continue
        prefix = match.group("prefix").upper()
        recording = match.group("recording")
        chapter = int(match.group("chapter"))
        buckets.setdefault((prefix, recording), []).append((chapter, path))

    recordings: list[Recording] = []
    for (prefix, recording), items in sorted(buckets.items()):
        rec = Recording(
            recording_id=f"{prefix}-{recording}",
            prefix=prefix,
            recording=recording,
        )
        offset = 0.0
        skipped = False
        for chapter, path in sorted(items, key=lambda item: item[0]):
            try:
                info = probe(path)
            except ProbeError as exc:
                print(f"skip unreadable {path.name}: {exc}")
                skipped = True
                continue
            rec.chapters.append(
                Chapter(
                    path=path,
                    prefix=prefix,
                    chapter=chapter,
                    recording=recording,
                    duration=info.duration,
                    offset=offset,
                    info=info,
                )
            )
            offset += info.duration
        if rec.chapters:
            recordings.append(rec)
        elif skipped:
            continue

    for path in loners:
        try:
            info = probe(path)
        except ProbeError as exc:
            print(f"skip unreadable {path.name}: {exc}")
            continue
        rec = Recording(
            recording_id=path.stem,
            prefix="",
            recording=path.stem,
        )
        rec.chapters.append(
            Chapter(
                path=path,
                prefix="",
                chapter=1,
                recording=path.stem,
                duration=info.duration,
                offset=0.0,
                info=info,
            )
        )
        recordings.append(rec)
    return recordings


def chapter_at(recording: Recording, global_t: float) -> Chapter | None:
    for chapter in recording.chapters:
        if chapter.offset <= global_t < chapter.offset + chapter.duration:
            return chapter
    if recording.chapters and global_t >= recording.duration:
        return recording.chapters[-1]
    if recording.chapters and global_t < 0:
        return recording.chapters[0]
    return None

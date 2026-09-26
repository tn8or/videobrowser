from __future__ import annotations

import os
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


def qualified_recording_id(path: Path, input_roots: list[Path] | None = None) -> str:
    """GoPro id, prefixed with the relative folder when the file is nested.

    Direct children of an input folder stay ``GX-0510``. Nested library layout
    becomes ``2026-08-01_100GOPRO_GX-0510`` so the same card index on two
    different days does not collide in ``scan.sqlite`` or output names.
    """
    base = recording_id_for(path)
    if not input_roots:
        return base
    parent = path.parent.resolve()
    for root in input_roots:
        root = root.expanduser().resolve()
        try:
            rel = parent.relative_to(root)
        except ValueError:
            continue
        if rel.parts:
            return "_".join(rel.parts) + "_" + base
        return base
    return base


def _contained_in(path: Path, roots: list[Path]) -> bool:
    path = path.resolve()
    return any(path == root or root in path.parents for root in roots)


def _is_video_file(path: Path) -> bool:
    name = path.name
    if name.startswith(".") or name.startswith("._"):
        return False
    return path.suffix.lower() in VIDEO_EXTS


def _is_skipped_dir_name(name: str) -> bool:
    low = name.lower()
    return low.startswith(".") or low.endswith(".fcpbundle") or low == "__asynccopying"


def _is_fcp_internal(path: Path) -> bool:
    return any(
        part.lower().endswith(".fcpbundle") or part.lower() == "__asynccopying"
        for part in path.parts
    )


def _in_tree_path(path: Path, roots: list[Path]) -> Path | None:
    """Return a readable path that stays inside *roots*, or None to skip.

    Final Cut Original Media often contains aliases into a ``.fcpbundle``
    (including broken ``__AsyncCopying`` targets). Those must not pull the
    scan outside the folders the user passed.
    """
    if _is_fcp_internal(path):
        return None
    try:
        if path.is_symlink():
            target = path.resolve()
            if not target.exists() or not target.is_file():
                return None
            if _is_fcp_internal(target):
                return None
            if roots and not _contained_in(target, roots):
                return None
            return target
        if not path.is_file():
            return None
    except OSError:
        return None
    return path.resolve()


def _normalize_excludes(patterns: list[str] | None) -> list[str]:
    return [p.strip().lower() for p in (patterns or []) if p and p.strip()]


def _matches_exclude(path: Path, needles: list[str], root: Path | None = None) -> bool:
    """True if any needle is a substring of the file/folder name or relative path."""
    if not needles:
        return False
    texts = [path.name.lower()]
    if root is not None:
        try:
            texts.append(path.relative_to(root).as_posix().lower())
        except ValueError:
            pass
    else:
        texts.append(Path(path).as_posix().lower())
    return any(needle in text for text in texts for needle in needles)


def iter_video_paths(
    inputs: list[Path],
    *,
    exclude_roots: list[Path] | None = None,
    exclude_substrings: list[str] | None = None,
) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    exclude = [p.expanduser().resolve() for p in (exclude_roots or [])]
    needles = _normalize_excludes(exclude_substrings)
    skipped_outside = 0
    for item in inputs:
        item = item.expanduser().resolve()
        if exclude and _contained_in(item, exclude):
            continue
        if _matches_exclude(item, needles, root=item.parent if item.is_file() else item):
            continue
        stay_inside = [item] if item.is_dir() else [item.parent]
        candidates: list[Path]
        if item.is_dir():
            candidates = []
            for dirpath, dirnames, filenames in os.walk(item, followlinks=False):
                current = Path(dirpath)
                if exclude and _contained_in(current, exclude):
                    dirnames[:] = []
                    continue
                if _is_fcp_internal(current) or _matches_exclude(current, needles, root=item):
                    dirnames[:] = []
                    continue
                dirnames[:] = [
                    name
                    for name in dirnames
                    if not _is_skipped_dir_name(name)
                    and not (
                        exclude and _contained_in(current / name, exclude)
                    )
                    and not _matches_exclude(current / name, needles, root=item)
                ]
                for name in filenames:
                    path = current / name
                    if not _is_video_file(path):
                        continue
                    if _matches_exclude(path, needles, root=item):
                        continue
                    candidates.append(path)
            candidates.sort()
        elif item.is_file():
            if _matches_exclude(item, needles, root=item.parent):
                continue
            candidates = [item]
        else:
            raise FileNotFoundError(item)
        for path in candidates:
            usable = _in_tree_path(path, stay_inside)
            if usable is None:
                if path.is_symlink():
                    skipped_outside += 1
                continue
            if usable in seen:
                continue
            if exclude and _contained_in(usable, exclude):
                continue
            seen.add(usable)
            found.append(usable)
    if skipped_outside:
        print(
            f"skipped {skipped_outside} linked file(s) pointing outside the input folders"
        )
    return found


def group_recordings(
    paths: list[Path],
    input_roots: list[Path] | None = None,
) -> list[Recording]:
    buckets: dict[tuple[str, str, str], list[tuple[int, Path]]] = {}
    loners: list[Path] = []
    for path in paths:
        match = stem_match(path)
        if not match:
            loners.append(path)
            continue
        prefix = match.group("prefix").upper()
        recording = match.group("recording")
        chapter = int(match.group("chapter"))
        parent = str(path.parent.resolve())
        buckets.setdefault((parent, prefix, recording), []).append((chapter, path))

    recordings: list[Recording] = []
    for (_parent, prefix, recording), items in sorted(buckets.items()):
        rec = Recording(
            recording_id=qualified_recording_id(items[0][1], input_roots),
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
            recording_id=qualified_recording_id(path, input_roots),
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

"""Чтение сплошного текста и сегментов Whisper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re

_HEADER = re.compile(r"^# ([^:]+):\s*(.*)$")
_SEGMENT = re.compile(
    r"^\[(?P<start>[0-9:.]+)\s+-->\s+(?P<end>[0-9:.]+)\]\s*(?P<text>.*)$"
)
_VIDEO_ID = re.compile(r"(-?\d+_\d+)")


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class LectureText:
    slug: str
    plain_path: Path
    timed_path: Path | None
    header: dict[str, str]
    text: str
    segments: list[Segment]
    video_id: str | None = None

    @property
    def source_url(self) -> str:
        return self.header.get("Источник", "")

    @property
    def model(self) -> str:
        return self.header.get("Модель", "")

    @property
    def language(self) -> str:
        return self.header.get("Язык", "")

    @property
    def transcribed_at(self) -> str:
        return self.header.get("Дата", "")

    @property
    def header_title(self) -> str:
        return self.header.get("Название", "")


def corpus_hash(result_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(result_dir.glob("*.txt")):
        if not path.is_file():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_lectures(result_dir: Path, timed_dir: Path) -> tuple[list[LectureText], str]:
    lectures: list[LectureText] = []
    digest = hashlib.sha256()
    for path in sorted(result_dir.glob("*.txt")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(raw)
        lectures.append(_load_one(path, raw, timed_dir))
    return lectures, digest.hexdigest()


def parse_clock(value: str) -> float:
    text = value.strip()
    millis = 0
    if "." in text:
        main, frac = text.split(".", 1)
        millis = int((frac + "000")[:3])
    else:
        main = text
    parts = [int(part) for part in main.split(":") if part != ""]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 1:
        hours, minutes, seconds = 0, 0, parts[0]
    else:
        raise ValueError(value)
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def video_id_from_url(url: str) -> str | None:
    match = _VIDEO_ID.search(url or "")
    if not match:
        return None
    return match.group(1)


def _load_one(path: Path, raw: bytes, timed_dir: Path) -> LectureText:
    header, text = split_header(raw.decode("utf-8", errors="replace"))
    timed_path = timed_dir / f"{path.stem}.mp4.txt"
    segments: list[Segment] = []
    if timed_path.is_file() and timed_path.stat().st_size > 0:
        segments = parse_segments(timed_path.read_text(encoding="utf-8", errors="replace"))
    return LectureText(
        slug=path.stem,
        plain_path=path,
        timed_path=timed_path if segments else None,
        header=header,
        text=text,
        segments=segments,
        video_id=video_id_from_url(header.get("Источник", "")),
    )


def split_header(raw: str) -> tuple[dict[str, str], str]:
    lines = raw.splitlines()
    header: dict[str, str] = {}
    index = 0
    for index, line in enumerate(lines):
        match = _HEADER.match(line)
        if match is None:
            break
        header[match.group(1).strip()] = match.group(2).strip()
    else:
        index = len(lines)
    if header and index < len(lines) and lines[index].strip() == "":
        index += 1
    if not header:
        index = 0
    text = "\n".join(lines[index:]).strip()
    return header, text


def parse_segments(raw: str) -> list[Segment]:
    segments: list[Segment] = []
    for line in raw.splitlines():
        match = _SEGMENT.match(line.strip())
        if match is None:
            if segments and line.strip() and not line.startswith("#"):
                segments[-1].text = f"{segments[-1].text} {line.strip()}"
            continue
        start = parse_clock(match.group("start"))
        end = parse_clock(match.group("end"))
        if end < start:
            end = start
        segments.append(Segment(start=start, end=end, text=match.group("text").strip()))
    return segments

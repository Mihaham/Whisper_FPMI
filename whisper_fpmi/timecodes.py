"""Таймкоды из описания VK, включая метки длиннее часа."""

from __future__ import annotations

from dataclasses import dataclass
import html
import re

# Сначала часы:минуты:секунды, иначе «1:05:30» распадётся на минуты.
_TS = re.compile(
    r"(?<![\d:])(?:"
    r"(?P<h>\d{1,3}):(?P<m>[0-5]\d):(?P<s>[0-5]\d)"
    r"|(?P<mm>\d{1,3}):(?P<ss>[0-5]\d)"
    r")(?!\d)"
)
_LABEL = re.compile(
    r"^\s*(?:таймкоды|таймкоды лекции|главы|содержание|timestamps?)\s*[:\-–—]?\s*",
    re.IGNORECASE,
)
_PREFIX = re.compile(r"\s*(?:(?:\d{1,2}[.)]|[-–—*•])\s*)?")
_BR = re.compile(r"(?i)<br\s*/?>")
_TAG = re.compile(r"<[^>]+>")
_ROLE = re.compile(
    r"^\s*(?P<role>лекторы|лектор|семинаристы|семинарист|преподаватели|"
    r"преподаватель|читает|читали)\s*[:\-–—]\s*(?P<name>.+?)\s*$",
    re.IGNORECASE,
)
_DATE = re.compile(
    r"(?im)^[ \t]*дата[^:\n]{0,48}:[ \t]*(\d{1,2}\.\d{1,2}\.\d{2,4})[ \t]*$"
)
_ROLE_RANK = {
    "лектор": 0,
    "лекторы": 0,
    "семинарист": 1,
    "семинаристы": 1,
    "преподаватель": 2,
    "преподаватели": 2,
    "читает": 3,
    "читали": 3,
}


@dataclass(frozen=True)
class Timecode:
    seconds: int
    clock: str
    title: str
    hour_field: bool

    def as_dict(self) -> dict:
        return {
            "seconds": self.seconds,
            "clock": self.clock,
            "title": self.title,
            "hour_field": self.hour_field,
        }


def format_clock(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_clock_full(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def html_to_text(value: str) -> str:
    text = _BR.sub("\n", value or "")
    text = _TAG.sub("", text)
    text = html.unescape(text)
    return text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")


def parse_description_timecodes(description: str) -> list[Timecode]:
    """Достаёт главы. «1:05:30» — это час, пять минут и тридцать секунд."""
    found: list[Timecode] = []
    for raw_line in html_to_text(description).splitlines():
        found.extend(_parse_line(raw_line))
    found.sort(key=lambda item: (item.seconds, item.title))
    unique: list[Timecode] = []
    seen: set[tuple[int, str]] = set()
    for item in found:
        key = (item.seconds, item.title)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def parse_lecturer(description: str) -> str | None:
    """Лектор, семинарист или преподаватель. Оператор и монтажёр не считаются."""
    best: tuple[int, str] | None = None
    for raw_line in html_to_text(description).splitlines():
        match = _ROLE.match(raw_line.strip())
        if not match:
            continue
        name = re.sub(r"\s+", " ", match.group("name")).strip(" .-")
        if len(name) < 3:
            continue
        rank = _ROLE_RANK.get(match.group("role").casefold(), 9)
        if best is None or rank < best[0]:
            best = (rank, name)
    if best is None:
        return None
    return best[1]


def parse_event_date(description: str) -> str | None:
    match = _DATE.search(html_to_text(description))
    if not match:
        return None
    day, month, year = match.group(1).split(".")
    if len(year) == 2:
        year = f"20{year}"
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _parse_line(raw_line: str) -> list[Timecode]:
    line = _LABEL.sub("", raw_line, count=1)
    matches = list(_TS.finditer(line))
    if not matches:
        return []
    prefix = _PREFIX.match(line)
    if prefix is None or prefix.end() != matches[0].start():
        return []
    codes: list[Timecode] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        title = line[match.end() : end].strip(" \t-–—:|.,;")
        title = re.sub(r"\s+", " ", title).strip()
        seconds, hour_field = _seconds(match)
        codes.append(
            Timecode(
                seconds=seconds,
                clock=format_clock_full(seconds) if hour_field else format_clock(seconds),
                title=title,
                hour_field=hour_field,
            )
        )
    return codes


def _seconds(match: re.Match) -> tuple[int, bool]:
    if match.group("h") is not None:
        hours = int(match.group("h"))
        minutes = int(match.group("m"))
        seconds = int(match.group("s"))
        return hours * 3600 + minutes * 60 + seconds, True
    minutes = int(match.group("mm"))
    seconds = int(match.group("ss"))
    return minutes * 60 + seconds, False

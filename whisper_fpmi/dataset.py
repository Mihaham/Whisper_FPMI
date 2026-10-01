"""Таблица лекций: длительность, лектор, таймкоды, служебные слова."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import statistics

from whisper_fpmi.lang import TextStats, analyze, stem_token
from whisper_fpmi.names import ParsedLecture, parse_stem
from whisper_fpmi.timecodes import parse_description_timecodes, parse_event_date, parse_lecturer
from whisper_fpmi.transcript import LectureText


def lecture_kind(parsed: ParsedLecture) -> str:
    blob = " ".join(
        part for part in (parsed.tag, parsed.stream, parsed.course) if part
    ).casefold()
    if "консультац" in blob:
        return "консультация"
    if "допсем" in blob or "доп-сем" in blob:
        return "допсем"
    if "липс" in blob:
        return "липс"
    if "семинар" in blob:
        return "семинар"
    return "лекция"


def make_record(
    lecture: LectureText,
    *,
    description: str = "",
    vk_duration: float | None = None,
    lemmatize=stem_token,
    root: Path | None = None,
) -> tuple[dict, TextStats]:
    parsed = parse_stem(lecture.slug)
    stats = analyze(lecture.text, lemmatize)
    timing = _timing(lecture, vk_duration)
    codes = [item.as_dict() for item in parse_description_timecodes(description)]
    duration = timing["duration_sec"]
    words = stats.tokens
    record = {
        "slug": lecture.slug,
        "video_id": lecture.video_id,
        "title": parsed.display_title,
        "header_title": lecture.header_title,
        "course": parsed.display_course,
        "number": parsed.number,
        "stream": parsed.stream,
        "tag": parsed.tag,
        "kind": lecture_kind(parsed),
        "url": lecture.source_url,
        "model": lecture.model,
        "language": lecture.language,
        "transcribed_at": lecture.transcribed_at,
        "plain": _rel(lecture.plain_path, root),
        "timed": _rel(lecture.timed_path, root) if lecture.timed_path else None,
        "vk_duration_sec": _num(vk_duration, 3),
        "chars": len(lecture.text),
        "words": words,
        "wpm": _wpm(words, duration),
        "function_words": stats.function_words,
        "function_per_1k": _num(stats.per_1k(stats.function_words), 2),
        "fillers": stats.fillers,
        "filler_per_1k": _num(stats.per_1k(stats.fillers), 2),
        "filler_counts": stats.filler_counts,
        "lexical_density": _num(stats.lexical_density, 3),
        "unique_lemmas": len(stats.counts),
        "lecturer": parse_lecturer(description),
        "event_date": parse_event_date(description),
        "description": description.strip(),
        "timecodes": codes,
        "timecode_count": len(codes),
        "timecodes_with_hour_field": sum(1 for item in codes if item["hour_field"]),
        "timecodes_past_one_hour": sum(1 for item in codes if item["seconds"] >= 3600),
    }
    record.update(timing)
    return record, stats


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    text = "\n".join(lines)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


CSV_COLUMNS = (
    "slug",
    "video_id",
    "title",
    "header_title",
    "course",
    "number",
    "stream",
    "tag",
    "kind",
    "lecturer",
    "event_date",
    "url",
    "model",
    "language",
    "transcribed_at",
    "plain",
    "timed",
    "duration_sec",
    "vk_duration_sec",
    "segments",
    "median_segment_sec",
    "max_segment_sec",
    "speech_sec",
    "gap_sec",
    "chars",
    "words",
    "wpm",
    "function_words",
    "function_per_1k",
    "fillers",
    "filler_per_1k",
    "filler_counts",
    "lexical_density",
    "unique_lemmas",
    "timecode_count",
    "timecodes_with_hour_field",
    "timecodes_past_one_hour",
    "timecodes",
    "description",
)


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({column: _csv_cell(record, column) for column in CSV_COLUMNS})


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _timing(lecture: LectureText, vk_duration: float | None) -> dict:
    segments = lecture.segments
    if not segments:
        duration = _num(vk_duration, 3)
        return {
            "duration_sec": duration,
            "segments": 0,
            "median_segment_sec": None,
            "max_segment_sec": None,
            "speech_sec": None,
            "gap_sec": None,
        }
    lengths = [max(segment.end - segment.start, 0.0) for segment in segments]
    gaps = []
    for previous, following in zip(segments, segments[1:]):
        gap = following.start - previous.end
        if gap > 0.3:
            gaps.append(gap)
    return {
        "duration_sec": round(segments[-1].end, 3),
        "segments": len(segments),
        "median_segment_sec": round(float(statistics.median(lengths)), 3),
        "max_segment_sec": round(max(lengths), 3),
        "speech_sec": round(sum(lengths), 3),
        "gap_sec": round(sum(gaps), 3),
    }


def _wpm(words: int, duration: float | None) -> float | None:
    if not duration or duration <= 0 or not words:
        return None
    return round(words / (duration / 60.0), 2)


def _num(value, digits: int):
    if value is None:
        return None
    return round(float(value), digits)


def _csv_cell(record: dict, column: str) -> str:
    if column == "timecodes":
        parts = []
        for item in record.get("timecodes") or []:
            clock = item.get("clock") or ""
            title = (item.get("title") or "").replace("\n", " ").strip()
            parts.append(f"{clock} {title}".strip())
        return "; ".join(parts)
    if column == "filler_counts":
        counts = record.get("filler_counts") or {}
        return "; ".join(f"{word}={count}" for word, count in sorted(counts.items()))
    value = record.get(column)
    if value is None:
        return ""
    return str(value)


def _rel(path: Path | None, root: Path | None) -> str | None:
    if path is None:
        return None
    if root is None:
        return path.as_posix()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()

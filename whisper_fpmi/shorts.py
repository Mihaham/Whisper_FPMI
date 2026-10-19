"""Нарезка шортса: места, где в лекции звучит выбранное слово.

Видео в репозитории не лежат. Индекс строится по пословным таймкодам из
`download/*.mp4.txt`, а на сборке нужные куски докачиваются с VK или YouTube.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from urllib.parse import parse_qs, quote, urlparse

from loguru import logger as lg

from whisper_fpmi.lang import STOPWORDS, fold, get_lemmatizer
from whisper_fpmi.names import parse_stem, sanitize_title
from whisper_fpmi.download import parse_download_hook
from whisper_fpmi.paths import (
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_FRAGMENT_THREADS,
    LECTURES_CSV,
    SHORTS_DIR,
    SHORTS_INDEX,
    TIMED_DIR,
    WORDCLOUD_JSON,
)
from whisper_fpmi.search import vk_link
from whisper_fpmi.timecodes import format_clock_full
from whisper_fpmi.transcript import parse_segments, split_header

INDEX_VERSION = 2
MAX_VIDEOS = 5
SAMPLE_PER_LECTURE = 8
INTRO_SECONDS = 1.7
_EDGE = ".,!?;:\"'«»“”()[]{}<>…\\/"
_INDEX_LOCK = threading.Lock()
_PHRASE_LOCK = threading.Lock()
_PHRASE_CACHE: dict[tuple, list["Hit"]] = {}

SOURCE_DIR = SHORTS_DIR / "sources"
POPULAR_DIR = SHORTS_DIR / "popular"
# Ниже этого запаса на диске исходники, которые больше не нужны, удаляются.
DEFAULT_RESERVE_GB = 50.0
_MEAN_VOLUME = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB")
_VK_ID = re.compile(r"^-?\d+_\d+$")
TARGET_MEAN_DB = -16.0
_URLS_LOCK = threading.Lock()
_URLS: dict[str, str] | None = None
_LECTURERS: dict[str, str] | None = None
# Без потолка по высоте: yt-dlp берёт лучшее видео и лучший звук, иначе лучший цельный файл.
BEST_FORMAT = "bestvideo*+bestaudio/best"
_ENCODE = [
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "21",
    "-pix_fmt", "yuv420p",
    "-profile:v", "high",
    "-level", "4.0",
    "-r", "30",
    "-g", "60",
    "-c:a", "aac",
    "-b:a", "160k",
    "-ar", "48000",
    "-ac", "2",
    "-movflags", "+faststart",
]


class RenderError(RuntimeError):
    pass


class BusyError(RuntimeError):
    pass


@dataclass(frozen=True)
class CutStyle:
    pad_before: float
    pad_after: float
    min_clip: float
    max_clip: float
    min_gap: float


STYLES = {
    "tight": CutStyle(0.14, 0.28, 0.50, 1.00, 0.55),
    "phrase": CutStyle(0.40, 0.75, 1.00, 2.60, 1.00),
}


@dataclass(frozen=True)
class Settings:
    query: str
    tokens: tuple[str, ...]
    mode: str
    style: str
    seconds: float
    max_per_lecture: int
    seed: int
    course: str


@dataclass(frozen=True)
class Hit:
    slug: str
    title: str
    course: str
    url: str
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class Clip:
    hit: Hit
    start: float
    end: float
    context: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class WordSearch:
    query: str
    label: str
    mode: str
    total: int
    lectures: int
    forms: list[dict[str, int | str]]
    hits: list[Hit]
    indexed: bool
    note: str
    mentions: dict[str, int] | None = None


@dataclass
class Catalog:
    path: Path
    timed_dir: Path
    lemmatize: object
    name: str
    signature: str


@dataclass
class Rendered:
    path: Path
    clips: int
    skipped: int


def cue_word(text: str) -> str | None:
    """Одно слово Whisper, без краевой пунктуации. Фразу из нескольких слов не режем."""
    raw = text.strip()
    if not raw or any(char.isspace() for char in raw):
        return None
    cleaned = raw.strip(_EDGE + "-–—")
    if not cleaned or not any(char.isalnum() for char in cleaned):
        return None
    return cleaned


def query_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in text.split():
        word = cue_word(raw)
        if word:
            tokens.append(fold(word))
    return tokens


def overlay_label(query: str) -> str:
    text = " ".join(query.split())
    if not text:
        return "слово"
    if " " not in text:
        return text.upper()
    return text


def indexable(word: str) -> bool:
    if len(word) < 3 or len(word) > 40 or word.isdigit():
        return False
    return word not in STOPWORDS


def even_sample(items: list, limit: int) -> list:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (limit - 1)
    picked: list = []
    seen: set[int] = set()
    for index in range(limit):
        pos = round(index * step)
        if pos in seen:
            continue
        seen.add(pos)
        picked.append(items[pos])
    return picked


def clip_window(start: float, end: float, style: CutStyle) -> tuple[float, float]:
    spoken_end = min(max(end, start), start + 0.8)
    cut_start = max(0.0, start - style.pad_before)
    cut_end = spoken_end + style.pad_after
    if cut_end - cut_start < style.min_clip:
        cut_end = cut_start + style.min_clip
    if cut_end - cut_start > style.max_clip:
        mid = (start + spoken_end) / 2.0
        cut_start = max(0.0, mid - style.max_clip / 2.0)
        cut_end = cut_start + style.max_clip
    return cut_start, cut_end


def phrase_spans(
    cues: list[tuple[float, float, str]],
    tokens: tuple[str, ...] | list[str],
    key,
) -> list[tuple[float, float]]:
    """Подряд идущие слова. Между ними не больше 1.5 секунды тишины."""
    if not tokens:
        return []
    want = [key(token) for token in tokens]
    got = [key(word) for _, _, word in cues]
    size = len(want)
    spans: list[tuple[float, float]] = []
    index = 0
    last = len(cues) - size
    while index <= last:
        if got[index:index + size] != want or not _tight_gap(cues, index, size):
            index += 1
            continue
        spans.append((cues[index][0], cues[index + size - 1][1]))
        index += size
    return spans


def _tight_gap(cues: list[tuple[float, float, str]], index: int, size: int) -> bool:
    for step in range(size - 1):
        gap = cues[index + step + 1][0] - cues[index + step][1]
        if gap > 1.5:
            return False
    return True


def plan_clips(
    hits: list[Hit],
    *,
    style: CutStyle,
    target_seconds: float,
    max_per_lecture: int,
    seed: int,
    intro_seconds: float = INTRO_SECONDS,
    max_clips: int = 70,
    counts: dict[str, int] | None = None,
    max_videos: int = MAX_VIDEOS,
    lecturers: dict[str, str] | None = None,
) -> list[Clip]:
    """Первый ролик: внутри него клипы одной лекции идут подряд."""
    videos = plan_shorts(
        hits,
        style=style,
        target_seconds=target_seconds,
        max_per_lecture=max_per_lecture,
        seed=seed,
        intro_seconds=intro_seconds,
        max_clips=max_clips,
        counts=counts,
        max_videos=max_videos,
        lecturers=lecturers,
    )
    return videos[0] if videos else []


def plan_shorts(
    hits: list[Hit],
    *,
    style: CutStyle,
    target_seconds: float,
    max_per_lecture: int,
    seed: int,
    intro_seconds: float = INTRO_SECONDS,
    max_clips: int = 70,
    counts: dict[str, int] | None = None,
    max_videos: int = MAX_VIDEOS,
    lecturers: dict[str, str] | None = None,
) -> list[list[Clip]]:
    """До max_videos роликов. Сначала лектор, у которого слово звучит чаще."""
    prepared = _lectures_in_order(hits, style, max_per_lecture, seed, counts, lecturers)
    budget = target_seconds - intro_seconds
    if budget <= 0:
        budget = target_seconds
    videos: list[list[Clip]] = []
    current: list[Clip] = []
    used = 0.0
    for block in prepared:
        span = sum(clip.duration for clip in block)
        overflows = current and (
            used + span > budget + 0.35 or len(current) + len(block) > max_clips
        )
        if overflows:
            videos.append(current)
            current = []
            used = 0.0
        current.extend(block)
        used += span
    if current:
        videos.append(current)
    if max_videos > 0:
        return videos[:max_videos]
    return videos


def _lectures_in_order(
    hits: list[Hit],
    style: CutStyle,
    max_per_lecture: int,
    seed: int,
    counts: dict[str, int] | None = None,
    lecturers: dict[str, str] | None = None,
) -> list[list[Clip]]:
    del seed  # порядок задаёт число вхождений, не перемешивание
    grouped: dict[str, list[Hit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.slug].append(hit)
    ranked: list[tuple[int, str, list[Clip]]] = []
    for slug, group in grouped.items():
        group.sort(key=lambda item: item.start)
        picked = even_sample(_thin_lecture(group, style), max_per_lecture)
        if not picked:
            continue
        weight = counts[slug] if counts and slug in counts else len(group)
        ranked.append((weight, slug, picked))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return _by_lecturer(ranked, lecturers)


def _by_lecturer(
    ranked: list[tuple[int, str, list[Clip]]],
    lecturers: dict[str, str] | None,
) -> list[list[Clip]]:
    """Лекции одного человека идут подряд. Сначала тот, у кого слово чаще."""
    buckets: dict[str, list[tuple[int, str, list[Clip]]]] = defaultdict(list)
    for weight, slug, picked in ranked:
        name = ""
        if lecturers is not None:
            name = lecturers.get(slug, "")
        else:
            name = lecturer_of(slug)
        key = name or f"#{slug}"
        buckets[key].append((weight, slug, picked))
    bundles = []
    for key, items in buckets.items():
        total = sum(weight for weight, _slug, _picked in items)
        bundles.append((total, key, items))
    bundles.sort(key=lambda item: (-item[0], item[1]))
    ordered: list[list[Clip]] = []
    for _total, _key, items in bundles:
        items.sort(key=lambda item: (-item[0], item[1]))
        for _weight, _slug, picked in items:
            ordered.append(picked)
    return ordered


def _thin_lecture(group: list[Hit], style: CutStyle) -> list[Clip]:
    clips: list[Clip] = []
    last_end = -1e9
    for hit in group:
        start, end = clip_window(hit.start, hit.end, style)
        if end - start < 0.2 or start < last_end + style.min_gap:
            continue
        clips.append(Clip(hit, start, end))
        last_end = end
    return clips


def timed_slug(path: Path) -> str:
    name = path.name
    marker = ".mp4.txt"
    if name.endswith(marker):
        return name[: -len(marker)]
    return path.stem


def corpus_signature(timed_dir: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in timed_dir.glob("*.mp4.txt") if path.is_file())
    digest.update(str(len(paths)).encode())
    for path in paths:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    digest.update(str(INDEX_VERSION).encode())
    return digest.hexdigest()


def make_settings(
    query: str,
    *,
    mode: str = "forms",
    style: str = "tight",
    seconds: float = 45,
    max_per_lecture: int = 3,
    seed: int = 1,
    course: str | None = None,
) -> Settings:
    text = " ".join(str(query or "").split())
    tokens = query_tokens(text)
    if not tokens:
        raise ValueError("Нужно слово")
    if len(tokens) > 5:
        raise ValueError("Для нарезки хватит фразы до пяти слов")
    if mode not in {"exact", "forms"}:
        mode = "forms"
    if style not in STYLES:
        style = "tight"
    try:
        length = float(seconds)
    except (TypeError, ValueError):
        length = 45.0
    try:
        per_lecture = int(max_per_lecture)
    except (TypeError, ValueError):
        per_lecture = 3
    try:
        mix = int(seed)
    except (TypeError, ValueError):
        mix = 1
    return Settings(
        query=text,
        tokens=tuple(tokens),
        mode=mode,
        style=style,
        seconds=min(180.0, max(8.0, length)),
        max_per_lecture=min(8, max(1, per_lecture)),
        seed=mix,
        course=(course or "").strip(),
    )


def ensure_index(
    timed_dir: Path = TIMED_DIR,
    db_path: Path = SHORTS_INDEX,
    *,
    reindex: bool = False,
    lemmatize=None,
    lemmatizer_name: str | None = None,
    on_progress=None,
) -> Catalog:
    with _INDEX_LOCK:
        lemmatize, lemmatizer_name = _resolve_lemmatizer(lemmatize, lemmatizer_name)
        signature = corpus_signature(timed_dir)
        if not reindex and _meta_matches(db_path, signature, lemmatizer_name):
            return Catalog(db_path, timed_dir, lemmatize, lemmatizer_name, signature)
        lg.info("Собираю индекс слов для шортсов")
        _build_index(timed_dir, db_path, signature, lemmatize, lemmatizer_name, on_progress)
        _PHRASE_CACHE.clear()
        lg.info("Индекс слов готов: {}", db_path.name)
        return Catalog(db_path, timed_dir, lemmatize, lemmatizer_name, signature)


def _resolve_lemmatizer(lemmatize, name: str | None):
    if lemmatize is not None:
        return lemmatize, name or "custom"
    function, detected = get_lemmatizer()
    return function, detected


def _meta_matches(db_path: Path, signature: str, lemmatizer_name: str) -> bool:
    if not db_path.is_file():
        return False
    try:
        con = sqlite3.connect(db_path)
        rows = dict(con.execute("SELECT key, value FROM meta"))
        con.close()
    except sqlite3.Error:
        return False
    return (
        rows.get("version") == str(INDEX_VERSION)
        and rows.get("signature") == signature
        and rows.get("lemmatizer") == lemmatizer_name
    )


def _build_index(timed_dir, db_path, signature, lemmatize, lemmatizer_name, on_progress) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".sqlite.tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    try:
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        _create_tables(con)
        builder = _IndexBuilder(con, lemmatize)
        paths = sorted(path for path in timed_dir.glob("*.mp4.txt") if path.is_file())
        total = len(paths)
        for index, path in enumerate(paths, start=1):
            builder.add_file(path)
            if on_progress and (index == total or index % 100 == 0):
                on_progress(index, total)
        builder.finish()
        con.execute("CREATE INDEX idx_hits_word ON hits(word_id)")
        con.execute("CREATE INDEX idx_lecture_counts_word ON lecture_counts(word_id)")
        con.execute("CREATE INDEX idx_words_lemma ON words(lemma)")
        con.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("version", str(INDEX_VERSION)),
                ("signature", signature),
                ("lemmatizer", lemmatizer_name),
            ],
        )
        con.commit()
    except Exception:
        con.close()
        if tmp.exists():
            tmp.unlink()
        raise
    con.close()
    os.replace(tmp, db_path)


def _create_tables(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            slug TEXT NOT NULL,
            title TEXT NOT NULL,
            course TEXT NOT NULL,
            course_fold TEXT NOT NULL,
            url TEXT NOT NULL
        );
        CREATE TABLE words (
            id INTEGER PRIMARY KEY,
            word TEXT NOT NULL UNIQUE,
            lemma TEXT NOT NULL,
            count INTEGER NOT NULL,
            lectures INTEGER NOT NULL
        );
        CREATE TABLE hits (
            word_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL
        );
        CREATE TABLE lecture_counts (
            word_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            count INTEGER NOT NULL
        );
        """
    )


class _IndexBuilder:
    def __init__(self, con: sqlite3.Connection, lemmatize):
        self.con = con
        self.lemmatize = lemmatize
        self.word_ids: dict[str, int] = {}
        self.counts: Counter[int] = Counter()
        self.lectures: Counter[int] = Counter()
        self.pending: list[tuple[int, int, int, int]] = []
        self.pending_counts: list[tuple[int, int, int]] = []
        self.file_id = 0

    def add_file(self, path: Path) -> None:
        raw = path.read_text(encoding="utf-8", errors="replace")
        header, _body = split_header(raw)
        slug = timed_slug(path)
        parsed = parse_stem(slug)
        title = header.get("Название") or parsed.display_title or slug
        course = parsed.display_course
        self.file_id += 1
        self.con.execute(
            "INSERT INTO files (id, slug, title, course, course_fold, url) VALUES (?, ?, ?, ?, ?, ?)",
            (self.file_id, slug, title, course, course.casefold(), header.get("Источник", "")),
        )
        spans: dict[int, list[tuple[int, int]]] = defaultdict(list)
        seen_in_file: set[int] = set()
        for segment in parse_segments(raw):
            word = cue_word(segment.text)
            if word is None:
                continue
            key = fold(word)
            if not key or len(key) > 40:
                continue
            wid = self._wid(key)
            self.counts[wid] += 1
            if wid not in seen_in_file:
                seen_in_file.add(wid)
                self.lectures[wid] += 1
            if not indexable(key):
                continue
            start_ms = int(round(segment.start * 1000))
            end_ms = int(round(max(segment.end, segment.start) * 1000))
            bucket = spans[wid]
            if bucket and bucket[-1][0] == start_ms:
                continue
            bucket.append((start_ms, end_ms))
        for wid, times in spans.items():
            self.pending_counts.append((wid, self.file_id, len(times)))
            for start_ms, end_ms in even_sample(times, SAMPLE_PER_LECTURE):
                self.pending.append((wid, self.file_id, start_ms, end_ms))
        if len(self.pending) >= 5000 or len(self.pending_counts) >= 5000:
            self._flush()

    def finish(self) -> None:
        self._flush()
        rows = []
        for word, wid in self.word_ids.items():
            rows.append((wid, word, _lemma(self.lemmatize, word), self.counts[wid], self.lectures[wid]))
        self.con.executemany(
            "INSERT INTO words (id, word, lemma, count, lectures) VALUES (?, ?, ?, ?, ?)",
            rows,
        )

    def _wid(self, word: str) -> int:
        found = self.word_ids.get(word)
        if found is None:
            found = len(self.word_ids) + 1
            self.word_ids[word] = found
        return found

    def _flush(self) -> None:
        if self.pending:
            self.con.executemany(
                "INSERT INTO hits (word_id, file_id, start_ms, end_ms) VALUES (?, ?, ?, ?)",
                self.pending,
            )
            self.pending.clear()
        if self.pending_counts:
            self.con.executemany(
                "INSERT INTO lecture_counts (word_id, file_id, count) VALUES (?, ?, ?)",
                self.pending_counts,
            )
            self.pending_counts.clear()


def _lemma(lemmatize, word: str) -> str:
    try:
        value = lemmatize(word)
    except Exception:  # noqa: BLE001 — кривое слово не должно ронять индекс
        return word
    text = fold(str(value or word))
    return text or word


def suggest(catalog: Catalog, prefix: str, limit: int = 12) -> list[dict]:
    token = ""
    tokens = query_tokens(prefix)
    if tokens:
        token = tokens[-1]
    con = sqlite3.connect(catalog.path)
    try:
        if not token:
            return _idea_rows(con, limit)
        escaped = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = con.execute(
            """
            SELECT word, count, lectures FROM words
            WHERE word LIKE ? ESCAPE '\\'
            ORDER BY count DESC
            LIMIT ?
            """,
            (escaped + "%", max(1, limit)),
        ).fetchall()
    finally:
        con.close()
    return [{"word": word, "count": count, "lectures": lectures} for word, count, lectures in rows]


def ideas(catalog: Catalog, limit: int = 18) -> list[dict]:
    con = sqlite3.connect(catalog.path)
    try:
        rows = _idea_rows(con, limit)
    finally:
        con.close()
    return rows


def _idea_rows(con: sqlite3.Connection, limit: int) -> list[dict]:
    rows = con.execute(
        """
        SELECT word, count, lectures FROM words
        WHERE count BETWEEN 30 AND 3500 AND lectures >= 2 AND length(word) >= 5
        ORDER BY count DESC
        LIMIT 80
        """
    ).fetchall()
    picked = []
    for word, count, lectures in rows:
        if word in STOPWORDS:
            continue
        picked.append({"word": word, "count": count, "lectures": lectures})
        if len(picked) >= limit:
            break
    return picked


def collect(catalog: Catalog, settings: Settings) -> WordSearch:
    label = overlay_label(settings.query)
    if len(settings.tokens) > 1:
        hits = _phrase_hits(catalog, settings)
        lectures = len({hit.slug for hit in hits})
        note = "Фраза ищется по всем лекциям подряд, без усечения выборки."
        mentions: dict[str, int] = {}
        for hit in hits:
            mentions[hit.slug] = mentions.get(hit.slug, 0) + 1
        return WordSearch(
            settings.query, label, settings.mode, len(hits), lectures, [], hits, True, note, mentions
        )
    return _collect_word(catalog, settings, label)


def _collect_word(catalog: Catalog, settings: Settings, label: str) -> WordSearch:
    token = settings.tokens[0]
    key = _lemma(catalog.lemmatize, token) if settings.mode == "forms" else token
    con = sqlite3.connect(catalog.path)
    try:
        forms = _forms(con, token, catalog)
        if settings.mode == "forms":
            total = con.execute("SELECT COALESCE(SUM(count), 0) FROM words WHERE lemma = ?", (key,)).fetchone()[0]
            hit_rows = con.execute(
                """
                SELECT f.slug, f.title, f.course, f.url, h.start_ms, h.end_ms, w.word
                FROM hits h
                JOIN files f ON f.id = h.file_id
                JOIN words w ON w.id = h.word_id
                WHERE w.lemma = ?
                ORDER BY f.slug, h.start_ms
                """,
                (key,),
            ).fetchall()
        else:
            total = con.execute("SELECT COALESCE(SUM(count), 0) FROM words WHERE word = ?", (token,)).fetchone()[0]
            hit_rows = con.execute(
                """
                SELECT f.slug, f.title, f.course, f.url, h.start_ms, h.end_ms, w.word
                FROM hits h
                JOIN files f ON f.id = h.file_id
                JOIN words w ON w.id = h.word_id
                WHERE w.word = ?
                ORDER BY f.slug, h.start_ms
                """,
                (token,),
            ).fetchall()
        mentions = _mention_map(con, settings.mode, key, token)
    finally:
        con.close()
    hits = [_hit_from_row(row) for row in hit_rows]
    if settings.course:
        needle = settings.course.casefold()
        hits = [hit for hit in hits if needle in hit.course.casefold()]
        mentions = {slug: count for slug, count in mentions.items() if slug in {hit.slug for hit in hits}}
    indexed = indexable(token) or (settings.mode == "forms" and indexable(key))
    shown_total = len(hits) if settings.course and indexed else int(total)
    note = _word_note(indexed, int(total), bool(settings.course))
    lectures = len({hit.slug for hit in hits})
    if not indexed:
        stored = _stored_lectures(catalog, token)
        lectures = stored
    return WordSearch(
        settings.query,
        label,
        settings.mode,
        shown_total,
        lectures,
        forms,
        hits if indexed else [],
        indexed,
        note,
        mentions if indexed else {},
    )


def _mention_map(con: sqlite3.Connection, mode: str, lemma: str, token: str) -> dict[str, int]:
    if mode == "forms":
        rows = con.execute(
            """
            SELECT f.slug, SUM(lc.count)
            FROM lecture_counts lc
            JOIN files f ON f.id = lc.file_id
            JOIN words w ON w.id = lc.word_id
            WHERE w.lemma = ?
            GROUP BY f.slug
            """,
            (lemma,),
        )
    else:
        rows = con.execute(
            """
            SELECT f.slug, lc.count
            FROM lecture_counts lc
            JOIN files f ON f.id = lc.file_id
            JOIN words w ON w.id = lc.word_id
            WHERE w.word = ?
            """,
            (token,),
        )
    return {slug: int(count) for slug, count in rows}


def _forms(con: sqlite3.Connection, token: str, catalog: Catalog) -> list[dict[str, int | str]]:
    lemma = _lemma(catalog.lemmatize, token)
    rows = con.execute(
        """
        SELECT word, count FROM words
        WHERE lemma = ?
        ORDER BY count DESC
        LIMIT 8
        """,
        (lemma,),
    ).fetchall()
    return [{"word": word, "count": count} for word, count in rows]


def _stored_lectures(catalog: Catalog, token: str) -> int:
    con = sqlite3.connect(catalog.path)
    try:
        row = con.execute("SELECT lectures FROM words WHERE word = ?", (token,)).fetchone()
    finally:
        con.close()
    if row is None:
        return 0
    return int(row[0])


def _hit_from_row(row) -> Hit:
    slug, title, course, url, start_ms, end_ms, word = row
    return Hit(slug, title, course, url, start_ms / 1000.0, end_ms / 1000.0, word)


def _word_note(indexed: bool, total: int, course: bool) -> str:
    if total <= 0:
        return "В расшифровках такого слова нет."
    if not indexed:
        return "Короткие и служебные слова в ролик не режутся: их слишком много, и шортс из них пустой."
    note = f"В индекс попадает до {SAMPLE_PER_LECTURE} произнесений на лекцию, равномерно по времени."
    if course:
        note += " Фильтр курса применяется к этой выборке."
    if total > 5000:
        note += " Слово частое: в ролик войдёт только часть мест."
    return note


def _phrase_hits(catalog: Catalog, settings: Settings) -> list[Hit]:
    key = (
        catalog.signature,
        settings.tokens,
        settings.mode,
        settings.course.casefold(),
    )
    with _PHRASE_LOCK:
        cached = _PHRASE_CACHE.get(key)
    if cached is not None:
        return cached
    lg.info("Ищу фразу «{}» по расшифровкам", settings.query)
    hits = _scan_phrase(catalog, settings)
    with _PHRASE_LOCK:
        if len(_PHRASE_CACHE) > 24:
            _PHRASE_CACHE.clear()
        _PHRASE_CACHE[key] = hits
    return hits


def _scan_phrase(catalog: Catalog, settings: Settings) -> list[Hit]:
    surface = settings.mode == "exact"
    compare = (lambda word: word) if surface else (lambda word: _lemma(catalog.lemmatize, word))
    needle = settings.course.casefold()
    label = " ".join(settings.tokens)
    hits: list[Hit] = []
    paths = sorted(path for path in catalog.timed_dir.glob("*.mp4.txt") if path.is_file())
    for path in paths:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if surface and not _file_has_tokens(raw, settings.tokens):
            continue
        header, _body = split_header(raw)
        slug = timed_slug(path)
        parsed = parse_stem(slug)
        course = parsed.display_course
        if needle and needle not in course.casefold():
            continue
        cues = _cues_from_segments(parse_segments(raw))
        spans = phrase_spans(cues, settings.tokens, compare)
        if not spans:
            continue
        title = header.get("Название") or parsed.display_title or slug
        url = header.get("Источник", "")
        for start, end in spans:
            hits.append(Hit(slug, title, course, url, start, end, label))
    return hits


def _file_has_tokens(raw: str, tokens: tuple[str, ...]) -> bool:
    folded = raw.casefold().replace("ё", "е")
    return all(token in folded for token in tokens)


def _cues_from_segments(segments) -> list[tuple[float, float, str]]:
    cues = []
    for segment in segments:
        word = cue_word(segment.text)
        if word:
            cues.append((segment.start, segment.end, fold(word)))
    return cues


def load_contexts(clips: list[Clip], timed_dir: Path, limit: int = 40) -> list[Clip]:
    cache: dict[str, list[tuple[float, float, str]]] = {}
    filled: list[Clip] = []
    for clip in clips[:limit]:
        cues = cache.get(clip.hit.slug)
        if cues is None:
            path = timed_dir / f"{clip.hit.slug}.mp4.txt"
            cues = _read_cues(path)
            cache[clip.hit.slug] = cues
        text = context_line(cues, clip.hit.start, clip.hit.end)
        filled.append(Clip(clip.hit, clip.start, clip.end, text))
    return filled


def _read_cues(path: Path) -> list[tuple[float, float, str]]:
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace")
    return _cues_from_segments(parse_segments(raw))


def context_line(cues: list[tuple[float, float, str]], start: float, end: float, radius: int = 6) -> str:
    if not cues:
        return ""
    first = 0
    for index, (_cue_start, cue_end, _word) in enumerate(cues):
        if cue_end >= start - 0.05:
            first = index
            break
    else:
        return ""
    last = first
    for index in range(first, len(cues)):
        if cues[index][0] <= end + 0.05:
            last = index
        else:
            break
    left = max(0, first - radius)
    right = min(len(cues), last + 1 + radius)
    return " ".join(word for _start, _end, word in cues[left:right])


def preview_payload(catalog: Catalog, settings: Settings, view: int = 1) -> dict:
    found = collect(catalog, settings)
    videos: list[list[Clip]] = []
    if found.indexed and found.hits:
        videos = plan_shorts(
            found.hits,
            style=STYLES[settings.style],
            target_seconds=settings.seconds,
            max_per_lecture=settings.max_per_lecture,
            seed=settings.seed,
            counts=found.mentions,
        )
    chosen = _pick_part(videos, view)
    shown = load_contexts(chosen, catalog.timed_dir, 40) if chosen else []
    cards = [_video_card(index, clips) for index, clips in enumerate(videos, start=1)]
    duration = cards[max(0, min(view, len(cards)) - 1)]["seconds"] if cards else 0.0
    return {
        "query": found.query,
        "label": found.label,
        "mode": found.mode,
        "total": found.total,
        "lectures": found.lectures,
        "forms": found.forms,
        "indexed": found.indexed,
        "note": found.note,
        "planned": len(chosen),
        "planned_seconds": duration,
        "parts": len(videos),
        "view": max(1, min(view, len(videos) or 1)),
        "seed": settings.seed,
        "videos": cards,
        "clips": [_public_clip(clip) for clip in shown],
        "hidden_clips": max(0, len(chosen) - len(shown)),
    }


def _pick_part(videos: list[list[Clip]], view: int) -> list[Clip]:
    if not videos:
        return []
    index = max(1, min(view, len(videos))) - 1
    return videos[index]


def _video_card(index: int, clips: list[Clip]) -> dict:
    titles: list[str] = []
    for clip in clips:
        if not titles or titles[-1] != clip.hit.title:
            titles.append(clip.hit.title)
    seconds = INTRO_SECONDS + sum(clip.duration for clip in clips)
    return {
        "part": index,
        "seconds": round(seconds, 1),
        "clips": len(clips),
        "lectures": len(titles),
        "titles": titles[:8],
    }


def _public_clip(clip: Clip) -> dict:
    hit = clip.hit
    return {
        "course": hit.course,
        "title": hit.title,
        "clock": format_clock_full(hit.start),
        "url": media_link(hit.url, hit.start),
        "context": clip.context,
        "word": hit.label,
    }


def media_link(url: str, seconds: float) -> str:
    if not url:
        return ""
    lowered = url.casefold()
    if "youtube.com" in lowered or "youtu.be" in lowered:
        joiner = "&" if "?" in url else "?"
        return f"{url}{joiner}t={max(int(seconds), 0)}"
    return vk_link(url, seconds)


def render_short(
    clips: list[Clip],
    *,
    label: str,
    meta_line: str,
    output: Path,
    cookies: str | None = None,
    source_dir: Path = SOURCE_DIR,
    progress: BuildProgress | None = None,
) -> Rendered:
    if not clips:
        raise RenderError("Нет фрагментов для нарезки")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RenderError("Нужен ffmpeg в PATH: с ним режется картинка и звук")
    font = find_font()
    if font is None:
        raise RenderError("Не нашёл шрифт с кириллицей (Arial или Segoe UI)")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f".{output.stem}-parts"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    label_path = work / "label.txt"
    meta_path = work / "meta.txt"
    label_path.write_text(_draw_text(label), encoding="utf-8")
    meta_path.write_text(_draw_text(meta_line), encoding="utf-8")
    tracker = progress or BuildProgress()
    try:
        sources = fetch_sources(clips, source_dir, cookies, progress=tracker)
        intro = work / "000-intro.mp4"
        tracker.begin_cut(len(clips))
        _render_intro(ffmpeg, intro, font, label_path, meta_path, label)
        parts = [intro]
        session = _MediaSession(ffprobe)
        skipped = 0
        for index, clip in enumerate(clips, start=1):
            media = sources.get(clip.hit.slug)
            tracker.tick_cut(index, clip.hit.title)
            if media is None:
                skipped += 1
                lg.warning("Нет файла: {}", clip.hit.title)
                continue
            dest = work / f"{index:03d}.mp4"
            try:
                _transcode(ffmpeg, media, dest, font, label_path, clip.start, clip.end, session)
            except RenderError as exc:
                skipped += 1
                lg.warning("Пропуск {} {}: {}", clip.hit.title, format_clock_full(clip.hit.start), exc)
                continue
            parts.append(dest)
        if len(parts) == 1:
            raise RenderError("Ни один фрагмент не скачался. Если VK просит вход, добавь --cookies cookies.txt")
        tracker.begin_assemble()
        _concat(ffmpeg, parts, output)
    except Exception:
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        raise
    shutil.rmtree(work, ignore_errors=True)
    tracker.finish()
    return Rendered(output, len(parts) - 1, skipped)


def _draw_text(text: str) -> str:
    return text.replace("%", "％").replace("\n", " ").strip()


def find_font() -> Path | None:
    env = os.environ.get("SHORTS_FONT", "").strip()
    candidates = [
        Path(env) if env else None,
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ]
    for path in candidates:
        if path is not None and path.is_file():
            return path
    return None


def _font_size(text: str) -> int:
    length = len(text)
    if length <= 10:
        return 86
    if length <= 18:
        return 64
    return 46


def _escape_filter_path(path: Path) -> str:
    return path.resolve().as_posix().replace("\\", "/").replace(":", "\\:")


def _render_intro(ffmpeg: str, dest: Path, font: Path, label: Path, meta: Path, text: str) -> None:
    kicker = dest.with_name("kicker.txt")
    kicker.write_text("каждый раз, когда лектор говорит", encoding="utf-8")
    font_esc = _escape_filter_path(font)
    label_esc = _escape_filter_path(label)
    meta_esc = _escape_filter_path(meta)
    kicker_esc = _escape_filter_path(kicker)
    size = _font_size(text)
    video_filter = (
        f"drawtext=fontfile='{font_esc}':textfile='{kicker_esc}':"
        "fontsize=42:fontcolor=0xD7E2EE:x=(w-text_w)/2:y=(h/2)-170,"
        f"drawtext=fontfile='{font_esc}':textfile='{label_esc}':fontsize={size}:"
        "fontcolor=0xFFE08A:borderw=8:bordercolor=0x10151c:x=(w-text_w)/2:y=(h-text_h)/2,"
        f"drawtext=fontfile='{font_esc}':textfile='{meta_esc}':"
        "fontsize=34:fontcolor=0x9FB3C8:x=(w-text_w)/2:y=(h/2)+140"
    )
    _run(
        [
            ffmpeg, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=0x10151c:s=1080x1920:r=30:d={INTRO_SECONDS}",
            "-f", "lavfi", "-t", f"{INTRO_SECONDS}", "-i", "anullsrc=r=48000:cl=stereo",
            "-vf", video_filter,
            "-map", "0:v:0", "-map", "1:a:0",
            *_ENCODE,
            str(dest),
        ]
    )


def _transcode(ffmpeg, source, dest, font, label, start, end, session: "_MediaSession") -> None:
    duration = None if end is None else max(0.2, end - start)
    try:
        if session.has_audio(str(source)):
            _encode(ffmpeg, source, dest, font, label, start, duration, audio=True)
            return
    except RenderError as exc:
        if "0:a" not in str(exc) and "audio" not in str(exc).lower():
            raise
    _encode(ffmpeg, source, dest, font, label, start, duration, audio=False)


def level_gain(mean_db: float | None) -> float:
    """Усиление, чтобы средняя громкость речи стала одной и той же."""
    if mean_db is None or mean_db <= -55:
        return 0.0
    return max(-24.0, min(18.0, TARGET_MEAN_DB - mean_db))


def _clip_gain(ffmpeg: str, source: str, start: float, duration: float | None) -> float:
    cmd = [ffmpeg, "-nostdin", "-hide_banner"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", source]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-af", "volumedetect", "-vn", "-f", "null", "-"]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        return 0.0
    match = _MEAN_VOLUME.search(completed.stderr or "")
    if match is None:
        return 0.0
    return level_gain(float(match.group(1)))


def _encode(ffmpeg, source, dest, font, label, start, duration, *, audio: bool) -> None:
    video = _vertical_filter(font, label, _font_size(label.read_text(encoding="utf-8")))
    cmd = [ffmpeg, "-y", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if str(source).startswith(("http://", "https://")):
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(source)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    if audio:
        gain = _clip_gain(ffmpeg, str(source), start or 0.0, duration)
        fade = max(0.0, (duration or 1.0) - 0.05)
        graph = (
            f"{video};[0:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"volume={gain:.2f}dB,alimiter=limit=0.89:attack=5:release=50:level=false,"
            f"afade=t=in:st=0:d=0.015,afade=t=out:st={fade:.3f}:d=0.04[a]"
        )
        cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]"]
    else:
        silence = f"{duration:.3f}" if duration is not None else "2"
        cmd += ["-f", "lavfi", "-t", silence, "-i", "anullsrc=r=48000:cl=stereo"]
        cmd += ["-filter_complex", video, "-map", "[v]", "-map", "1:a:0"]
    cmd += [*_ENCODE, "-shortest", str(dest)]
    _run(cmd)


def _vertical_filter(font: Path, label: Path, size: int) -> str:
    font_esc = _escape_filter_path(font)
    text_esc = _escape_filter_path(label)
    return (
        "[0:v]split=2[a][b];"
        "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=18:4[bg];"
        "[b]scale=1080:-2:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,fps=30,format=yuv420p,"
        f"drawtext=fontfile='{font_esc}':textfile='{text_esc}':fontsize={size}:"
        "fontcolor=white:borderw=7:bordercolor=black:x=(w-text_w)/2:y=h-text_h-150[v]"
    )


def _concat(ffmpeg: str, parts: list[Path], dest: Path) -> None:
    listing = dest.with_suffix(".concat.txt")
    lines = []
    for part in parts:
        path = part.resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{path}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        _run(
            [
                ffmpeg, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", "-movflags", "+faststart", str(dest),
            ]
        )
    finally:
        listing.unlink(missing_ok=True)


def _run(args: list[str]) -> None:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=240,
        )
    except subprocess.TimeoutExpired as exc:
        raise RenderError("ffmpeg думал слишком долго и был остановлен") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()
        raise RenderError(tail[-1800:] or f"ffmpeg завершился с кодом {completed.returncode}")


class _MediaSession:
    def __init__(self, ffprobe: str):
        self.ffprobe = ffprobe
        self._audio: dict[str, bool] = {}

    def has_audio(self, source: str) -> bool:
        cached = self._audio.get(source)
        if cached is not None:
            return cached
        try:
            completed = subprocess.run(
                [
                    self.ffprobe, "-v", "error", "-select_streams", "a",
                    "-show_entries", "stream=index", "-of", "csv=p=0", source,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=40,
            )
        except subprocess.TimeoutExpired:
            self._audio[source] = True
            return True
        found = bool((completed.stdout or "").strip()) or completed.returncode != 0
        self._audio[source] = found
        return found


class BuildProgress:
    """Общий прогресс сборки: скачивание, нарезка, склейка и оценка времени."""

    def __init__(self, on_change=None):
        self._lock = threading.Lock()
        self._on_change = on_change
        self._shown = -1
        self.started = time.monotonic()
        self.phase = "prepare"
        self.message = "Готовлю нарезку"
        self.percent = 0.0
        self.eta: int | None = None
        self.current = 0
        self.total = 0
        self.download_weight = 0.0
        self.files: dict[str, dict] = {}
        self.streams: dict[tuple[str, str], tuple] = {}
        self.cut_done = 0
        self.cut_total = 0

    def set_message(self, message: str) -> None:
        with self._lock:
            self.message = message
            self._emit()

    def begin_downloads(self, hits: list[Hit]) -> None:
        with self._lock:
            self.phase = "download"
            self.download_weight = 0.78 if hits else 0.0
            self.total = len(hits)
            self.current = 0
            self.files = {
                hit.slug: {
                    "title": hit.title,
                    "state": "wait",
                    "done": 0,
                    "total": None,
                    "unit": "",
                    "error": "",
                }
                for hit in hits
            }
            self.streams.clear()
            self._download_message()
            self._recompute()

    def mark_downloading(self, slug: str) -> None:
        with self._lock:
            item = self.files.get(slug)
            if item and item["state"] == "wait":
                item["state"] = "download"
            self._download_message()
            self._recompute()

    def clear_streams(self, slug: str) -> None:
        with self._lock:
            for key in [item for item in self.streams if item[0] == slug]:
                self.streams.pop(key, None)
            item = self.files.get(slug)
            if item and item["state"] != "error":
                item["done"] = 0
                item["total"] = None
                item["state"] = "download"

    def tick_download(
        self,
        slug: str,
        key: str,
        done: int,
        total: int | None,
        unit: str,
        finished: bool,
    ) -> None:
        with self._lock:
            item = self.files.get(slug)
            if item is None or item["state"] in {"done", "error"}:
                return
            self.streams[(slug, key)] = (done, total, unit, finished)
            rows = [row for (file_slug, _), row in self.streams.items() if file_slug == slug]
            item["done"] = sum(row[0] for row in rows)
            totals = [row[1] for row in rows]
            item["total"] = sum(totals) if totals and all(value is not None for value in totals) else None
            item["unit"] = unit
            item["state"] = "download"
            self._recompute()

    def finish_download(self, slug: str, ok: bool, error: str = "") -> None:
        with self._lock:
            item = self.files.get(slug)
            if item is None:
                return
            item["state"] = "done" if ok else "error"
            item["error"] = "" if ok else error
            if ok and item["total"]:
                item["done"] = item["total"]
            self.current = sum(1 for row in self.files.values() if row["state"] in {"done", "error"})
            self._download_message()
            self._recompute()

    def begin_cut(self, total: int) -> None:
        with self._lock:
            self.phase = "cut"
            self.cut_total = total
            self.cut_done = 0
            self.current = 0
            self.total = total
            self.message = "Нарезаю фрагменты"
            self._recompute()

    def tick_cut(self, done: int, title: str) -> None:
        with self._lock:
            self.cut_done = done
            self.current = done
            self.total = self.cut_total
            self.message = f"Нарезаю {title}"
            self._recompute()

    def begin_assemble(self) -> None:
        with self._lock:
            self.phase = "assemble"
            self.message = "Собираю ролик"
            self._recompute()

    def finish(self) -> None:
        with self._lock:
            self.phase = "done"
            self.message = "Готово"
            self.percent = 100.0
            self.eta = 0
            self._emit()

    def snapshot(self) -> dict:
        with self._lock:
            self._recompute()
            visible = []
            for item in self.files.values():
                if item["state"] == "done":
                    continue
                visible.append(
                    {
                        "title": item["title"],
                        "state": item["state"],
                        "detail": _item_detail(item),
                    }
                )
            visible.sort(key=lambda row: 0 if row["state"] == "download" else 1)
            return {
                "phase": self.phase,
                "message": self.message,
                "percent": self.percent,
                "eta_seconds": self.eta,
                "current": self.current,
                "total": self.total,
                "items": visible[:8],
            }

    def _download_message(self) -> None:
        done = sum(1 for item in self.files.values() if item["state"] in {"done", "error"})
        active = sum(1 for item in self.files.values() if item["state"] == "download")
        self.current = done
        self.message = f"Скачиваю лекции: готово {done} из {self.total}, сейчас {active}"

    def _recompute(self) -> None:
        if self.phase == "done":
            self.percent = 100.0
            self.eta = 0
            return
        fraction = self._fraction()
        self.percent = round(min(99.0, 100.0 * fraction), 1)
        elapsed = time.monotonic() - self.started
        if fraction >= 0.04 and elapsed >= 2:
            self.eta = int(elapsed * (1.0 - fraction) / fraction)
        else:
            self.eta = None
        self._emit()

    def _fraction(self) -> float:
        cut_span = max(0.0, 0.97 - self.download_weight)
        if self.phase == "download":
            return self.download_weight * self._download_fraction()
        if self.phase == "cut":
            cut = 0.0 if not self.cut_total else self.cut_done / self.cut_total
            return self.download_weight + cut_span * cut
        if self.phase == "assemble":
            return 0.98
        return 0.0

    def _download_fraction(self) -> float:
        if not self.files:
            return 1.0
        acc = 0.0
        for item in self.files.values():
            if item["state"] in {"done", "error"}:
                acc += 1.0
            elif item["total"]:
                acc += min(0.99, item["done"] / item["total"])
            elif item["state"] == "download":
                acc += 0.05
        return acc / len(self.files)

    def _emit(self) -> None:
        if self._on_change is None:
            return
        shown = int(self.percent)
        if shown == self._shown and self.phase != "assemble":
            return
        self._shown = shown
        self._on_change(self.message, self.percent, self.eta)


def _item_detail(item: dict) -> str:
    if item["state"] == "error":
        return item["error"] or "ошибка"
    if item["state"] == "wait":
        return "в очереди"
    if item["unit"] == "frag" and item["total"]:
        return f"{item['done']} / {item['total']} фрагментов"
    if item["unit"] == "B" and item["total"]:
        return f"{_size(item['done'])} / {_size(item['total'])}"
    if item["unit"] == "B" and item["done"]:
        return _size(item["done"])
    return "скачивается"


def _size(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f} ГБ"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.0f} МБ"
    if value >= 1_000:
        return f"{value / 1_000:.0f} КБ"
    return f"{value} Б"


def vk_url_from_id(video_id: str) -> str:
    text = (video_id or "").strip()
    if _VK_ID.fullmatch(text):
        return f"https://vk.com/video{text}"
    return ""


def page_url(url: str, video_id: str) -> str:
    text = (url or "").strip()
    if text.startswith(("http://", "https://")):
        return text
    return vk_url_from_id(video_id)


def lecturer_of(slug: str) -> str:
    return _lecturer_map().get(slug, "")


def _lecturer_map() -> dict[str, str]:
    global _LECTURERS
    with _URLS_LOCK:
        if _LECTURERS is not None:
            return _LECTURERS
        found: dict[str, str] = {}
        if LECTURES_CSV.is_file():
            with LECTURES_CSV.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle, delimiter=";"):
                    slug = (row.get("slug") or "").strip()
                    name = re.sub(r"\s+", " ", (row.get("lecturer") or "")).strip()
                    if slug and name:
                        found[slug] = name
        _LECTURERS = found
        return found


def lecture_page(slug: str) -> str:
    return _lecture_urls().get(slug, "")


def _lecture_urls() -> dict[str, str]:
    global _URLS
    with _URLS_LOCK:
        if _URLS is not None:
            return _URLS
        found: dict[str, str] = {}
        if LECTURES_CSV.is_file():
            with LECTURES_CSV.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle, delimiter=";"):
                    slug = (row.get("slug") or "").strip()
                    url = page_url(row.get("url") or "", row.get("video_id") or "")
                    if slug and url:
                        found[slug] = url
        _URLS = found
        return found


def _transient(exc: BaseException) -> bool:
    text = str(exc).casefold()
    needles = ("timed out", "timeout", "connection", "reset", "429", "503", "temporarily", "transport")
    return any(needle in text for needle in needles)


def _short_error(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else "ошибка скачивания"
    lowered = text.casefold()
    if "timed out" in lowered or "timeout" in lowered:
        return "VK не ответил вовремя, повтор тоже не помог"
    if "unable to download json" in lowered:
        return "VK не отдал страницу видео"
    if len(text) > 180:
        return text[:180]
    return text


def local_video(slug: str, video_dir: Path) -> Path | None:
    if not video_dir.is_dir():
        return None
    suffixes = {".mp4", ".mkv", ".webm", ".m4v"}
    for path in video_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stat().st_size <= 0:
            continue
        if path.stem == slug or path.stem.endswith("_" + slug):
            return path
    return None


def fetch_sources(
    clips: list[Clip],
    folder: Path,
    cookies: str | None,
    progress: BuildProgress | None = None,
    download=None,
) -> dict[str, Path]:
    """Качает недостающие лекции пачкой. Уже лежащий файл не трогает."""
    ready: dict[str, Path] = {}
    pending: list[Hit] = []
    seen: set[str] = set()
    for clip in clips:
        hit = _with_page(clip.hit)
        if hit.slug in seen:
            continue
        seen.add(hit.slug)
        local = local_video(hit.slug, folder)
        if local is not None:
            ready[hit.slug] = local
            continue
        if not hit.url:
            lg.warning("Нет ссылки, пропуск: {}", hit.title)
            continue
        pending.append(hit)
    if not pending:
        return ready
    if progress:
        progress.begin_downloads(pending)
    workers = max(1, min(len(pending), DEFAULT_DOWNLOAD_WORKERS))
    lg.info("Скачиваю {} роликов, одновременно {}, лучшее качество", len(pending), workers)

    def one(hit: Hit) -> tuple[str, Path | None]:
        if progress:
            progress.mark_downloading(hit.slug)
        try:
            if download is None:
                path = download_best(hit, folder, cookies, progress=progress)
            else:
                path = download(hit, folder, cookies)
        except Exception as exc:  # noqa: BLE001
            lg.warning("Не скачался {}: {}", hit.title, _short_error(exc))
            if progress:
                progress.finish_download(hit.slug, False, _short_error(exc))
            return hit.slug, None
        if progress:
            progress.finish_download(hit.slug, True)
        return hit.slug, path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for slug, path in pool.map(one, pending):
            if path is not None:
                ready[slug] = path
    return ready


def _with_page(hit: Hit) -> Hit:
    if hit.url:
        return hit
    page = lecture_page(hit.slug)
    if not page:
        return hit
    return Hit(hit.slug, hit.title, hit.course, page, hit.start, hit.end, hit.label)


def download_best(
    hit: Hit,
    folder: Path,
    cookies: str | None,
    progress: BuildProgress | None = None,
) -> Path:
    last: Exception | None = None
    for attempt in range(3):
        if progress:
            progress.clear_streams(hit.slug)
        try:
            return _download_once(hit, folder, cookies, progress)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt == 2 or not _transient(exc):
                break
            time.sleep(1.5 * (attempt + 1))
            lg.info("Повтор {} для {}", attempt + 2, hit.title)
    raise RenderError(_short_error(last or RuntimeError("ошибка скачивания")))


def _download_once(
    hit: Hit,
    folder: Path,
    cookies: str | None,
    progress: BuildProgress | None,
) -> Path:
    import yt_dlp

    folder.mkdir(parents=True, exist_ok=True)

    def hook(data: dict) -> None:
        if progress is None:
            return
        tick = parse_download_hook(data)
        if tick is None:
            return
        progress.tick_download(hit.slug, tick.key, tick.done, tick.total, tick.unit, tick.finished)

    options = {
        "format": BEST_FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(folder / f"{hit.slug}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 2,
        "socket_timeout": 45,
        "windowsfilenames": True,
        "cachedir": False,
        "noprogress": True,
        "overwrites": True,
        "continuedl": True,
        "concurrent_fragment_downloads": max(1, min(4, DEFAULT_FRAGMENT_THREADS)),
        "progress_hooks": [hook],
        "logger": _SilentLog(),
    }
    if cookies:
        options["cookiefile"] = cookies
    try:
        with yt_dlp.YoutubeDL(options) as client:
            client.download([hit.url])
    except Exception as exc:  # noqa: BLE001
        raise RenderError(str(exc)) from exc
    found = local_video(hit.slug, folder)
    if found is None:
        raise RenderError(f"{hit.title}: файл не появился")
    return found


class _SilentLog:
    def debug(self, message: str) -> None:
        return

    def info(self, message: str) -> None:
        return

    def warning(self, message: str) -> None:
        return

    def error(self, message: str) -> None:
        return


def _eta_text(seconds: int) -> str:
    if seconds < 60:
        return "меньше минуты"
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return f"{minutes} мин"
    hours, rest = divmod(minutes, 60)
    if rest:
        return f"{hours} ч {rest} мин"
    return f"{hours} ч"


def planned_meta(clips: list[Clip]) -> str:
    pieces = f"{len(clips)} фрагментов"
    lectures = len({clip.hit.slug for clip in clips})
    if lectures:
        pieces += f" · {lectures} лекций"
    return pieces


def load_popular_words(path: Path | None = None) -> list[str]:
    file = path or WORDCLOUD_JSON
    payload = json.loads(file.read_text(encoding="utf-8"))
    words: list[str] = []
    for item in payload.get("words") or []:
        text = str(item.get("text") or "").strip()
        if text:
            words.append(text)
    return words


def slugs_of(videos: list[list[Clip]]) -> set[str]:
    return {clip.hit.slug for block in videos for clip in block}


def disk_free(path: Path) -> int:
    probe = path if path.exists() else path.parent
    return shutil.disk_usage(probe).free


def ensure_space(
    folder: Path,
    keep: set[str],
    reserve: int,
    *,
    free=None,
) -> list[Path]:
    """Если места мало, удаляет исходники, которых нет в keep. Иначе не трогает."""
    if not folder.is_dir():
        return []
    free_now = free or (lambda: disk_free(folder))
    if free_now() >= reserve:
        return []
    candidates = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mkv", ".webm", ".m4v"}:
            continue
        if path.stem in keep:
            continue
        candidates.append(path)
    candidates.sort(key=lambda item: item.stat().st_size, reverse=True)
    removed: list[Path] = []
    for path in candidates:
        if free_now() >= reserve:
            break
        lg.info("Мало места, удаляю исходник {}", path.name)
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def run_shorts(args) -> int:
    if args.batch:
        catalog = ensure_index(reindex=args.reindex, on_progress=_index_progress)
        return run_popular_batch(
            catalog,
            seconds=args.seconds,
            mode=args.mode,
            style=args.style,
            max_per_lecture=args.max_per_lecture,
            seed=args.seed,
            cookies=args.cookies,
            reserve_gb=args.reserve_gb,
        )
    if not args.word:
        serve_shorts(args.port, reindex=args.reindex, cookies=args.cookies)
        return 0
    try:
        settings = make_settings(
            args.word,
            mode=args.mode,
            style=args.style,
            seconds=args.seconds,
            max_per_lecture=args.max_per_lecture,
            seed=args.seed,
            course=args.course,
        )
    except ValueError as exc:
        lg.error("{}", exc)
        return 1
    catalog = ensure_index(reindex=args.reindex, on_progress=_index_progress)
    if args.preview:
        return _print_preview(catalog, settings)
    part = 0 if args.all else args.part
    return _render_cli(catalog, settings, args.output, args.cookies, part)


def run_popular_batch(
    catalog: Catalog,
    *,
    seconds: float,
    mode: str,
    style: str,
    max_per_lecture: int,
    seed: int,
    cookies: str | None,
    reserve_gb: float,
) -> int:
    words = load_popular_words()
    if not words:
        lg.error("В облаке слов пусто")
        return 1
    reserve = int(reserve_gb * 1024**3)
    planned: list[tuple[str, list[list[Clip]]]] = []
    for word in words:
        try:
            settings = make_settings(
                word,
                mode=mode,
                style=style,
                seconds=seconds,
                max_per_lecture=max_per_lecture,
                seed=seed,
            )
        except ValueError as exc:
            lg.warning("Пропуск «{}»: {}", word, exc)
            continue
        found = collect(catalog, settings)
        if not found.indexed or not found.hits:
            lg.info("«{}»: нечего резать", word)
            continue
        videos = plan_shorts(
            found.hits,
            style=STYLES[settings.style],
            target_seconds=settings.seconds,
            max_per_lecture=settings.max_per_lecture,
            seed=settings.seed,
            counts=found.mentions,
        )
        if videos:
            planned.append((word, videos))
    lectures = set().union(*(slugs_of(videos) for _, videos in planned)) if planned else set()
    parts = sum(len(videos) for _, videos in planned)
    lg.info(
        "Популярных слов с роликами: {}, роликов {}, разных лекций {}",
        len(planned),
        parts,
        len(lectures),
    )
    POPULAR_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    done = 0
    for index, (word, videos) in enumerate(planned):
        keep = set().union(*(slugs_of(later) for _, later in planned[index:]))
        removed = ensure_space(SOURCE_DIR, keep, reserve)
        if disk_free(SOURCE_DIR) < reserve:
            removed += ensure_space(SOURCE_DIR, slugs_of(videos), reserve)
        if removed:
            lg.info(
                "Удалено исходников: {}, свободно {:.0f} ГБ",
                len(removed),
                disk_free(SOURCE_DIR) / 1024**3,
            )
        safe = sanitize_title(word) or "word"
        lg.info("Слово {}/{} «{}», роликов {}", index + 1, len(planned), word, len(videos))
        for part, clips in enumerate(videos, start=1):
            dest = POPULAR_DIR / f"{safe}-{part}.mp4"
            if dest.is_file() and dest.stat().st_size > 0:
                lg.info("Уже есть {}", dest.name)
                continue
            try:
                rendered = render_short(
                    clips,
                    label=overlay_label(word),
                    meta_line=planned_meta(clips),
                    output=dest,
                    cookies=cookies,
                )
            except RenderError as exc:
                lg.error("«{}» ролик {}: {}", word, part, exc)
                continue
            lg.info("Готово {}", rendered.path)
            done += 1
        still = set().union(*(slugs_of(later) for _, later in planned[index + 1 :]))
        ensure_space(SOURCE_DIR, still, reserve)
    lg.info("Собрано новых роликов: {}", done)
    return 0


def _index_progress(done: int, total: int) -> None:
    lg.info("Индекс слов: {} / {}", done, total)


def _print_preview(catalog: Catalog, settings: Settings) -> int:
    payload = preview_payload(catalog, settings)
    print(
        f"{payload['query']}: {payload['total']} раз, {payload['lectures']} лекций, "
        f"роликов {payload['parts']} по ~{settings.seconds:.0f} с"
    )
    print("Эпизоды одной лекции идут подряд.")
    if payload["note"]:
        print(payload["note"])
    for video in payload["videos"]:
        names = ", ".join(video["titles"])
        print(f"  {video['part']}. {video['seconds']} с, {video['lectures']} лекций — {names}")
    for form in payload["forms"]:
        print(f"  {form['word']}\t{form['count']}")
    print(f"Ролик {payload['view']}:")
    for clip in payload["clips"]:
        print(f"{clip['clock']}  {clip['course']} — {clip['title']}")
        if clip["context"]:
            print(f"    {clip['context']}")
        if clip["url"]:
            print(f"    {clip['url']}")
    hidden = payload["hidden_clips"]
    if hidden:
        print(f"… и ещё {hidden} фрагментов")
    return 0


def _part_number(value, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def choose_parts(videos: list[list[Clip]], part: int) -> list[list[Clip]]:
    if not videos:
        return []
    if part <= 0:
        return videos
    index = min(part, len(videos)) - 1
    return [videos[index]]


def _part_paths(dest: Path, count: int) -> list[Path]:
    if count <= 1:
        return [dest]
    return [dest.with_name(f"{dest.stem}-{index}{dest.suffix}") for index in range(1, count + 1)]


def _render_cli(
    catalog: Catalog,
    settings: Settings,
    output: Path | None,
    cookies: str | None,
    part: int,
) -> int:
    found = collect(catalog, settings)
    if not found.indexed or not found.hits:
        lg.error("{}", found.note or "Нечего резать")
        return 1
    videos = plan_shorts(
        found.hits,
        style=STYLES[settings.style],
        target_seconds=settings.seconds,
        max_per_lecture=settings.max_per_lecture,
        seed=settings.seed,
        counts=found.mentions,
    )
    chosen = choose_parts(videos, part)
    if not chosen:
        lg.error("Под эту длину не попало ни одного фрагмента")
        return 1
    safe = sanitize_title(settings.tokens[0]) or "shorts"
    dest = output or (SHORTS_DIR / f"{safe}.mp4")
    paths = _part_paths(dest, len(chosen))
    lg.info("Роликов: {} из {}", len(chosen), len(videos))

    def changed(message: str, percent: float, eta: int | None) -> None:
        tail = f", ещё ~{_eta_text(eta)}" if eta else ""
        lg.info("{:.0f}% {}{}", percent, message, tail)

    try:
        for index, clips in enumerate(chosen):
            lg.info("Ролик {} из {}", index + 1, len(chosen))
            rendered = render_short(
                clips,
                label=found.label,
                meta_line=planned_meta(clips),
                output=paths[index],
                cookies=cookies,
                progress=BuildProgress(on_change=changed),
            )
            print(rendered.path)
    except RenderError as exc:
        lg.error("{}", exc)
        return 1
    return 0


class _Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.status = "running"
        self.file: str | None = None
        self.files: list[str] = []
        self.error: str | None = None
        self.progress = BuildProgress()

    def as_dict(self) -> dict:
        data = self.progress.snapshot()
        data["id"] = self.id
        data["status"] = self.status
        data["file"] = self.files[0] if self.files else self.file
        data["files"] = self.files
        data["error"] = self.error
        if self.status == "error" and self.error:
            data["message"] = self.error
        return data


class Studio:
    def __init__(self, reindex: bool = False, cookies: str | None = None):
        self.reindex = reindex
        self.cookies = cookies
        self.state = "building"
        self.message = "Собираю индекс слов"
        self.index_done = 0
        self.index_total = 0
        self.index_started = time.monotonic()
        self.catalog: Catalog | None = None
        self.jobs: dict[str, _Job] = {}
        self._render_lock = threading.Lock()

    def start(self) -> None:
        threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self) -> None:
        def progress(done: int, total: int) -> None:
            self.index_done = done
            self.index_total = total
            self.message = f"Индекс слов: {done} из {total}"

        try:
            self.catalog = ensure_index(reindex=self.reindex, on_progress=progress)
        except Exception as exc:  # noqa: BLE001
            self.state = "error"
            self.message = str(exc)
            lg.exception("Индекс шортсов не собрался")
            return
        self.state = "ready"
        self.message = "Можно выбирать слово"

    def status(self) -> dict:
        lectures = 0
        words = 0
        if self.catalog is not None and self.catalog.path.is_file():
            con = sqlite3.connect(self.catalog.path)
            try:
                lectures = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                words = con.execute("SELECT COUNT(*) FROM words").fetchone()[0]
            finally:
                con.close()
        eta = None
        if self.state == "building" and self.index_total and self.index_done:
            elapsed = time.monotonic() - self.index_started
            fraction = self.index_done / self.index_total
            if fraction > 0 and elapsed >= 2:
                eta = int(elapsed * (1.0 - fraction) / fraction)
        return {
            "state": self.state,
            "message": self.message,
            "lectures": lectures,
            "words": words,
            "ffmpeg": bool(shutil.which("ffmpeg")),
            "cookies": bool(self.cookies),
            "index_done": self.index_done,
            "index_total": self.index_total,
            "eta_seconds": eta,
        }

    def submit(self, payload: dict) -> _Job:
        if self.catalog is None or self.state != "ready":
            raise RuntimeError(self.message or "Индекс ещё собирается")
        settings = make_settings(
            str(payload.get("q") or ""),
            mode=str(payload.get("mode") or "forms"),
            style=str(payload.get("style") or "tight"),
            seconds=payload.get("seconds") or 45,
            max_per_lecture=payload.get("max_per_lecture") or 3,
            seed=payload.get("seed") or 1,
            course=str(payload.get("course") or ""),
        )
        part = _part_number(payload.get("part"), default=1)
        with self._render_lock:
            if any(job.status == "running" for job in self.jobs.values()):
                raise BusyError("Уже собирается другой ролик")
            job = _Job(os.urandom(4).hex())
            self.jobs[job.id] = job
        threading.Thread(target=self._render, args=(job, settings, part), daemon=True).start()
        return job

    def _render(self, job: _Job, settings: Settings, part: int) -> None:
        assert self.catalog is not None
        try:
            job.progress.set_message("Ищу фрагменты")
            found = collect(self.catalog, settings)
            if not found.indexed or not found.hits:
                raise RenderError(found.note or "Нечего резать")
            videos = plan_shorts(
                found.hits,
                style=STYLES[settings.style],
                target_seconds=settings.seconds,
                max_per_lecture=settings.max_per_lecture,
                seed=settings.seed,
                counts=found.mentions,
            )
            chosen = choose_parts(videos, part)
            if not chosen:
                raise RenderError("Под эту длину не попало ни одного фрагмента")
            dest = SHORTS_DIR / f"{sanitize_title(settings.tokens[0]) or 'shorts'}-{job.id}.mp4"
            paths = _part_paths(dest, len(chosen))
            ready: list[str] = []
            for index, clips in enumerate(chosen):
                job.progress.set_message(f"Ролик {index + 1} из {len(chosen)}")
                rendered = render_short(
                    clips,
                    label=found.label,
                    meta_line=planned_meta(clips),
                    output=paths[index],
                    cookies=self.cookies,
                    progress=job.progress,
                )
                ready.append(rendered.path.name)
                job.files = list(ready)
        except Exception as exc:  # noqa: BLE001
            job.status = "error"
            job.error = str(exc)
            job.message = "Не собралось"
            lg.error("Шортс не собрался: {}", exc)
            return
        job.status = "done"
        job.files = ready
        job.file = ready[0]
        job.progress.finish()


def serve_shorts(port: int = 8770, reindex: bool = False, cookies: str | None = None) -> None:
    studio = Studio(reindex=reindex, cookies=cookies)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._bytes(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            params = {key: values[0] for key, values in parse_qs(parsed.query).items() if values}
            try:
                if parsed.path == "/api/status":
                    self._json(studio.status())
                    return
                if parsed.path == "/api/ideas":
                    self._need_catalog()
                    self._json({"words": ideas(studio.catalog, 18)})
                    return
                if parsed.path == "/api/suggest":
                    self._need_catalog()
                    self._json({"words": suggest(studio.catalog, params.get("q", ""), 12)})
                    return
                if parsed.path == "/api/preview":
                    self._need_catalog()
                    settings = _settings_from_params(params)
                    view = _part_number(params.get("view") or params.get("part") or 1)
                    self._json(preview_payload(studio.catalog, settings, view=max(view, 1)))
                    return
                if parsed.path == "/api/job":
                    job = studio.jobs.get(params.get("id", ""))
                    if job is None:
                        self._json({"error": "нет такой сборки"}, 404)
                        return
                    self._json(job.as_dict())
                    return
                if parsed.path.startswith("/media/"):
                    self._media(parsed.path[len("/media/"):], download="download" in params)
                    return
            except BusyError as exc:
                self._json({"error": str(exc)}, 409)
                return
            except (ValueError, RuntimeError, RenderError) as exc:
                self._json({"error": str(exc)}, 400)
                return
            self._bytes(404, b"not found", "text/plain; charset=utf-8")

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/render":
                self._bytes(404, b"not found", "text/plain; charset=utf-8")
                return
            try:
                payload = _read_json(self)
                job = studio.submit(payload)
            except BusyError as exc:
                self._json({"error": str(exc)}, 409)
                return
            except (ValueError, RuntimeError, RenderError) as exc:
                self._json({"error": str(exc)}, 400)
                return
            self._json(job.as_dict())

        def log_message(self, fmt, *args):
            return

        def _need_catalog(self) -> None:
            if studio.catalog is None or studio.state != "ready":
                raise RuntimeError(studio.message or "Индекс ещё собирается")

        def _media(self, name: str, download: bool) -> None:
            safe = Path(name).name
            path = (SHORTS_DIR / safe).resolve()
            if path.parent != SHORTS_DIR.resolve() or not path.is_file():
                self._bytes(404, "ролика ещё нет\n".encode("utf-8"), "text/plain; charset=utf-8")
                return
            _send_file(self, path, download)

        def _json(self, payload: dict, status: int = 200) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._bytes(status, raw, "application/json; charset=utf-8")

        def _bytes(self, status: int, raw: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

    studio.start()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"http://127.0.0.1:{port}")
    lg.info("Шортс открыт на http://127.0.0.1:{}", port)
    server.serve_forever()


def _settings_from_params(params: dict) -> Settings:
    return make_settings(
        params.get("q", ""),
        mode=params.get("mode", "forms"),
        style=params.get("style", "tight"),
        seconds=params.get("seconds", 45),
        max_per_lecture=params.get("max_per_lecture", 3),
        seed=params.get("seed", 1),
        course=params.get("course"),
    )


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or "0")
    if length <= 0 or length > 100_000:
        raise ValueError("Пустой запрос")
    raw = handler.rfile.read(length)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Не JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Не JSON")
    return payload


def _send_file(handler: BaseHTTPRequestHandler, path: Path, download: bool) -> None:
    size = path.stat().st_size
    start, end = _byte_range(handler.headers.get("Range"), size)
    status = 206 if handler.headers.get("Range") else 200
    length = end - start + 1
    handler.send_response(status)
    handler.send_header("Content-Type", "video/mp4")
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(length))
    if status == 206:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    if download:
        handler.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(path.name)}")
    handler.end_headers()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            chunk = handle.read(min(256 * 1024, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)


def _byte_range(header: str | None, size: int) -> tuple[int, int]:
    if not header or not header.startswith("bytes="):
        return 0, size - 1
    spec = header.removeprefix("bytes=").split(",", 1)[0]
    start_text, _, end_text = spec.partition("-")
    try:
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else size - 1
    except ValueError:
        return 0, size - 1
    start = min(max(start, 0), size - 1)
    end = min(max(end, start), size - 1)
    return start, end


_PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Шортс из лекций ФПМИ</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #10151c; color: #f7f4ea;
    font: 16px/1.45 "Segoe UI", sans-serif; }
  main { max-width: 880px; margin: 0 auto; padding: 28px 16px 80px; }
  h1 { font-size: 32px; margin: 0 0 8px; }
  .lead, .note, .status { color: #9fb3c8; }
  form, .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  input, select, button { background: #182230; color: #f7f4ea;
    border: 1px solid #31445c; border-radius: 8px; padding: 10px 12px; font: inherit; }
  input[type="text"] { flex: 1; min-width: 180px; }
  button { background: #2a6f97; color: white; cursor: pointer; }
  button.gold { background: #d4a017; color: #1a1406; font-weight: 650; }
  button:disabled { opacity: 0.5; cursor: default; }
  #ideas, #suggest { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
  .chip { border: 1px solid #31445c; background: #182230; color: #f7f4ea;
    border-radius: 999px; padding: 6px 10px; cursor: pointer; }
  .chip b { color: #ffe08a; font-weight: 650; }
  .chip.gold { background: #d4a017; color: #1a1406; }
  .chip.gold b { color: #1a1406; }
  #videos { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
  section { margin-top: 18px; padding: 14px; border: 1px solid #243140; border-radius: 12px; }
  article { padding: 10px 0; border-top: 1px solid #243140; }
  .meta { color: #8ecaff; }
  video { width: min(360px, 100%); margin-top: 16px; border-radius: 12px; background: black; }
  a { color: #8ecaff; }
  .track { height: 12px; margin-top: 10px; background: #243140; border-radius: 999px; overflow: hidden; }
  .track > div { height: 100%; width: 0; background: #d4a017; transition: width 0.3s linear; }
  #progress { margin-top: 16px; }
  #barItems { margin: 8px 0 0; padding-left: 18px; color: #9fb3c8; }
  #barItems li { margin: 4px 0; }
</style>
</head>
<body>
<main>
  <h1>Шортс из лекций</h1>
  <p class="lead">Выбери слово. Ролик соберётся из кусков, где лектор его говорит: вертикальный кадр, слово на экране, склейка под YouTube Shorts.</p>
  <p class="status" id="status">Собираю индекс слов…</p>
  <div class="track" id="indexWrap" hidden><div id="indexFill"></div></div>
  <form id="find">
    <input id="q" type="text" placeholder="кузнечик" autocomplete="off" required/>
    <button type="submit">Показать</button>
  </form>
  <div id="suggest"></div>
  <div id="ideas"></div>
  <section id="setup" hidden>
    <div class="row">
      <input id="course" type="text" placeholder="курс, если нужен"/>
      <select id="mode">
        <option value="forms">все формы слова</option>
        <option value="exact">как написано</option>
      </select>
      <select id="style">
        <option value="tight">только слово</option>
        <option value="phrase">с короткой фразой</option>
      </select>
      <select id="seconds">
        <option>20</option>
        <option>35</option>
        <option selected>45</option>
        <option>60</option>
      </select>
      <select id="per" title="Сколько кусков брать из одной лекции">
        <option value="1">1 с лекции</option>
        <option value="2">2 с лекции</option>
        <option value="3" selected>3 с лекции</option>
        <option value="4">4 с лекции</option>
        <option value="6">6 с лекции</option>
      </select>
    </div>
    <div class="row" style="margin-top:10px">
      <button type="button" id="again">Другая нарезка</button>
      <button type="button" class="gold" id="render" disabled>Собрать шортс</button>
      <button type="button" id="renderAll" hidden>Собрать все</button>
    </div>
  </section>
  <p class="note" id="summary"></p>
  <div id="videos"></div>
  <div id="clips"></div>
  <div id="progress" hidden>
    <div class="track"><div id="barFill"></div></div>
    <p class="note" id="job"></p>
    <ul id="barItems"></ul>
  </div>
  <video id="player" controls playsinline hidden></video>
  <p class="note">Лекции качаются в лучшем качестве, по несколько сразу: VK обрывает связь, если открыть десятки роликов в одну секунду. Сорвавшееся скачивание повторяется. Нужен ffmpeg. Если VK просит вход, запусти с <code>--cookies cookies.txt</code>.</p>
</main>
<script>
const statusEl = document.getElementById("status");
const suggestEl = document.getElementById("suggest");
const ideasEl = document.getElementById("ideas");
const setupEl = document.getElementById("setup");
const summaryEl = document.getElementById("summary");
const clipsEl = document.getElementById("clips");
const jobEl = document.getElementById("job");
const progressEl = document.getElementById("progress");
const barFill = document.getElementById("barFill");
const barItems = document.getElementById("barItems");
const indexWrap = document.getElementById("indexWrap");
const indexFill = document.getElementById("indexFill");
const player = document.getElementById("player");
const renderBtn = document.getElementById("render");
const renderAllBtn = document.getElementById("renderAll");
const videosEl = document.getElementById("videos");
let seed = 1;
let view = 1;
let ready = false;

function escapeHtml(value) {
  return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function params() {
  const data = new URLSearchParams();
  data.set("q", document.getElementById("q").value.trim());
  data.set("course", document.getElementById("course").value.trim());
  data.set("mode", document.getElementById("mode").value);
  data.set("style", document.getElementById("style").value);
  data.set("seconds", document.getElementById("seconds").value);
  data.set("max_per_lecture", document.getElementById("per").value);
  data.set("seed", String(seed));
  data.set("view", String(view));
  data.set("part", String(view));
  return data;
}
function formatEta(seconds) {
  if (seconds == null || !isFinite(seconds) || seconds < 0) return "";
  if (seconds < 45) return "осталось меньше минуты";
  const minutes = Math.max(1, Math.round(seconds / 60));
  if (minutes < 60) return "осталось около " + minutes + " мин";
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  if (!rest) return "осталось около " + hours + " ч";
  return "осталось около " + hours + " ч " + rest + " мин";
}
function showBar(fill, percent) {
  fill.style.width = Math.max(0, Math.min(100, percent || 0)) + "%";
}
async function refreshStatus() {
  const response = await fetch("/api/status");
  const payload = await response.json();
  let text = payload.message || "";
  const eta = formatEta(payload.eta_seconds);
  if (eta && payload.state === "building") text += " · " + eta;
  if (payload.lectures) text += " · лекций " + payload.lectures + ", слов " + payload.words;
  if (!payload.ffmpeg) text += " · нет ffmpeg";
  statusEl.textContent = text;
  if (payload.state === "building" && payload.index_total) {
    indexWrap.hidden = false;
    showBar(indexFill, 100 * payload.index_done / payload.index_total);
  } else {
    indexWrap.hidden = true;
  }
  if (payload.state === "ready" && !ready) {
    ready = true;
    loadIdeas();
  }
  if (payload.state === "building") setTimeout(refreshStatus, 800);
}
async function loadIdeas() {
  const response = await fetch("/api/ideas");
  const payload = await response.json();
  ideasEl.innerHTML = (payload.words || []).map(chip).join("");
}
function chip(word) {
  return '<button type="button" class="chip" data-word="' + escapeHtml(word.word) +
    '"><b>' + escapeHtml(word.word) + '</b> ' + word.count + '</button>';
}
ideasEl.addEventListener("click", (event) => {
  const button = event.target.closest("[data-word]");
  if (!button) return;
  document.getElementById("q").value = button.dataset.word;
  seed = 1;
  view = 1;
  loadPreview();
});
let suggestTimer = 0;
document.getElementById("q").addEventListener("input", () => {
  clearTimeout(suggestTimer);
  suggestTimer = setTimeout(loadSuggest, 180);
});
async function loadSuggest() {
  if (!ready) return;
  const q = document.getElementById("q").value.trim();
  if (q.length < 2) { suggestEl.innerHTML = ""; return; }
  const response = await fetch("/api/suggest?q=" + encodeURIComponent(q));
  const payload = await response.json();
  suggestEl.innerHTML = (payload.words || []).map(chip).join("");
}
suggestEl.addEventListener("click", (event) => {
  const button = event.target.closest("[data-word]");
  if (!button) return;
  document.getElementById("q").value = button.dataset.word;
  suggestEl.innerHTML = "";
  seed = 1;
  view = 1;
  loadPreview();
});
document.getElementById("find").addEventListener("submit", (event) => {
  event.preventDefault();
  seed = 1;
  view = 1;
  loadPreview();
});
for (const id of ["course", "mode", "style", "seconds", "per"]) {
  document.getElementById(id).addEventListener("change", () => {
    if (document.getElementById("q").value.trim()) loadPreview();
  });
}
document.getElementById("again").addEventListener("click", () => {
  seed += 1;
  view = 1;
  loadPreview();
});
async function loadPreview() {
  summaryEl.textContent = document.getElementById("q").value.includes(" ")
    ? "Ищу фразу по всем лекциям…"
    : "Считаю вхождения…";
  clipsEl.innerHTML = "";
  renderBtn.disabled = true;
  player.hidden = true;
  const response = await fetch("/api/preview?" + params());
  const payload = await response.json();
  if (!response.ok) {
    summaryEl.textContent = payload.error || "Не вышло";
    return;
  }
  setupEl.hidden = false;
  view = payload.view || view;
  const forms = (payload.forms || []).map((item) => item.word + " " + item.count).join(", ");
  const parts = payload.parts || 1;
  let head = payload.query + ": " + payload.total + " раз. ";
  if (parts > 1) {
    head += "Можно собрать " + parts + " роликов примерно по " +
      document.getElementById("seconds").value + " с. Сейчас ролик " + payload.view +
      " (~" + payload.planned_seconds + " с). ";
  } else {
    head += "Один ролик, " + payload.planned + " фрагментов, около " + payload.planned_seconds + " с. ";
  }
  head += "Эпизоды одной лекции идут подряд. " + (payload.note || "");
  if (forms) head += " Формы: " + forms + ".";
  summaryEl.textContent = head;
  paintVideos(payload.videos || []);
  let previous = "";
  clipsEl.innerHTML = (payload.clips || []).map((clip) => {
    const heading = clip.title !== previous ? "<h3>" + escapeHtml(clip.title) + "</h3>" : "";
    previous = clip.title;
    const link = clip.url ? '<a href="' + clip.url + '">' + clip.clock + "</a>" : clip.clock;
    return heading + "<article><div class=meta>" + link + " · " + escapeHtml(clip.course) +
      "</div><p>" + escapeHtml(clip.context || "") + "</p></article>";
  }).join("");
  renderBtn.disabled = !payload.indexed || !payload.planned;
  renderBtn.textContent = parts > 1 ? "Собрать ролик " + payload.view : "Собрать шортс";
}
function paintVideos(videos) {
  if (!videos || videos.length < 2) {
    videosEl.innerHTML = "";
    renderAllBtn.hidden = true;
    return;
  }
  videosEl.innerHTML = videos.map((video) => {
    const mark = video.part === view ? " gold" : "";
    return '<button type="button" class="chip' + mark + '" data-part="' + video.part + '">' +
      video.part + " · " + video.seconds + " с · " + video.lectures + " лекц.</button>";
  }).join("");
  renderAllBtn.hidden = false;
  renderAllBtn.textContent = "Собрать все " + videos.length;
}
videosEl.addEventListener("click", (event) => {
  const button = event.target.closest("[data-part]");
  if (!button) return;
  view = Number(button.dataset.part);
  loadPreview();
});
async function startRender(part) {
  renderBtn.disabled = true;
  renderAllBtn.disabled = true;
  progressEl.hidden = false;
  showBar(barFill, 1);
  barItems.innerHTML = "";
  jobEl.textContent = "Ставлю в сборку…";
  const body = Object.fromEntries(params());
  body.part = String(part);
  const response = await fetch("/api/render", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) {
    jobEl.textContent = payload.error || "Не вышло";
    renderBtn.disabled = false;
    renderAllBtn.disabled = false;
    return;
  }
  pollJob(payload.id);
}
renderBtn.addEventListener("click", () => startRender(view));
renderAllBtn.addEventListener("click", () => startRender(0));
function paintJob(job) {
  progressEl.hidden = false;
  showBar(barFill, job.percent || 0);
  const eta = formatEta(job.eta_seconds);
  jobEl.textContent = (job.message || "") + (eta ? " · " + eta : "");
  barItems.innerHTML = (job.items || []).map((item) => {
    return "<li>" + escapeHtml(item.title) + (item.detail ? " · " + escapeHtml(item.detail) : "") + "</li>";
  }).join("");
}
async function pollJob(id) {
  const response = await fetch("/api/job?id=" + encodeURIComponent(id));
  const job = await response.json();
  paintJob(job);
  if (job.status === "running") {
    setTimeout(() => pollJob(id), 700);
    return;
  }
  renderBtn.disabled = false;
  renderAllBtn.disabled = false;
  if (job.status !== "done") {
    jobEl.textContent = job.error || job.message || "Не собралось";
    return;
  }
  showBar(barFill, 100);
  barItems.innerHTML = "";
  const files = job.files && job.files.length ? job.files : [job.file];
  jobEl.innerHTML = files.map((name, index) => {
    return '<a href="/media/' + encodeURIComponent(name) + '?download=1">Ролик ' + (index + 1) + "</a>";
  }).join(" · ");
  player.hidden = false;
  player.src = "/media/" + encodeURIComponent(files[0]);
  player.play().catch(() => {});
}
refreshStatus();
</script>
</body>
</html>
"""

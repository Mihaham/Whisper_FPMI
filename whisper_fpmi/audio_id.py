"""Один и тот же звук расшифровывается один раз.

Отпечаток — Chromaprint первых 90 секунд. Повтор с тем же звуком не идёт в Whisper:
в датасете у него своя ссылка, а текст берётся из уже готовой расшифровки.

Порог 0.92: та же лекция в 32 кбит/с даёт около 0.97, две разные — около 0.54.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import json
import os
import re
import shutil
import struct
import subprocess
import threading

from loguru import logger as lg

from whisper_fpmi.names import fold_key, sanitize_title
from whisper_fpmi.paths import (
    AUDIO_INDEX_PATH,
    AUDIO_SAMPLE_DIR,
    RESULT_DIR,
    ROOT,
    TIMED_DIR,
)
from whisper_fpmi.transcript import assess_timed
from whisper_fpmi.vk import ChannelVideo

FINGERPRINT_SECONDS = 90.0
MIN_FRAMES = 80
MAX_SHIFT = 64
MATCH_SIMILARITY = 0.92
DURATION_SLACK = 45.0
DURATION_RATIO = 0.03
_SAVE_EVERY = 10
_BACKFILL_WORKERS = 2

_INDEX: AudioIndex | None = None
_INDEX_GUARD = threading.Lock()


@dataclass(frozen=True)
class Decision:
    transcribe: bool
    slug: str | None = None
    plain: str | None = None
    same_audio_as: str | None = None
    same_title: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class AudioCopy:
    video_id: str
    title: str
    url: str
    slug: str
    duration: float | None
    same_audio_as: str
    plain: str
    timed: str


@dataclass
class _Track:
    video_id: str
    title: str
    url: str
    slug: str
    duration: float | None
    fingerprint: tuple[int, ...]
    plain: str
    timed: str


@dataclass
class _Pending:
    fingerprint: tuple[int, ...]
    duration: float | None
    event: threading.Event


def get_index() -> AudioIndex:
    global _INDEX
    with _INDEX_GUARD:
        if _INDEX is None:
            _INDEX = AudioIndex()
        return _INDEX


def chromaprint_file(path: Path, *, length: float = FINGERPRINT_SECONDS) -> tuple[int, ...]:
    """Сырой Chromaprint начала ролика. Пустой кортеж, если ffmpeg не справился."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-t",
        str(length),
        "-i",
        str(path),
        "-vn",
        "-f",
        "chromaprint",
        "-fp_format",
        "raw",
        "-",
    ]
    try:
        raw = subprocess.check_output(
            command,
            stderr=subprocess.DEVNULL,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        lg.warning("ffmpeg не снял отпечаток {}: {}", path.name, exc)
        return ()
    count = len(raw) // 4
    if count <= 0:
        return ()
    return struct.unpack("<" + "I" * count, raw[: count * 4])


def similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """1.0 — тот же отпечаток. Сдвиг до ~8 с, чтобы пережить другой кодек."""
    if len(left) < MIN_FRAMES or len(right) < MIN_FRAMES:
        return 0.0
    best = _aligned_score(left, right, 0)
    if best >= MATCH_SIMILARITY:
        return best
    for shift in range(1, MAX_SHIFT + 1):
        for signed in (shift, -shift):
            score = _aligned_score(left, right, signed)
            if score >= MATCH_SIMILARITY:
                return score
            if score > best:
                best = score
    return best


def final_slug_by_fold(result_dir: Path, timed_dir: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    if not result_dir.exists():
        return found
    for path in sorted(result_dir.glob("*.txt")):
        if not path.is_file():
            continue
        if assess_timed(timed_dir / f"{path.stem}.mp4.txt") != "final":
            continue
        found.setdefault(fold_key(path.stem), path.stem)
    return found


def transcript_slug(
    video: ChannelVideo,
    state: dict,
    slug_by_fold: dict[str, str],
    timed_dir: Path,
    *,
    finals: set[str] | None = None,
) -> str | None:
    """Slug уже готовой large-v3 расшифровки этого ролика, если она есть."""
    record = (state.get("videos") or {}).get(video.video_id) or {}
    stored = str(record.get("slug") or "")
    ready = finals if finals is not None else set(slug_by_fold.values())
    if stored and stored in ready:
        return stored
    folded = slug_by_fold.get(fold_key(sanitize_title(video.title)))
    if folded:
        return folded
    if stored and assess_timed(timed_dir / f"{stored}.mp4.txt") == "final":
        return stored
    return None


class AudioIndex:
    def __init__(self, path: Path | None = None, root: Path | None = None) -> None:
        self.path = path or AUDIO_INDEX_PATH
        self.root = root or ROOT
        self._lock = threading.Lock()
        self._tracks: dict[str, _Track] = {}
        self._copies: dict[str, AudioCopy] = {}
        self._pending: dict[str, _Pending] = {}
        self._linked: dict[str, str] = {}
        self._dirty = 0
        self._load()

    @property
    def track_count(self) -> int:
        with self._lock:
            return len(self._tracks)

    @property
    def copy_count(self) -> int:
        with self._lock:
            return len(self._copies)

    def has(self, video_id: str) -> bool:
        with self._lock:
            return video_id in self._tracks or video_id in self._copies

    def copies(self) -> list[AudioCopy]:
        with self._lock:
            return list(self._copies.values())

    def linked_slug(self, video_id: str) -> str | None:
        with self._lock:
            linked = self._linked.get(video_id)
            if linked:
                return linked
            copy = self._copies.get(video_id)
            return copy.slug if copy else None

    def same_audio_as(self, video_id: str) -> str | None:
        with self._lock:
            copy = self._copies.get(video_id)
            return copy.same_audio_as if copy else None

    def prepare(self, path: Path, video: ChannelVideo) -> Decision:
        fingerprint = chromaprint_file(path)
        if len(fingerprint) < MIN_FRAMES:
            return Decision(transcribe=True)
        from whisper_fpmi.progress import probe_media_duration

        duration = video.duration or probe_media_duration(path)
        return self.consider(fingerprint, duration, video)

    def consider(
        self,
        fingerprint: tuple[int, ...],
        duration: float | None,
        video: ChannelVideo,
    ) -> Decision:
        if len(fingerprint) < MIN_FRAMES:
            return Decision(transcribe=True)
        while True:
            with self._lock:
                known = self._known_decision(video.video_id)
                if known is not None:
                    return known
                hit = self._match_track(fingerprint, duration, video.video_id)
                if hit is not None:
                    track, score = hit
                    self._add_copy(video, duration, track)
                    self._linked[video.video_id] = track.slug
                    self._write()
                    return _skip(track, track.video_id, score)
                pending = self._match_pending(fingerprint, duration, video.video_id)
                if pending is None:
                    self._pending[video.video_id] = _Pending(
                        fingerprint=fingerprint,
                        duration=duration,
                        event=threading.Event(),
                    )
                    return Decision(transcribe=True)
                event = pending.event
            event.wait()

    def observe(
        self,
        fingerprint: tuple[int, ...],
        duration: float | None,
        video: ChannelVideo,
        slug: str,
    ) -> str:
        """Уже готовая расшифровка: запомнить звук или привязать к уже известному."""
        if len(fingerprint) < MIN_FRAMES:
            return "short"
        while True:
            with self._lock:
                if video.video_id in self._tracks or video.video_id in self._copies:
                    return "known"
                hit = self._match_track(fingerprint, duration, video.video_id)
                if hit is not None:
                    track, _score = hit
                    self._add_copy(video, duration, track)
                    self._mark()
                    return "copy"
                pending = self._match_pending(fingerprint, duration, video.video_id)
                if pending is None:
                    plain, timed = _rel_paths(slug)
                    self._tracks[video.video_id] = _Track(
                        video_id=video.video_id,
                        title=video.title,
                        url=video.url,
                        slug=slug,
                        duration=duration,
                        fingerprint=fingerprint,
                        plain=plain,
                        timed=timed,
                    )
                    self._mark()
                    return "track"
                event = pending.event
            event.wait()

    def commit(self, video: ChannelVideo, slug: str) -> None:
        with self._lock:
            pending = self._pending.pop(video.video_id, None)
            if pending is None:
                return
            plain, timed = _rel_paths(slug)
            self._tracks[video.video_id] = _Track(
                video_id=video.video_id,
                title=video.title,
                url=video.url,
                slug=slug,
                duration=pending.duration,
                fingerprint=pending.fingerprint,
                plain=plain,
                timed=timed,
            )
            try:
                self._write()
            finally:
                pending.event.set()

    def abort(self, video_id: str) -> None:
        with self._lock:
            pending = self._pending.pop(video_id, None)
            if pending is not None:
                pending.event.set()

    def flush(self) -> None:
        with self._lock:
            if self._dirty:
                self._write()

    def _known_decision(self, video_id: str) -> Decision | None:
        copy = self._copies.get(video_id)
        if copy is not None and self._usable(copy.timed):
            self._linked[video_id] = copy.slug
            return _skip(copy, copy.same_audio_as, None)
        tracked = self._tracks.get(video_id)
        if tracked is not None and self._usable(tracked.timed):
            self._linked[video_id] = tracked.slug
            return _skip(tracked, None, None)
        return None

    def _match_track(
        self,
        fingerprint: tuple[int, ...],
        duration: float | None,
        ignore: str,
    ) -> tuple[_Track, float] | None:
        found: list[tuple[float, _Track]] = []
        for track in self._tracks.values():
            if track.video_id == ignore or not _same_length(track.duration, duration):
                continue
            score = similarity(fingerprint, track.fingerprint)
            if score >= MATCH_SIMILARITY:
                found.append((score, track))
        found.sort(key=lambda item: item[0], reverse=True)
        for score, track in found:
            if self._usable(track.timed):
                return track, score
        return None

    def _match_pending(
        self,
        fingerprint: tuple[int, ...],
        duration: float | None,
        ignore: str,
    ) -> _Pending | None:
        for video_id, pending in self._pending.items():
            if video_id == ignore:
                continue
            if not _same_length(pending.duration, duration):
                continue
            if similarity(fingerprint, pending.fingerprint) >= MATCH_SIMILARITY:
                return pending
        return None

    def _usable(self, timed: str) -> bool:
        return assess_timed(self.root / timed) == "final"

    def _add_copy(self, video: ChannelVideo, duration: float | None, track: _Track) -> AudioCopy:
        self._tracks.pop(video.video_id, None)
        copy = AudioCopy(
            video_id=video.video_id,
            title=video.title,
            url=video.url,
            slug=track.slug,
            duration=duration if duration is not None else track.duration,
            same_audio_as=track.video_id,
            plain=track.plain,
            timed=track.timed,
        )
        self._copies[video.video_id] = copy
        return copy

    def _mark(self) -> None:
        self._dirty += 1
        if self._dirty >= _SAVE_EVERY:
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "tracks": {video_id: _track_json(track) for video_id, track in self._tracks.items()},
            "copies": {video_id: _copy_json(copy) for video_id, copy in self._copies.items()},
        }
        blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(blob, encoding="utf-8")
        os.replace(temporary, self.path)
        self._dirty = 0

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            broken = self.path.with_suffix(".broken.json")
            lg.warning("Битый {} ({}), оставляю {}", self.path.name, exc, broken.name)
            os.replace(self.path, broken)
            return
        if payload.get("version") != 1:
            return
        for video_id, raw in (payload.get("tracks") or {}).items():
            track = _track_from(str(video_id), raw)
            if track is not None:
                self._tracks[track.video_id] = track
        for video_id, raw in (payload.get("copies") or {}).items():
            copy = _copy_from(str(video_id), raw)
            if copy is not None:
                self._copies[copy.video_id] = copy


def start_audio_backfill(
    videos: list[ChannelVideo],
    state: dict,
    stop: threading.Event,
    *,
    cookies: str | None = None,
    skip_ids: set[str] | None = None,
    slug_by_fold: dict[str, str] | None = None,
    index: AudioIndex | None = None,
) -> threading.Thread | None:
    """В фоне снимает отпечатки уже готовых роликов, пока GPU расшифровывает новые."""
    index = index or get_index()
    skip = skip_ids or set()
    if slug_by_fold is None:
        slug_by_fold = final_slug_by_fold(RESULT_DIR, TIMED_DIR)
    finals = set(slug_by_fold.values())
    jobs: list[tuple[ChannelVideo, str]] = []
    for video in videos:
        if video.video_id in skip or index.has(video.video_id):
            continue
        slug = transcript_slug(video, state, slug_by_fold, TIMED_DIR, finals=finals)
        if slug is None:
            continue
        jobs.append((video, slug))
    if not jobs:
        if index.track_count:
            lg.info(
                "Отпечатки звука: {}, копий: {}",
                index.track_count,
                index.copy_count,
            )
        return None
    thread = threading.Thread(
        target=_run_backfill,
        args=(jobs, cookies, stop, index),
        name="audio-fingerprint",
        daemon=True,
    )
    thread.start()
    return thread


def _run_backfill(
    jobs: list[tuple[ChannelVideo, str]],
    cookies: str | None,
    stop: threading.Event,
    index: AudioIndex,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    lg.info(
        "Снимаю отпечатки звука у {} уже готовых видео, копии Whisper не повторит",
        len(jobs),
    )
    added = [0]
    copies = [0]
    errors = [0]
    counter = threading.Lock()

    def one(video: ChannelVideo, slug: str) -> None:
        if stop.is_set():
            return
        try:
            kind = _fingerprint_known(video, slug, cookies, index)
        except Exception as exc:  # noqa: BLE001
            with counter:
                errors[0] += 1
                count = errors[0]
            if count <= 5 or count % 100 == 0:
                lg.warning("Отпечаток {} не снялся ({}): {}", video.video_id, count, exc)
            return
        with counter:
            if kind == "track":
                added[0] += 1
            elif kind == "copy":
                copies[0] += 1
            done = added[0] + copies[0] + errors[0]
            snapshot = (added[0], copies[0], errors[0])
        if done and done % 50 == 0:
            lg.info(
                "Отпечатки: {} новых, {} тот же звук, ошибок {}",
                snapshot[0],
                snapshot[1],
                snapshot[2],
            )

    with ThreadPoolExecutor(max_workers=_BACKFILL_WORKERS) as pool:
        futures = []
        for video, slug in jobs:
            if stop.is_set():
                break
            futures.append(pool.submit(one, video, slug))
        for future in futures:
            future.result()
    index.flush()
    lg.info(
        "Отпечатки готовых видео: {} новых, {} тот же звук, ошибок {}",
        added[0],
        copies[0],
        errors[0],
    )


def _fingerprint_known(
    video: ChannelVideo,
    slug: str,
    cookies: str | None,
    index: AudioIndex,
) -> str:
    if index.has(video.video_id):
        return "known"
    folder = _sample_folder(video.video_id)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        sample = _download_head(folder, video, cookies)
        fingerprint = chromaprint_file(sample)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    if len(fingerprint) < MIN_FRAMES:
        return "short"
    return index.observe(fingerprint, video.duration, video, slug)


def _download_head(folder: Path, video: ChannelVideo, cookies: str | None) -> Path:
    import yt_dlp
    from yt_dlp.utils import download_range_func

    from whisper_fpmi.download import FORMAT_SORT, format_for_url, source_url

    page = source_url(video)
    options = {
        "format": format_for_url(page),
        "format_sort": FORMAT_SORT,
        "outtmpl": str(folder / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": 30,
        "overwrites": True,
        "windowsfilenames": True,
        "download_ranges": download_range_func(None, [(0, FINGERPRINT_SECONDS + 10)]),
    }
    if cookies:
        options["cookiefile"] = cookies
    with yt_dlp.YoutubeDL(options) as client:
        info = client.extract_info(page, download=True) or {}
        filename = Path(client.prepare_filename(info))
    if filename.is_file():
        return filename
    matches = [
        item
        for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() not in {".part", ".ytdl", ".tmp"}
    ]
    if not matches:
        raise FileNotFoundError(video.video_id)
    return matches[0]


def _sample_folder(video_id: str) -> Path:
    safe = re.sub(r"[^\w.-]+", "_", video_id).strip("._") or "video"
    return AUDIO_SAMPLE_DIR / safe


def _aligned_score(left: tuple[int, ...], right: tuple[int, ...], shift: int) -> float:
    if shift >= 0:
        first, second = left, right[shift:]
    else:
        first, second = left[-shift:], right
    count = min(len(first), len(second))
    if count < MIN_FRAMES:
        return 0.0
    budget = int(count * 32 * (1.0 - MATCH_SIMILARITY))
    errors = 0
    for index in range(count):
        errors += (first[index] ^ second[index]).bit_count()
        if errors > budget:
            return 0.0
    return 1.0 - errors / (count * 32)


def _same_length(left: float | None, right: float | None) -> bool:
    if left is None or right is None or left <= 0 or right <= 0:
        return True
    slack = max(DURATION_SLACK, DURATION_RATIO * max(left, right))
    return abs(left - right) <= slack


def _rel_paths(slug: str) -> tuple[str, str]:
    return f"result/{slug}.txt", f"download/{slug}.mp4.txt"


def _skip(source, same_audio_as: str | None, score: float | None) -> Decision:
    return Decision(
        transcribe=False,
        slug=source.slug,
        plain=source.plain,
        same_audio_as=same_audio_as,
        same_title=source.title,
        score=score,
    )


def _pack(values: tuple[int, ...]) -> str:
    if not values:
        return ""
    return base64.b64encode(struct.pack("<" + "I" * len(values), *values)).decode("ascii")


def _unpack(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    raw = base64.b64decode(value)
    count = len(raw) // 4
    if count <= 0:
        return ()
    return struct.unpack("<" + "I" * count, raw[: count * 4])


def _track_json(track: _Track) -> dict:
    return {
        "title": track.title,
        "url": track.url,
        "slug": track.slug,
        "duration": track.duration,
        "fingerprint": _pack(track.fingerprint),
        "plain": track.plain,
        "timed": track.timed,
    }


def _copy_json(copy: AudioCopy) -> dict:
    return {
        "title": copy.title,
        "url": copy.url,
        "slug": copy.slug,
        "duration": copy.duration,
        "same_audio_as": copy.same_audio_as,
        "plain": copy.plain,
        "timed": copy.timed,
    }


def _track_from(video_id: str, raw: dict) -> _Track | None:
    if not isinstance(raw, dict):
        return None
    slug = str(raw.get("slug") or "")
    fingerprint = _unpack(str(raw.get("fingerprint") or ""))
    if not slug or len(fingerprint) < MIN_FRAMES:
        return None
    plain, timed = _rel_paths(slug)
    return _Track(
        video_id=video_id,
        title=str(raw.get("title") or ""),
        url=str(raw.get("url") or ""),
        slug=slug,
        duration=_duration(raw.get("duration")),
        fingerprint=fingerprint,
        plain=str(raw.get("plain") or plain),
        timed=str(raw.get("timed") or timed),
    )


def _copy_from(video_id: str, raw: dict) -> AudioCopy | None:
    if not isinstance(raw, dict):
        return None
    slug = str(raw.get("slug") or "")
    same = str(raw.get("same_audio_as") or "")
    if not slug or not same:
        return None
    plain, timed = _rel_paths(slug)
    return AudioCopy(
        video_id=video_id,
        title=str(raw.get("title") or ""),
        url=str(raw.get("url") or ""),
        slug=slug,
        duration=_duration(raw.get("duration")),
        same_audio_as=same,
        plain=str(raw.get("plain") or plain),
        timed=str(raw.get("timed") or timed),
    )


def _duration(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number

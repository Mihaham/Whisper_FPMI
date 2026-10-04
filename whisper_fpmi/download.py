from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from loguru import logger as lg

from whisper_fpmi.names import sanitize_title
from whisper_fpmi.paths import DEFAULT_FRAGMENT_THREADS, VIDEO_DIR
from whisper_fpmi.progress import HourDayTqdm
from whisper_fpmi.vk import ChannelVideo

# Низкое https-аудио. HLS с YouTube без JS-рантайма не собирается.
YOUTUBE_FORMAT = (
    "worstaudio[protocol^=https]/"
    "bestaudio[protocol^=https]/"
    "worstaudio"
)

# 144p со звуком, если VK отдаёт url144; иначе минимальное видео+аудио.
WORST_FORMAT = (
    "url144/url240/hls-275/"
    "worstvideo*[height<=240]+bestaudio/"
    "worstvideo*+worstaudio/worst"
)
FORMAT_SORT = ["+res", "+size", "+br"]

_BYTE_BAR = (
    "{desc}: {percentage:3.0f}%|{bar}| "
    "{n_fmt}/{total_fmt} "
    "прошло {elapsed} осталось {left}"
)
_FRAG_BAR = (
    "{desc}: {percentage:3.0f}%|{bar}| "
    "{n_fmt}/{total_fmt} фраг "
    "прошло {elapsed} осталось {left}"
)
_UNKNOWN_BAR = "{desc}: {n_fmt} прошло {elapsed}"


@dataclass(frozen=True)
class DownloadTick:
    key: str
    done: int
    total: int | None
    unit: str
    label: str
    finished: bool


def parse_download_hook(data: dict) -> DownloadTick | None:
    status = data.get("status")
    if status not in {"downloading", "finished"}:
        return None
    info = data.get("info_dict") or {}
    key = str(
        info.get("format_id")
        or data.get("filename")
        or data.get("tmpfilename")
        or "media"
    )
    finished = status == "finished"
    label = _stream_label(info)
    total_bytes = data.get("total_bytes") or data.get("total_bytes_estimate")
    downloaded = int(data.get("downloaded_bytes") or 0)
    if total_bytes:
        total = max(int(total_bytes), 1)
        done = max(downloaded, total) if finished else min(downloaded, total)
        total = max(total, done)
        return DownloadTick(key, done, total, "B", label, finished)
    fragment_count = data.get("fragment_count")
    if fragment_count:
        total = max(int(fragment_count), 1)
        index = int(data.get("fragment_index") or 0)
        done = total if finished else min(max(index, 0), total)
        return DownloadTick(key, done, total, "frag", label, finished)
    total = downloaded if finished and downloaded else None
    return DownloadTick(key, downloaded, total, "B", label, finished)


def _stream_label(info: dict) -> str:
    has_video = info.get("vcodec") not in (None, "none")
    has_audio = info.get("acodec") not in (None, "none")
    if has_video and not has_audio:
        return "видео"
    if has_audio and not has_video:
        return "аудио"
    return "файл"


class ParallelHttpError(Exception):
    pass


def split_ranges(total: int, parts: int) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    parts = max(1, min(parts, total))
    base = total // parts
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        length = base if index < parts - 1 else total - start
        end = start + length - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def _progressive_url(info: dict) -> str | None:
    if info.get("requested_formats"):
        return None
    protocol = str(info.get("protocol") or "https")
    url = info.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return None
    if protocol.startswith("m3u8") or ".m3u8" in url:
        return None
    return url


def _http_headers(info: dict) -> dict[str, str]:
    raw = info.get("http_headers") or {}
    headers = {
        str(key): str(value)
        for key, value in raw.items()
        if value is not None
    }
    headers.setdefault("Accept-Encoding", "identity")
    return headers


def _content_length(url: str, headers: dict[str, str]) -> int:
    import requests

    probe = dict(headers)
    probe["Range"] = "bytes=0-0"
    response = requests.get(url, headers=probe, timeout=(15, 30), stream=True)
    try:
        if response.status_code != 206:
            raise ParallelHttpError(f"сервер не отдаёт Range (HTTP {response.status_code})")
        match = re.search(r"/(\d+)\s*$", response.headers.get("Content-Range", ""))
        if not match:
            raise ParallelHttpError("нет полного размера в Content-Range")
        total = int(match.group(1))
    finally:
        response.close()
    if total <= 0:
        raise ParallelHttpError("пустой файл")
    return total


def _fetch_range(
    url: str,
    headers: dict[str, str],
    start: int,
    end: int,
    path: Path,
    progress: list[int],
    lock: threading.Lock,
    on_progress,
    total: int,
) -> None:
    import requests

    expected = end - start + 1
    range_headers = dict(headers)
    range_headers["Range"] = f"bytes={start}-{end}"
    last_error: Exception | None = None
    for _attempt in range(4):
        got = 0
        try:
            response = requests.get(
                url,
                headers=range_headers,
                timeout=(15, 120),
                stream=True,
            )
            try:
                if response.status_code != 206:
                    code = response.status_code
                    if code in {408, 429, 500, 502, 503, 504}:
                        raise OSError(f"HTTP {code} на {start}-{end}")
                    raise ParallelHttpError(f"HTTP {code} на {start}-{end}")
                with open(path, "r+b") as handle:
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        if not chunk:
                            continue
                        handle.seek(start + got)
                        handle.write(chunk)
                        with lock:
                            got += len(chunk)
                            progress[0] += len(chunk)
                            done = progress[0]
                        if on_progress is not None:
                            on_progress(done, total)
            finally:
                response.close()
            if got != expected:
                raise OSError(f"диапазон {start}-{end}: {got} из {expected}")
            return
        except ParallelHttpError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            with lock:
                progress[0] -= got
    raise ParallelHttpError(str(last_error or "сбой диапазона"))


def download_parallel_http(
    url: str,
    destination: Path,
    *,
    headers: dict[str, str] | None = None,
    connections: int = DEFAULT_FRAGMENT_THREADS,
    on_progress=None,
) -> None:
    header_map = dict(headers or {})
    header_map.setdefault("Accept-Encoding", "identity")
    total = _content_length(url, header_map)
    ranges = split_ranges(total, max(1, connections))
    part = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(part, "wb") as handle:
        handle.truncate(total)
    progress = [0]
    lock = threading.Lock()
    try:
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [
                pool.submit(
                    _fetch_range,
                    url,
                    header_map,
                    start,
                    end,
                    part,
                    progress,
                    lock,
                    on_progress,
                    total,
                )
                for start, end in ranges
            ]
            for future in futures:
                future.result()
        os.replace(part, destination)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    if on_progress is not None:
        on_progress(total, total)


def _clip(title: str, limit: int = 28) -> str:
    text = " ".join(title.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class DownloadProgress:
    def __init__(
        self,
        title: str,
        *,
        position: int = 0,
        disable: bool = False,
    ) -> None:
        self._title = _clip(title)
        self._position = position
        self._disable = disable
        self._bar = None
        self._key: str | None = None
        self._unit: str | None = None
        self._known_total = False
        self._lock = threading.RLock()

    def hook(self, data: dict) -> None:
        tick = parse_download_hook(data)
        if tick is None:
            return
        with self._lock:
            known = tick.total is not None
            if (
                self._bar is None
                or tick.key != self._key
                or tick.unit != self._unit
                or known != self._known_total
            ):
                self._open(tick)
            assert self._bar is not None
            if tick.total is not None:
                self._bar.total = tick.total
            self._bar.n = tick.done
            self._bar.refresh()

    def close(self) -> None:
        with self._lock:
            if self._bar is not None:
                self._bar.close()
                self._bar = None

    def _open(self, tick: DownloadTick) -> None:
        self.close()
        self._key = tick.key
        self._unit = tick.unit
        self._known_total = tick.total is not None
        if tick.unit == "frag":
            bar_format = _FRAG_BAR
        elif tick.total is not None:
            bar_format = _BYTE_BAR
        else:
            bar_format = _UNKNOWN_BAR
        self._bar = HourDayTqdm(
            total=tick.total,
            initial=tick.done,
            desc=f"Скачивание {tick.label}: {self._title}",
            position=self._position,
            leave=False,
            unit="B" if tick.unit == "B" else "фраг",
            unit_scale=tick.unit == "B" and tick.total is not None,
            unit_divisor=1024,
            dynamic_ncols=True,
            mininterval=0.3,
            bar_format=bar_format,
            disable=self._disable,
            file=sys.stderr,
        )


def source_url(video: ChannelVideo) -> str:
    url = (video.url or "").strip()
    if url.startswith(("http://", "https://")):
        return url
    return video.vk_url


def format_for_url(url: str) -> str:
    lowered = url.casefold()
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return YOUTUBE_FORMAT
    return WORST_FORMAT


def _ydl_options(
    video: ChannelVideo,
    output_dir: Path,
    cookies: str | None = None,
    fragment_threads: int = DEFAULT_FRAGMENT_THREADS,
) -> dict:
    slug = sanitize_title(video.title)
    template = str(output_dir / f"{video.video_id}_{slug}.%(ext)s")
    page = source_url(video)
    options = {
        "format": format_for_url(page),
        "format_sort": FORMAT_SORT,
        "merge_output_format": "mp4",
        "outtmpl": template,
        "windowsfilenames": True,
        "retries": 10,
        "fragment_retries": 10,
        "noprogress": False,
        "quiet": False,
        "noplaylist": True,
        "overwrites": True,
        "concurrent_fragment_downloads": max(1, fragment_threads),
    }
    if cookies:
        options["cookiefile"] = cookies
    return options


def probe_filesize(video: ChannelVideo, cookies: str | None = None) -> int:
    import yt_dlp

    options = _ydl_options(video, VIDEO_DIR, cookies)
    options.update({"skip_download": True, "quiet": True, "noprogress": True})
    try:
        with yt_dlp.YoutubeDL(options) as client:
            info = client.extract_info(source_url(video), download=False)
    except Exception as exc:  # noqa: BLE001
        lg.debug("Не удалось оценить размер {}: {}", video.video_id, exc)
        return 0
    return int(
        (info or {}).get("filesize")
        or (info or {}).get("filesize_approx")
        or 0
    )


def download_worst_quality(
    video: ChannelVideo,
    output_dir: Path = VIDEO_DIR,
    cookies: str | None = None,
    quiet: bool = False,
    bar_position: int = 0,
    fragment_threads: int = DEFAULT_FRAGMENT_THREADS,
) -> Path:
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)
    options = _ydl_options(
        video,
        output_dir,
        cookies,
        fragment_threads=fragment_threads,
    )
    options.update({"quiet": True, "noprogress": True, "no_warnings": True})
    connections = max(1, fragment_threads)
    tracker = DownloadProgress(
        video.title,
        position=bar_position,
        disable=quiet or not sys.stderr.isatty(),
    )
    lg.info(
        "Скачиваю {} в худшем качестве, {} соед.: {}",
        video.video_id,
        connections,
        video.title,
    )

    def report(done: int, total: int) -> None:
        tracker.hook(
            {
                "status": "downloading" if done < total else "finished",
                "downloaded_bytes": done,
                "total_bytes": total,
                "filename": video.video_id,
                "info_dict": {"format_id": video.video_id},
            }
        )

    try:
        with yt_dlp.YoutubeDL(options) as client:
            info = client.extract_info(source_url(video), download=False) or {}
            filename = client.prepare_filename(info)
            direct = _progressive_url(info)
            if direct and connections > 1:
                try:
                    download_parallel_http(
                        direct,
                        Path(filename),
                        headers=_http_headers(info),
                        connections=connections,
                        on_progress=report,
                    )
                except ParallelHttpError as exc:
                    lg.warning(
                        "Параллельная закачка {} не вышла ({}), одно соединение",
                        video.video_id,
                        exc,
                    )
                    Path(filename).unlink(missing_ok=True)
                    client.add_progress_hook(tracker.hook)
                    client.process_info(info)
            else:
                client.add_progress_hook(tracker.hook)
                client.process_info(info)
    finally:
        tracker.close()
    path = Path(filename)
    if path.exists():
        return path
    matches = list(output_dir.glob(f"{video.video_id}_*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"yt-dlp не сохранил файл для {video.video_id}")

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from loguru import logger as lg

from whisper_fpmi.names import sanitize_title
from whisper_fpmi.paths import DEFAULT_FRAGMENT_THREADS, VIDEO_DIR
from whisper_fpmi.progress import HourDayTqdm
from whisper_fpmi.vk import ChannelVideo

# Отпечаток Яндекс Браузера: YouTube сверяет с ним сессию, иначе куки не принимаются.
YOUTUBE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/152.0.0.0 YaBrowser/26.10.0.0 Safari/537.36"
    ),
    "Upgrade-Insecure-Requests": "1",
    "device-memory": "32",
    "sec-ch-dpr": "1",
    "sec-ch-ua": '"Not?A_Brand";v="24", "Chromium";v="152"',
    "sec-ch-ua-arch": '"x86"',
    "sec-ch-ua-bitness": '"64"',
    "sec-ch-ua-form-factors": '"Desktop"',
    "sec-ch-ua-full-version": '"152.0.7977.140"',
    "sec-ch-ua-full-version-list": '"Not?A_Brand";v="24.0.0.0", "Chromium";v="152.0.7977.1854"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-model": '""',
    "sec-ch-ua-platform": '"Windows"',
    "sec-ch-ua-platform-version": '"19.0.0"',
    "sec-ch-ua-wow64": "?0",
    "sec-ch-viewport-width": "1920",
}

# Низкое https-аудио. Если клиент отдал только 360p, берём его.
YOUTUBE_FORMAT = (
    "worstaudio[protocol^=https]/"
    "worst[protocol^=https]/"
    "worstaudio/"
    "worst/"
    "ba/b"
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


class _QuietYdl:
    """Прячет stderr yt-dlp. Про куки и прочие ошибки пишет уже наш код, одной строкой."""

    def debug(self, msg: str) -> None:
        return

    def info(self, msg: str) -> None:
        return

    def warning(self, msg: str) -> None:
        return

    def error(self, msg: str) -> None:
        return


def brief_ydl_error(exc: BaseException) -> str:
    """Одна строка вместо простыни yt-dlp про --cookies и шаблон issue."""
    text = str(exc).strip()
    first = text.splitlines()[0] if text else type(exc).__name__
    lowered = first.casefold()
    if "cookies" in lowered or "not a bot" in lowered:
        return "YouTube закрыл веб-страницу без куков"
    cut = first.split("; please report", 1)[0].strip()
    return cut[:240]


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
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            response = requests.get(url, headers=probe, timeout=(30, 90), stream=True)
        except requests.RequestException as exc:
            last_error = exc
            continue
        try:
            if response.status_code != 206:
                raise ParallelHttpError(
                    f"сервер не отдаёт Range (HTTP {response.status_code})"
                )
            match = re.search(r"/(\d+)\s*$", response.headers.get("Content-Range", ""))
            if not match:
                raise ParallelHttpError("нет полного размера в Content-Range")
            total = int(match.group(1))
        finally:
            response.close()
        if total <= 0:
            raise ParallelHttpError("пустой файл")
        return total
    raise ParallelHttpError(str(last_error or "не удалось узнать размер"))


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
                timeout=(30, 180),
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
        "fragment_retries": 15,
        "extractor_retries": 4,
        "file_access_retries": 5,
        "socket_timeout": 60,
        "noprogress": False,
        "quiet": True,
        "no_warnings": True,
        "logger": _QuietYdl(),
        "noplaylist": True,
        "overwrites": True,
        "concurrent_fragment_downloads": max(1, fragment_threads),
    }
    if "youtube.com" in page.casefold() or "youtu.be" in page.casefold():
        # Запрос к ролику идёт с тем же отпечатком, что и страница в Яндекс Браузере.
        youtube_args: dict[str, list[str]] = {
            "player_client": ["web", "web_safari", "mweb"] if cookies else ["web_safari", "mweb", "tv_samsung"],
        }
        if _po_token_script().is_file():
            youtube_args["fetch_pot"] = ["always"]
        options["extractor_args"] = {"youtube": youtube_args}
        options["http_headers"] = dict(YOUTUBE_HEADERS)
        options["js_runtimes"] = {"node": {}}
    if cookies:
        options["cookiefile"] = cookies
    return options


def _po_token_script() -> Path:
    return Path.home() / "bgutil-ytdlp-pot-provider" / "server" / "build" / "generate_once.js"


_BOT_PAUSE_SECONDS = 180
_bot_pause_until = 0.0
_bot_lock = threading.Lock()
_po_server_started = False


def _transient_network(exc: BaseException) -> bool:
    text = str(exc).casefold()
    return any(
        token in text
        for token in (
            "timeout",
            "timed out",
            "handshake",
            "ssl",
            "connection",
            "reset",
            "temporar",
            "10054",
            "разорвал",
            "broken pipe",
            "eof occurred",
        )
    )


def _is_bot_wall(message: str) -> bool:
    lowered = message.casefold()
    return "без куков" in lowered or "not a bot" in lowered


def _note_bot_wall() -> None:
    global _bot_pause_until
    with _bot_lock:
        import time

        now = time.time()
        if now < _bot_pause_until:
            return
        _bot_pause_until = now + _BOT_PAUSE_SECONDS
    lg.warning(
        "YouTube требует вход с этого адреса. Следующая попытка через {} мин",
        _BOT_PAUSE_SECONDS // 60,
    )


def _wait_bot_pause() -> None:
    import time

    delay = _bot_pause_until - time.time()
    if delay > 0:
        time.sleep(delay)


def ensure_po_token_server() -> None:
    """Локальный генератор PO-токена, если он уже собран в домашней папке."""
    global _po_server_started
    if _po_server_started or not _po_token_script().is_file():
        return
    import socket
    import subprocess

    probe = socket.socket()
    try:
        probe.settimeout(0.3)
        if probe.connect_ex(("127.0.0.1", 4416)) == 0:
            _po_server_started = True
            return
    finally:
        probe.close()
    node = shutil.which("node")
    if not node:
        return
    server = _po_token_script().with_name("main.js")
    if not server.is_file():
        return
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [node, str(server)],
        cwd=str(server.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    _po_server_started = True


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


def _remove_partials(output_dir: Path, video_id: str) -> None:
    if not output_dir.exists():
        return
    for item in output_dir.glob(f"{video_id}_*"):
        name = item.name.casefold()
        if ".part" not in name and not name.endswith(".ytdl") and not name.endswith(".tmp"):
            continue
        try:
            if item.is_file():
                item.unlink(missing_ok=True)
        except OSError:
            continue


def _is_youtube(video: ChannelVideo) -> bool:
    page = source_url(video).casefold()
    return "youtube.com" in page or "youtu.be" in page


def _stream_protocol(fmt: object) -> str:
    protocol = getattr(fmt, "protocol", "")
    return str(getattr(protocol, "value", protocol)).casefold()


def _worst_youtube_stream(formats: list[object]) -> object | None:
    """Самый лёгкий https-поток. Для расшифровки хватает худшего аудио."""
    usable = [
        fmt
        for fmt in formats
        if getattr(fmt, "url", "")
        and not getattr(fmt, "has_drm", False)
        and not getattr(fmt, "requires_pot", False)
        and _stream_protocol(fmt) in {"https", "http"}
    ]
    audio = [fmt for fmt in usable if getattr(fmt, "is_audio", False)]
    pool = audio or [fmt for fmt in usable if getattr(fmt, "is_video", False)]
    if not pool:
        return None
    return min(
        pool,
        key=lambda fmt: (
            getattr(fmt, "bitrate", None) or 10**12,
            getattr(fmt, "height", None) or 10**12,
            getattr(fmt, "filesize", None) or 10**12,
        ),
    )


def _youtube_suffix(fmt: object) -> str:
    mime = str(getattr(fmt, "mime_type", "") or "").casefold()
    if "webm" in mime:
        return ".webm"
    if getattr(fmt, "is_audio", False):
        return ".m4a"
    return ".mp4"


def _googlevideo_mirrors(url: str) -> list[str]:
    """Запасные узлы из параметра mn. Первый хост из России часто рвёт связь."""
    from urllib.parse import parse_qs, urlparse, urlunparse

    parsed = urlparse(url)
    current_server = ""
    if "---" in parsed.netloc:
        current_server = parsed.netloc.split("---", 1)[1].removesuffix(".googlevideo.com")
    names: list[str] = []
    for item in parse_qs(parsed.query).get("mn", []):
        names.extend(part.strip() for part in item.split(",") if part.strip())
    others = [name for name in names if name.removesuffix(".googlevideo.com") != current_server]
    same = [name for name in names if name.removesuffix(".googlevideo.com") == current_server]
    mirrors = [url]
    seen = {parsed.netloc}
    for name in others + same:
        server = name.removesuffix(".googlevideo.com")
        for index in range(1, 6):
            host = f"rr{index}---{server}.googlevideo.com"
            if host in seen:
                continue
            seen.add(host)
            mirrors.append(urlunparse(parsed._replace(netloc=host)))
    return mirrors


def _pull_youtube_file(holder: dict, part: Path, report, refresh) -> None:
    """Качает файл кусками и, если узел рвёт связь, переходит на зеркало из той же ссылки."""
    import time

    import httpx

    from ydpy.downloader.utils import STREAM_HEADERS

    chunk = 1024 * 1024
    timeout = httpx.Timeout(120.0, connect=20.0)
    stalls = 0
    refreshes = 0
    mirrors = _googlevideo_mirrors(holder["url"])
    mirror_at = 0
    with httpx.Client(timeout=timeout, follow_redirects=True, http2=False) as client:
        while True:
            total = holder["total"]
            downloaded = part.stat().st_size if part.is_file() else 0
            if total is not None and downloaded >= total:
                report(downloaded, total)
                return
            end = downloaded + chunk - 1
            if total is not None:
                end = min(end, total - 1)
            headers = dict(STREAM_HEADERS)
            headers["Range"] = f"bytes={downloaded}-{end}"
            try:
                with client.stream("GET", holder["url"], headers=headers) as response:
                    if response.status_code in {401, 403, 410} or (
                        response.status_code == 200 and downloaded > 0
                    ):
                        raise _StaleUrl()
                    if response.status_code == 416:
                        if total is not None and downloaded >= total:
                            return
                        raise _StaleUrl()
                    if response.status_code not in {200, 206}:
                        raise OSError(f"HTTP {response.status_code}")
                    limit = None if response.status_code == 200 else end + 1
                    cursor = downloaded
                    with part.open("ab" if cursor else "wb") as handle:
                        for piece in response.iter_bytes(32 * 1024):
                            if not piece:
                                break
                            if limit is not None:
                                room = limit - cursor
                                if room <= 0:
                                    break
                                piece = piece[:room]
                            handle.write(piece)
                            handle.flush()
                            cursor += len(piece)
                            report(cursor, total)
                            if limit is not None and cursor >= limit:
                                break
                if cursor == downloaded:
                    stalls += 1
                else:
                    stalls = 0
                    refreshes = 0
                if stalls >= 30:
                    raise TimeoutError(f"нет прогресса на {downloaded} байтах")
            except _StaleUrl:
                refreshes += 1
                if refreshes > 12:
                    raise TimeoutError("ссылка на файл YouTube протухла")
                refresh()
                mirrors = _googlevideo_mirrors(holder["url"])
                mirror_at = 0
                stalls = 0
                time.sleep(2)
            except Exception as exc:  # noqa: BLE001
                if not _transient_network(exc):
                    raise
                stalls += 1
                if stalls >= 40:
                    raise
                mirror_at += 1
                if mirror_at < len(mirrors):
                    holder["url"] = mirrors[mirror_at]
                else:
                    refreshes += 1
                    if refreshes > 6:
                        raise
                    holder["url"] = refresh()
                    mirrors = _googlevideo_mirrors(holder["url"])
                    mirror_at = 0
                host = holder["url"].split("/", 3)[2]
                lg.warning("Узел оборвал связь на {} байтах, пробую {}", downloaded, host)
                time.sleep(1)


class _StaleUrl(Exception):
    pass


def _download_youtube_worst(
    video: ChannelVideo,
    output_dir: Path,
    *,
    quiet: bool,
    bar_position: int,
) -> Path:
    import ydpy

    import httpx

    output_dir.mkdir(parents=True, exist_ok=True)
    page = source_url(video)
    lg.info("Скачиваю {} через ydpy в худшем качестве: {}", video.video_id, video.title)
    data = None
    last_error: Exception | None = None
    timeout = httpx.Timeout(180.0, connect=60.0)
    for attempt in range(5):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                data = ydpy.PlayableVideo.fetch(page, http_client=client)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 4 or not _transient_network(exc):
                raise
            pause = 15 * (attempt + 1)
            lg.warning(
                "ydpy не открыл {} (попытка {}), жду {} с: {}",
                video.video_id,
                attempt + 1,
                pause,
                brief_ydl_error(exc),
            )
            import time

            time.sleep(pause)
    if data is None:
        raise last_error or RuntimeError(f"ydpy не открыл {video.video_id}")
    chosen = _worst_youtube_stream(list(data.formats))
    if chosen is None:
        raise RuntimeError(f"ydpy не нашёл поток для {video.video_id}")
    path = output_dir / f"{video.video_id}_{sanitize_title(video.title)}{_youtube_suffix(chosen)}"
    part = path.with_name(path.name + ".part")
    tracker = DownloadProgress(
        video.title,
        position=bar_position,
        disable=quiet or not sys.stderr.isatty(),
    )
    holder = {"url": str(chosen.url), "total": int(getattr(chosen, "filesize", None) or 0) or None}

    def refresh() -> str:
        fresh = ydpy.PlayableVideo.fetch(page, http_client=client)
        picked = _worst_youtube_stream(list(fresh.formats))
        if picked is None:
            raise RuntimeError(f"ydpy не нашёл поток для {video.video_id}")
        holder["url"] = str(picked.url)
        size = int(getattr(picked, "filesize", None) or 0)
        if size:
            holder["total"] = size
        lg.info("Обновил ссылку на файл {}", video.video_id)
        return holder["url"]

    def report(done: int, total: int | None) -> None:
        tracker.hook(
            {
                "status": "finished" if total and done >= total else "downloading",
                "downloaded_bytes": done,
                "total_bytes": total,
                "filename": path.name,
                "info_dict": {
                    "format_id": str(getattr(chosen, "itag", "audio")),
                    "vcodec": "none",
                    "acodec": "opus",
                },
            }
        )

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            _pull_youtube_file(holder, part, report, refresh)
        if not part.is_file() or part.stat().st_size <= 0:
            raise FileNotFoundError(f"ydpy не сохранил файл для {video.video_id}")
        path.unlink(missing_ok=True)
        part.replace(path)
    finally:
        tracker.close()
    lg.info(
        "ydpy сохранил {} ({} байт, itag {})",
        path.name,
        path.stat().st_size,
        getattr(chosen, "itag", "?"),
    )
    return path


def download_worst_quality(
    video: ChannelVideo,
    output_dir: Path = VIDEO_DIR,
    cookies: str | None = None,
    quiet: bool = False,
    bar_position: int = 0,
    fragment_threads: int = DEFAULT_FRAGMENT_THREADS,
) -> Path:
    if _is_youtube(video):
        return _download_youtube_worst(
            video,
            output_dir,
            quiet=quiet,
            bar_position=bar_position,
        )
    last_error: Exception | None = None
    ensure_po_token_server()
    for attempt in range(3):
        _wait_bot_pause()
        try:
            return _save_media(
                video,
                output_dir,
                cookies=cookies,
                quiet=quiet,
                bar_position=bar_position,
                fragment_threads=fragment_threads,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            _remove_partials(output_dir, video.video_id)
            message = brief_ydl_error(exc)
            folded = message.casefold()
            if (
                "no video formats" in folded
                or "format is not available" in folded
                or _is_bot_wall(message)
            ):
                if _is_bot_wall(message):
                    _note_bot_wall()
                break
            if attempt == 2:
                break
            import time

            time.sleep(5 * (attempt + 1))
            lg.warning(
                "Скачивание {} не вышло (попытка {}): {}",
                video.video_id,
                attempt + 1,
                message,
            )
    assert last_error is not None
    raise last_error


def _save_media(
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
                except Exception as exc:  # noqa: BLE001
                    lg.warning(
                        "Параллельная закачка {} не вышла ({}), одно соединение",
                        video.video_id,
                        brief_ydl_error(exc),
                    )
                    Path(filename).unlink(missing_ok=True)
                    Path(filename).with_name(Path(filename).name + ".part").unlink(
                        missing_ok=True
                    )
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

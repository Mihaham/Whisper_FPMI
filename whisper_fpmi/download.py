from __future__ import annotations

from pathlib import Path

from loguru import logger as lg

from whisper_fpmi.names import sanitize_title
from whisper_fpmi.paths import VIDEO_DIR
from whisper_fpmi.vk import ChannelVideo

# 144p со звуком, если VK отдаёт url144; иначе минимальное видео+аудио.
WORST_FORMAT = (
    "url144/url240/hls-275/"
    "worstvideo*[height<=240]+bestaudio/"
    "worstvideo*+worstaudio/worst"
)
FORMAT_SORT = ["+res", "+size", "+br"]


def _ydl_options(
    video: ChannelVideo,
    output_dir: Path,
    cookies: str | None = None,
) -> dict:
    slug = sanitize_title(video.title)
    template = str(output_dir / f"{video.video_id}_{slug}.%(ext)s")
    options = {
        "format": WORST_FORMAT,
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
            info = client.extract_info(video.vk_url, download=False)
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
) -> Path:
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)
    options = _ydl_options(video, output_dir, cookies)
    if quiet:
        options.update({"quiet": True, "noprogress": True, "no_warnings": True})
    lg.info("Скачиваю {} в худшем качестве: {}", video.video_id, video.title)
    with yt_dlp.YoutubeDL(options) as client:
        info = client.extract_info(video.vk_url, download=True)
        filename = client.prepare_filename(info or {})
    path = Path(filename)
    if path.exists():
        return path
    matches = list(output_dir.glob(f"{video.video_id}_*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"yt-dlp не сохранил файл для {video.video_id}")

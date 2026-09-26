from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import requests
from loguru import logger as lg

from whisper_fpmi.paths import CHANNEL_URL, CHANNEL_URL_VK

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
OWNER_ID_RE = re.compile(r'"owner_id"\s*:\s*(-?\d+)')
SCREEN_RE = re.compile(r"@([^/?#]+)")
VIDEO_HREF_RE = re.compile(
    r"https?://(?:vkvideo\.ru|vk\.com)/(?:video|clip)(-?\d+_\d+)",
    re.IGNORECASE,
)
DURATION_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d+)$")

# Канал лектория: страница vkvideo.ru не отдаёт oid, группа VK — отдаёт.
KNOWN_CHANNELS = {
    "lectorium_fpmi": "-206078025",
}


@dataclass(frozen=True)
class ChannelVideo:
    video_id: str
    title: str
    url: str
    duration: float | None = None

    @property
    def vk_url(self) -> str:
        return f"https://vk.com/video{self.video_id}"


def list_channel_videos(channel_url: str = CHANNEL_URL) -> list[ChannelVideo]:
    errors: list[str] = []
    for loader, label in (
        (_list_with_al_video, "vk al_video"),
        (_list_with_yt_dlp, "yt-dlp"),
    ):
        try:
            videos = loader(channel_url)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {exc}")
            lg.warning("Не удалось получить список через {}: {}", label, exc)
            continue
        if videos:
            lg.info("Получено {} видео через {}", len(videos), label)
            return _unique(videos)
        errors.append(f"{label}: пустой список")
    raise RuntimeError(
        "Не удалось получить список видео канала.\n" + "\n".join(errors)
    )


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "ru-RU,ru;q=0.9",
        }
    )
    return session


def _list_with_al_video(channel_url: str) -> list[ChannelVideo]:
    session = _session()
    page_id = _resolve_oid(session, channel_url)
    lg.info("VK oid канала: {}", page_id)
    videos: list[ChannelVideo] = []
    offset = 0
    seen_keys: set[str] = set()
    while True:
        payload = _al_video_page(session, page_id, offset)
        section = _section_payload(payload)
        items = section.get("list") or []
        total = int(section.get("total") or 0)
        if not items:
            break
        added = 0
        for item in items:
            video = _video_from_list_item(item)
            if not video or video.video_id in seen_keys:
                continue
            seen_keys.add(video.video_id)
            videos.append(video)
            added += 1
        offset += len(items)
        lg.debug("Страница VK: offset={}, всего собрано {}", offset, len(videos))
        if added == 0:
            break
        if total and offset >= total:
            break
    return videos


def _resolve_oid(session: requests.Session, channel_url: str) -> str:
    screen = _screen_name(channel_url) or "lectorium_fpmi"
    if screen in KNOWN_CHANNELS:
        return KNOWN_CHANNELS[screen]
    response = session.get(f"https://vk.com/{screen}", timeout=60)
    response.raise_for_status()
    match = OWNER_ID_RE.search(response.text)
    if match:
        return match.group(1)
    raise RuntimeError(f"Не удалось найти owner_id для {screen}")


def _screen_name(channel_url: str) -> str | None:
    match = SCREEN_RE.search(channel_url)
    if match:
        return match.group(1)
    match = re.search(r"vk\.com/([A-Za-z0-9_.]+)", channel_url)
    if match and match.group(1) not in {"video", "videos", "clip"}:
        return match.group(1)
    return None


def _list_with_yt_dlp(channel_url: str) -> list[ChannelVideo]:
    import yt_dlp

    videos: list[ChannelVideo] = []
    urls = _candidate_playlist_urls(channel_url)
    last_error: Exception | None = None
    for url in urls:
        try:
            with yt_dlp.YoutubeDL(
                {
                    "extract_flat": "in_playlist",
                    "quiet": True,
                    "no_warnings": True,
                    "skip_download": True,
                    "ignoreerrors": True,
                }
            ) as client:
                info = client.extract_info(url, download=False)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        entries = (info or {}).get("entries") or []
        for entry in entries:
            if not entry:
                continue
            video_id = str(entry.get("id") or "")
            title = (entry.get("title") or video_id).strip()
            webpage = entry.get("url") or entry.get("webpage_url")
            if not video_id:
                continue
            if "_" not in video_id and webpage:
                found = VIDEO_HREF_RE.search(str(webpage))
                if found:
                    video_id = found.group(1)
            videos.append(
                ChannelVideo(
                    video_id=video_id,
                    title=title,
                    url=webpage or f"https://vk.com/video{video_id}",
                    duration=entry.get("duration"),
                )
            )
        if videos:
            return videos
    if last_error and not videos:
        raise last_error
    return videos


def _al_video_page(session: requests.Session, page_id: str, offset: int) -> list:
    data = {
        "act": "load_videos_silent",
        "al": "1",
        "offset": str(offset),
        "oid": str(page_id),
        "section": "all",
    }
    response = session.post(
        "https://vk.com/al_video.php",
        data=data,
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://vk.com/al_video.php",
        },
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    return body["payload"]


def _section_payload(payload: list) -> dict:
    if not payload:
        raise RuntimeError("Пустой ответ al_video")
    block = payload[1] if len(payload) > 1 else payload[0]
    if isinstance(block, list):
        for item in block:
            if isinstance(item, dict) and "list" in item:
                return item
            if isinstance(item, dict) and "all" in item and isinstance(item["all"], dict):
                return item["all"]
        block = block[0] if block else {}
    if isinstance(block, dict):
        if "all" in block and isinstance(block["all"], dict):
            return block["all"]
        if "list" in block:
            return block
    raise RuntimeError("Неожиданный формат ответа al_video")


def _video_from_list_item(item: list) -> ChannelVideo | None:
    if not isinstance(item, (list, tuple)) or len(item) < 4:
        return None
    owner_id, video_id = item[0], item[1]
    video_key = f"{owner_id}_{video_id}"
    title = str(item[3]).strip() if item[3] else video_key
    duration = _parse_duration(item[5] if len(item) > 5 else None)
    return ChannelVideo(
        video_id=video_key,
        title=title,
        url=f"https://vk.com/video{video_key}",
        duration=duration,
    )


def _parse_duration(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    match = DURATION_RE.match(value.strip())
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def _candidate_playlist_urls(channel_url: str) -> list[str]:
    urls = [
        channel_url,
        channel_url.rstrip("/") + "/all",
        CHANNEL_URL,
        CHANNEL_URL_VK,
        "https://vkvideo.ru/@lectorium_fpmi/all",
        "https://vk.com/video/@lectorium_fpmi",
        "https://vk.com/videos-206078025",
    ]
    unique: list[str] = []
    for url in urls:
        if url not in unique:
            unique.append(url)
    return unique


def _unique(videos: Iterable[ChannelVideo]) -> list[ChannelVideo]:
    seen: set[str] = set()
    result: list[ChannelVideo] = []
    for video in videos:
        if video.video_id in seen:
            continue
        seen.add(video.video_id)
        result.append(video)
    return result

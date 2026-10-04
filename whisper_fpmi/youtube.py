from __future__ import annotations

from whisper_fpmi.paths import YOUTUBE_URL
from whisper_fpmi.vk import ChannelVideo


def list_youtube_videos(channel_url: str = YOUTUBE_URL) -> list[ChannelVideo]:
    import yt_dlp
    from loguru import logger as lg

    options = {
        "quiet": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "ignoreerrors": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(options) as client:
        info = client.extract_info(channel_url, download=False) or {}
    videos: list[ChannelVideo] = []
    seen: set[str] = set()
    for entry in info.get("entries") or []:
        if not entry:
            continue
        video_id = str(entry.get("id") or "").strip()
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        title = str(entry.get("title") or video_id).strip()
        url = entry.get("url") or entry.get("webpage_url") or ""
        if not str(url).startswith("http"):
            url = f"https://www.youtube.com/watch?v={video_id}"
        duration = entry.get("duration")
        videos.append(
            ChannelVideo(
                video_id=video_id,
                title=title,
                url=str(url),
                duration=float(duration) if duration else None,
            )
        )
    if not videos:
        raise RuntimeError(f"YouTube не отдал видео для {channel_url}")
    lg.info("YouTube: {} видео", len(videos))
    return videos

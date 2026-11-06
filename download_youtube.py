"""Докачивает только замороженный список data/youtube-queue.json.

Новые ролики канала сюда не добавляются. Расшифровку скрипт не запускает.
Повторный запуск пропускает файлы, которые уже лежат в videos/.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger as lg
from tqdm import tqdm

from whisper_fpmi.download import download_worst_quality
from whisper_fpmi.paths import DATA_DIR, MEDIA_SUFFIXES, VIDEO_DIR
from whisper_fpmi.vk import ChannelVideo

QUEUE_PATH = DATA_DIR / "youtube-queue.json"
FAILED_PATH = DATA_DIR / "youtube-queue-failed.json"


def _setup_logging() -> None:
    lg.remove()
    lg.add(
        lambda message: tqdm.write(message.rstrip("\n")),
        format="<level>{time:HH:mm:ss} | {level} | {message}</level>",
        level="INFO",
        colorize=True,
    )


def _load_queue() -> list[dict]:
    payload = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    videos = payload.get("videos") or []
    if not videos:
        raise RuntimeError(f"В {QUEUE_PATH} нет роликов")
    return videos


def _already_downloaded(video_id: str) -> Path | None:
    if not VIDEO_DIR.exists():
        return None
    prefix = f"{video_id}_"
    for path in VIDEO_DIR.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        if ".part" in path.name.casefold():
            continue
        if path.stat().st_size <= 0:
            continue
        if path.stem == video_id or path.stem.startswith(prefix):
            return path
    return None


def main() -> int:
    _setup_logging()
    videos = _load_queue()
    failed: list[dict] = []
    downloaded = 0
    skipped = 0
    lg.info("В списке {} старых роликов YouTube. Новые с канала не беру", len(videos))
    for index, item in enumerate(videos, start=1):
        video_id = str(item["video_id"])
        title = str(item.get("title") or video_id)
        found = _already_downloaded(video_id)
        if found is not None:
            skipped += 1
            lg.info("[{}/{}] уже на диске {}", index, len(videos), found.name)
            continue
        video = ChannelVideo(
            video_id=video_id,
            title=title,
            url=str(item.get("url") or f"https://www.youtube.com/watch?v={video_id}"),
            duration=item.get("duration"),
        )
        lg.info("[{}/{}] {}", index, len(videos), title)
        try:
            path = download_worst_quality(video, VIDEO_DIR)
        except Exception as exc:  # noqa: BLE001
            failed.append({"video_id": video_id, "title": title, "url": video.url, "error": str(exc)})
            FAILED_PATH.write_text(
                json.dumps(failed, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            lg.error("[{}/{}] не скачалось {}: {}", index, len(videos), video_id, exc)
            continue
        downloaded += 1
        lg.info("[{}/{}] готово {}", index, len(videos), path.name)
    if not failed and FAILED_PATH.is_file():
        FAILED_PATH.unlink()
    lg.info(
        "Готово. Скачано {}, уже было {}, ошибок {}",
        downloaded,
        skipped,
        len(failed),
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

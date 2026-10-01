"""Описания роликов VK: кэш и разбор ответа al_video."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

import requests
from loguru import logger as lg
from tqdm import tqdm

from whisper_fpmi.timecodes import html_to_text, parse_description_timecodes, parse_lecturer
from whisper_fpmi.vk import USER_AGENT

_DESC_KEYS = {"desc", "description"}


def load_cache(path: Path) -> dict:
    if not path.is_file():
        return {"videos": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"videos": {}}
    data.setdefault("videos", {})
    return data


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def cached_text(cache: dict, video_id: str) -> str | None:
    """None — записи нет или прошлая попытка упала и её надо повторить."""
    item = cache.get("videos", {}).get(video_id)
    if not item:
        return None
    if item.get("error"):
        return None
    return str(item.get("description") or "")


def fetch_descriptions(
    video_ids: list[str],
    cache_path: Path,
    *,
    workers: int = 4,
    refresh: bool = False,
) -> dict[str, str]:
    cache = load_cache(cache_path)
    if refresh:
        missing = list(dict.fromkeys(video_ids))
    else:
        missing = [video_id for video_id in dict.fromkeys(video_ids) if cached_text(cache, video_id) is None]
    if missing:
        lg.info("Качаю описания VK: {} из {}", len(missing), len(set(video_ids)))
        _fetch_into(cache, missing, workers, cache_path)
        save_cache(cache_path, cache)
    texts: dict[str, str] = {}
    for video_id in video_ids:
        item = cache.get("videos", {}).get(video_id) or {}
        texts[video_id] = str(item.get("description") or "")
    return texts


def description_from_payload(payload: object) -> str:
    candidates: list[str] = []
    _collect_strings(payload, candidates)
    cleaned = [html_to_text(item) for item in candidates]
    cleaned = [item.strip() for item in cleaned if item.strip()]
    signaled = [
        item
        for item in cleaned
        if parse_description_timecodes(item) or parse_lecturer(item)
    ]
    pool = signaled or [item for item in cleaned if len(item) > 40]
    if not pool:
        return ""
    return max(pool, key=len)


def _collect_strings(node: object, found: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _DESC_KEYS and isinstance(value, str) and len(value) < 20000:
                found.append(value)
            else:
                _collect_strings(value, found)
        return
    if isinstance(node, list):
        for item in node:
            _collect_strings(item, found)


def _fetch_into(cache: dict, video_ids: list[str], workers: int, cache_path: Path) -> None:
    videos = cache.setdefault("videos", {})
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        futures = {pool.submit(_fetch_one, video_id): video_id for video_id in video_ids}
        done = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="описания", unit="видео"):
            video_id = futures[future]
            text, error = future.result()
            videos[video_id] = {
                "description": text,
                "error": error,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            done += 1
            if done % 40 == 0:
                try:
                    save_cache(cache_path, cache)
                except OSError as exc:
                    lg.warning("Не записал кэш описаний: {}", exc)


def _fetch_one(video_id: str) -> tuple[str, str | None]:
    last_error = "пусто"
    for _attempt in range(2):
        try:
            return _request_description(video_id), None
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = str(exc)
    return "", last_error


def _request_description(video_id: str) -> str:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "ru-RU,ru;q=0.9",
        }
    )
    response = session.post(
        "https://vk.com/al_video.php",
        data={"act": "show", "al": "1", "video": video_id},
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://vk.com/video{video_id}",
        },
        timeout=25,
    )
    response.raise_for_status()
    payload = _decode_vk(response.content)
    block = payload.get("payload")
    if isinstance(block, list) and block and block[0] in (3, "3"):
        raise ValueError("VK просит логин")
    return description_from_payload(payload)


def _decode_vk(content: bytes) -> dict:
    for encoding in ("utf-8", "windows-1251"):
        try:
            return json.loads(content.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError("Не разобрали ответ VK")

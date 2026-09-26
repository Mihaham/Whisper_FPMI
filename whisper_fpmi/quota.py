from __future__ import annotations

from pathlib import Path

from whisper_fpmi.paths import MEDIA_SUFFIXES, VIDEO_DIR

GIB = 1024 ** 3


def bytes_from_gb(gigabytes: float) -> int:
    return int(gigabytes * GIB)


def format_gib(size: int) -> str:
    return f"{size / GIB:.2f} ГиБ"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def media_files(path: Path = VIDEO_DIR) -> list[Path]:
    if not path.exists():
        return []
    files = [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in MEDIA_SUFFIXES
    ]
    files.sort()
    return files


def can_add_to_batch(
    used: int,
    extra: int,
    max_bytes: int,
    batch_count: int,
) -> bool:
    """Пустую пачку всегда начинаем, даже если один файл больше лимита."""
    if max_bytes <= 0:
        return batch_count == 0
    if batch_count == 0:
        return True
    if extra <= 0:
        return used < max_bytes
    return used + extra <= max_bytes

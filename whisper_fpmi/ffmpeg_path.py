"""Найти ffmpeg, если PATH сессии устарел (часто в Cursor после winget)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_DONE = False


def ensure_ffmpeg_path() -> str | None:
    """Вернуть путь к ffmpeg.exe; при необходимости дописать его каталог в PATH."""
    global _DONE
    found = shutil.which("ffmpeg")
    if found:
        _DONE = True
        return found
    if _DONE:
        return shutil.which("ffmpeg")

    _DONE = True
    name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    for bin_dir in _candidate_bins():
        exe = bin_dir / name
        if not exe.is_file():
            continue
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        return str(exe)
    return None


def _candidate_bins() -> list[Path]:
    roots: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        winget = Path(local) / "Microsoft" / "WinGet" / "Packages"
        if winget.is_dir():
            roots.extend(sorted(winget.glob("Gyan.FFmpeg*"), reverse=True))
    roots.extend(
        (
            Path(r"C:\ffmpeg"),
            Path(r"C:\Program Files\ffmpeg"),
            Path(r"C:\Program Files\Gyan\FFmpeg"),
        )
    )

    bins: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for match in root.rglob("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"):
            bin_dir = match.parent
            if bin_dir in seen:
                continue
            seen.add(bin_dir)
            bins.append(bin_dir)
            break
    return bins

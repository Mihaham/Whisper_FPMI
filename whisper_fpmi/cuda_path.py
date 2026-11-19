"""На Windows pip-пакеты NVIDIA кладут DLL в site-packages, а не в PATH."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_DONE = False
_PINNED = False
# В машине две карты. Расшифровка по умолчанию идёт на 4060 Ti, не на 2060.
PREFERRED_GPU = "4060"

_NVIDIA_PKGS = (
    "nvidia.cublas",
    "nvidia.cuda_nvrtc",
    "nvidia.cuda_runtime",
    "nvidia.cudnn",
)


def parse_gpu_list(text: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for line in text.splitlines():
        if "," not in line:
            continue
        index, name = line.split(",", 1)
        index = index.strip()
        if index.isdigit():
            rows.append((int(index), name.strip()))
    return rows


def pick_gpu(rows: list[tuple[int, str]], needle: str = PREFERRED_GPU) -> tuple[int, str] | None:
    folded = needle.casefold()
    for index, name in rows:
        if folded in name.casefold():
            return index, name
    return None


def pin_preferred_gpu() -> None:
    """Оставить процессу только 4060 Ti, если пользователь сам карту не выбрал."""
    global _PINNED
    if _PINNED:
        return
    _PINNED = True
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    picked = pick_gpu(parse_gpu_list(completed.stdout or ""))
    if picked is None:
        return
    index, name = picked
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    os.environ["WHISPER_GPU_NAME"] = name


def ensure_cuda_dll_path() -> None:
    """Добавить bin/ CUDA-библиотек в DLL search path (нужно для cublas64_12.dll)."""
    pin_preferred_gpu()
    global _DONE
    if _DONE or sys.platform != "win32":
        return
    _DONE = True

    bins: list[Path] = []
    for name in _NVIDIA_PKGS:
        try:
            mod = __import__(name, fromlist=["__path__"])
        except ImportError:
            continue
        for root in getattr(mod, "__path__", []):
            bin_dir = Path(root) / "bin"
            if bin_dir.is_dir():
                bins.append(bin_dir)

    if not bins:
        return

    path_prefix = os.pathsep.join(str(p) for p in bins)
    os.environ["PATH"] = path_prefix + os.pathsep + os.environ.get("PATH", "")
    add = getattr(os, "add_dll_directory", None)
    if add is not None:
        for bin_dir in bins:
            add(str(bin_dir))

"""На Windows pip-пакеты NVIDIA кладут DLL в site-packages, а не в PATH."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DONE = False

_NVIDIA_PKGS = (
    "nvidia.cublas",
    "nvidia.cuda_nvrtc",
    "nvidia.cuda_runtime",
    "nvidia.cudnn",
)


def ensure_cuda_dll_path() -> None:
    """Добавить bin/ CUDA-библиотек в DLL search path (нужно для cublas64_12.dll)."""
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

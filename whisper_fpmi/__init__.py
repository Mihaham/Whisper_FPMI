"""Расшифровка лекций ФПМИ с канала VK Video."""

from whisper_fpmi.cuda_path import ensure_cuda_dll_path
from whisper_fpmi.ffmpeg_path import ensure_ffmpeg_path

ensure_cuda_dll_path()
ensure_ffmpeg_path()

__version__ = "2.0.0"

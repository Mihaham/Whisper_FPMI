from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESULT_DIR = ROOT / "result"
TIMED_DIR = ROOT / "download"
VIDEO_DIR = ROOT / "videos"
DATA_DIR = ROOT / "data"
STATE_PATH = DATA_DIR / "processed.json"
CATALOG_PATH = ROOT / "CATALOG.md"

CHANNEL_URL = "https://vkvideo.ru/@lectorium_fpmi"
CHANNEL_URL_VK = "https://vk.com/video/@lectorium_fpmi/all"

DEFAULT_MODEL = "large-v3"
LANGUAGE = "ru"
DEFAULT_MAX_GB = 10.0
# Из 28 потоков машины: Whisper на CPU, остальное — ffmpeg/yt-dlp/система.
DEFAULT_CPU_CORES = 20
DEFAULT_MAX_CPU_WORKERS = 4
MEDIA_SUFFIXES = {".mp4", ".mkv", ".webm", ".m4a", ".mp3", ".wav", ".m4v"}

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESULT_DIR = ROOT / "result"
TIMED_DIR = ROOT / "download"
VIDEO_DIR = ROOT / "videos"
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
STATE_PATH = DATA_DIR / "processed.json"
CATALOG_PATH = ROOT / "CATALOG.md"
LECTURES_PATH = DATA_DIR / "lectures.jsonl"
LECTURES_CSV = DATA_DIR / "lectures.csv"
DESCRIPTIONS_PATH = DATA_DIR / "descriptions.json"
WORDCLOUD_SVG = ASSETS_DIR / "wordcloud.svg"
WORDCLOUD_JSON = DATA_DIR / "wordcloud.json"
WORDCLOUD_CACHE = DATA_DIR / "wordcloud-cache.json"
TERM_GRAPH_HTML = ASSETS_DIR / "term-graph.html"
SEARCH_INDEX_PATH = DATA_DIR / "search.pkl"
METRICS_PATH = DATA_DIR / "metrics.json"

CHANNEL_URL = "https://vkvideo.ru/@lectorium_fpmi"
CHANNEL_URL_VK = "https://vk.com/video/@lectorium_fpmi/all"

DEFAULT_MODEL = "large-v3"
LANGUAGE = "ru"
DEFAULT_MAX_GB = 3.0
# Сколько лекций качать одновременно и сколько HTTP-соединений на один ролик.
DEFAULT_DOWNLOAD_WORKERS = 4
DEFAULT_FRAGMENT_THREADS = 8
# Из 28 потоков машины: запас системе, если CPU-расшифровка включена явно.
DEFAULT_CPU_CORES = 20
DEFAULT_MAX_CPU_WORKERS = 4
# Рядом с GPU: один процесс, 10 ядер, до 10 видео в очереди. GPU его не ждёт.
DEFAULT_CPU_LANE_THREADS = 10
DEFAULT_CPU_LANE_QUEUE = 10
MEDIA_SUFFIXES = {".mp4", ".mkv", ".webm", ".m4a", ".mp3", ".wav", ".m4v"}

"""Порядок шортсов по ожидаемым просмотрам и отложенная публикация на YouTube.

Ролик — это «каждый раз, когда лектор говорит слово». В ленте Shorts его
досматривают, если слово узнаётся за секунду: школьная математика, боль
доказательства или узнаваемая оговорка лектора. Служебные слова листают сразу.
Следующие части того же слова слабее: шутка уже была.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import re
import subprocess

from loguru import logger as lg

from whisper_fpmi.paths import (
    SHORTS_SCHEDULE_PATH,
    YOUTUBE_CLIENT_PATH,
    YOUTUBE_TOKEN_PATH,
)
from whisper_fpmi.shorts import POPULAR_DIR

# Москва без перехода на летнее время.
MSK = timezone(timedelta(hours=3))
PUBLISH_WEEKDAYS = {1, 4}  # вторник, пятница
PUBLISH_HOUR = 10
# Ближе этого к эфиру YouTube уже не даёт поставить publishAt.
MIN_LEAD = timedelta(minutes=20)
PART_DECAY = 0.78
MIN_SECONDS = 20.0
# Короткий обрывок досматривают хуже целого ролика.
SHORT_SECONDS = 30.0
SHORT_PENALTY = 0.7
DEFAULT_TOPIC = 15
# Те же права, что уже выданы токену канала Mihaham.
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube.readonly",
]
UPLOAD_CHUNK = 8 * 1024 * 1024
CHANNEL_HANDLE = "mihaham"
# Этот адрес прописан в OAuth-клиенте, другой Google не примет.
OAUTH_REDIRECT = "http://localhost:8000/youtube/oauth/callback"
# Education. Howto & Style и Film & Animation уводят лекционный шортс не в ту ленту.
CATEGORY_ID = "27"
_NAME = re.compile(r"^(?P<word>.+)-(?P<part>\d+)$")

# Ожидаемый охват первой части, 0–100. Дальше множитель части.
TOPIC_SCORE = {
    "интеграл": 96,
    "вероятность": 93,
    "теорема": 90,
    "матрица": 88,
    "формула": 86,
    "функция": 84,
    "доказать": 82,
    "наверное": 79,
    "определение": 77,
    "задача": 75,
    "вектор": 73,
    "пример": 72,
    "множество": 70,
    "дерево": 68,
    "квадрат": 66,
    "пространство": 64,
    "сумма": 63,
    "случай": 62,
    "корень": 61,
    "модуль": 59,
    "степень": 58,
    "штука": 56,
    "число": 52,
    "точка": 50,
    "вершина": 47,
    "ребро": 45,
    "единица": 40,
    "равный": 36,
    "значение": 32,
    "простой": 30,
    "существовать": 28,
    "лежать": 26,
    "правильно": 24,
    "время": 22,
    "вопрос": 21,
    "работать": 20,
    "хороший": 18,
    "часть": 17,
    "образ": 16,
    "пока": 15,
    "получиться": 14,
    "брать": 13,
    "каждый": 13,
    "самый": 12,
    "разный": 11,
    "далее": 10,
    "следующий": 10,
    "другой": 9,
    "любой": 9,
    "иметь": 8,
    "более": 8,
    "некоторый": 7,
    "далекий": 7,
    "должный": 7,
}


@dataclass(frozen=True)
class ShortFile:
    name: str
    path: Path
    word: str
    part: int
    duration: float


def parse_short_name(stem: str) -> tuple[str, int] | None:
    match = _NAME.fullmatch(stem)
    if not match:
        return None
    part = int(match.group("part"))
    word = match.group("word").strip()
    if part < 1 or not word:
        return None
    return word, part


def view_score(word: str, part: int, duration: float) -> float | None:
    """Чем выше, тем раньше ролик выходит. Слишком короткий ролик не ставим."""
    if part < 1 or duration < MIN_SECONDS:
        return None
    topic = TOPIC_SCORE.get(word.casefold(), DEFAULT_TOPIC)
    score = topic * (PART_DECAY ** (part - 1))
    if duration < SHORT_SECONDS:
        score *= SHORT_PENALTY
    return round(score, 2)


def short_title(word: str, part: int) -> str:
    title = f"Каждый раз, когда лектор говорит «{word.upper()}»"
    if part > 1:
        title = f"{title} · {part}"
    return title[:100]


def short_hashtag(word: str) -> str:
    return re.sub(r"\s+", "", word.casefold())


def _plural(word: str) -> str | None:
    """Именительный множественного, чтобы тег ловил «интегралы» рядом с «интеграл»."""
    from pymorphy3 import MorphAnalyzer

    morph = getattr(_plural, "_morph", None)
    if morph is None:
        morph = MorphAnalyzer()
        setattr(_plural, "_morph", morph)
    form = morph.parse(word)[0].inflect({"plur", "nomn"})
    if form is None:
        return None
    text = form.word.casefold()
    if text == word.casefold():
        return None
    return text


def short_description(word: str) -> str:
    # Первые три хештега YouTube показывает над названием, поэтому там само слово.
    return (
        f"Нарезка лекций ФПМИ: каждый раз, когда лектор говорит «{word}».\n\n"
        f"#{short_hashtag(word)} #лекция #мфти #shorts"
    )


def short_tags(word: str) -> list[str]:
    tags = [word.casefold()]
    plural = _plural(word)
    if plural and plural not in tags:
        tags.append(plural)
    for extra in ("лекция", "мфти", "фпми", "математика"):
        if extra not in tags:
            tags.append(extra)
    return tags


def publish_slots(count: int, now: datetime) -> list[datetime]:
    """Ближайшие вторники и пятницы в 10:00 по Москве, которые ещё успеваем поставить."""
    if count < 1:
        return []
    moment = now.astimezone(MSK) if now.tzinfo else now.replace(tzinfo=MSK)
    earliest = moment + MIN_LEAD
    slots: list[datetime] = []
    day = earliest.date()
    while len(slots) < count:
        if day.weekday() in PUBLISH_WEEKDAYS:
            slot = datetime(day.year, day.month, day.day, PUBLISH_HOUR, 0, tzinfo=MSK)
            if slot >= earliest:
                slots.append(slot)
        day += timedelta(days=1)
    return slots


def video_duration(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout or "{}")
        return float((payload.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0


def scan_shorts(folder: Path = POPULAR_DIR) -> tuple[list[ShortFile], list[dict]]:
    paths = sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.name,
    )
    durations: dict[Path, float] = {}
    if paths:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for path, duration in zip(paths, pool.map(video_duration, paths)):
                durations[path] = duration
    files: list[ShortFile] = []
    skipped: list[dict] = []
    for path in paths:
        parsed = parse_short_name(path.stem)
        duration = durations.get(path, 0.0)
        if parsed is None:
            skipped.append({"file": path.name, "reason": "имя не вида слово-номер"})
            continue
        word, part = parsed
        if view_score(word, part, duration) is None:
            skipped.append({"file": path.name, "reason": f"короче {MIN_SECONDS:.0f} с"})
            continue
        files.append(ShortFile(path.name, path, word, part, duration))
    return files, skipped


def plan_uploads(files: list[ShortFile], now: datetime) -> list[dict]:
    by_word: dict[str, list[ShortFile]] = {}
    for item in files:
        if view_score(item.word, 1, item.duration) is None:
            continue
        by_word.setdefault(item.word, []).append(item)
    scored: list[tuple[float, int, ShortFile]] = []
    for word, items in by_word.items():
        # Номер в имени — это кусок нарезки. Для зрителя счёт идёт по целым роликам.
        for edition, item in enumerate(sorted(items, key=lambda entry: entry.part), start=1):
            score = view_score(word, edition, item.duration)
            if score is None:
                continue
            scored.append((score, edition, item))
    scored.sort(key=lambda triple: (-triple[0], triple[1], triple[2].word, triple[2].name))
    slots = publish_slots(len(scored), now)
    rows: list[dict] = []
    for rank, ((score, edition, item), slot) in enumerate(zip(scored, slots), start=1):
        rows.append(
            {
                "rank": rank,
                "file": item.name,
                "word": item.word,
                "part": item.part,
                "edition": edition,
                "score": score,
                "seconds": round(item.duration, 1),
                "title": short_title(item.word, edition),
                "description": short_description(item.word),
                "publish_at": slot.isoformat(),
                "status": "pending",
                "youtube_id": None,
            }
        )
    return rows


def load_schedule(path: Path = SHORTS_SCHEDULE_PATH) -> dict:
    if not path.is_file():
        return {"channel": CHANNEL_HANDLE, "videos": [], "skipped": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("videos", [])
    payload.setdefault("skipped", [])
    return payload


def merge_plan(existing: list[dict], fresh: list[dict]) -> list[dict]:
    """Уже залитые ролики остаются на своих датах. Остальные заново встают в очередь."""
    locked = [dict(row) for row in existing if row.get("youtube_id")]
    locked_files = {row["file"] for row in locked}
    taken = {row["publish_at"] for row in locked}
    pending = [dict(row) for row in fresh if row["file"] not in locked_files]
    free = [row["publish_at"] for row in fresh if row["publish_at"] not in taken]
    # Если часть слотов занята заливкой, добираем даты дальше по той же сетке.
    if len(free) < len(pending) and pending:
        now = datetime.now(MSK)
        extra = [
            slot.isoformat()
            for slot in publish_slots(len(locked) + len(pending) + 8, now)
            if slot.isoformat() not in taken and slot.isoformat() not in set(free)
        ]
        free.extend(extra)
    for index, row in enumerate(pending):
        row["publish_at"] = free[index]
        row["status"] = "pending"
        row["youtube_id"] = None
    combined = locked + pending
    combined.sort(key=lambda row: row["publish_at"])
    for rank, row in enumerate(combined, start=1):
        row["rank"] = rank
    return combined


def write_schedule(
    videos: list[dict],
    skipped: list[dict],
    path: Path = SHORTS_SCHEDULE_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "channel": CHANNEL_HANDLE,
        "timezone": "Europe/Moscow",
        "weekdays": ["tuesday", "friday"],
        "hour": PUBLISH_HOUR,
        "videos": videos,
        "skipped": skipped,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_schedule(
    folder: Path = POPULAR_DIR,
    path: Path = SHORTS_SCHEDULE_PATH,
    now: datetime | None = None,
) -> dict:
    moment = now or datetime.now(MSK)
    files, skipped = scan_shorts(folder)
    fresh = plan_uploads(files, moment)
    existing = load_schedule(path).get("videos") or []
    videos = merge_plan(existing, fresh) if existing else fresh
    write_schedule(videos, skipped, path)
    return {"videos": videos, "skipped": skipped}


def publish_at_utc(value: str) -> str:
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=MSK)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _as_msk(value: str) -> datetime:
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        return moment.replace(tzinfo=MSK)
    return moment.astimezone(MSK)


def _shift_past_slots(videos: list[dict], now: datetime) -> None:
    """Прошедший слот ещё не залитого ролика переносится на свободные вторник или пятницу."""
    earliest = (now.astimezone(MSK) if now.tzinfo else now.replace(tzinfo=MSK)) + MIN_LEAD
    taken: set[str] = set()
    for row in videos:
        if row.get("youtube_id") or _as_msk(row["publish_at"]) >= earliest:
            taken.add(row["publish_at"])
    horizon = publish_slots(len(videos) + 16, now)
    free = [slot.isoformat() for slot in horizon if slot.isoformat() not in taken]
    cursor = 0
    for row in videos:
        if row.get("youtube_id") or _as_msk(row["publish_at"]) >= earliest:
            continue
        while cursor < len(free) and free[cursor] in taken:
            cursor += 1
        if cursor >= len(free):
            break
        row["publish_at"] = free[cursor]
        taken.add(free[cursor])
        cursor += 1


def wait_for_oauth(
    client_path: Path = YOUTUBE_CLIENT_PATH,
    token_path: Path = YOUTUBE_TOKEN_PATH,
) -> str:
    """Печатает ссылку входа и ждёт, пока браузер вернётся на localhost."""
    import os
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build

    flow = Flow.from_client_secrets_file(
        str(client_path),
        scopes=YOUTUBE_SCOPES,
        redirect_uri=OAUTH_REDIRECT,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    holder: dict[str, str | None] = {"state": state, "done": None, "error": None}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/youtube/oauth/callback":
                self._send(404, "Не тот адрес")
                return
            query = parse_qs(parsed.query)
            if query.get("error"):
                holder["error"] = query["error"][0]
                self._send(400, "Google не выдал доступ. Окно можно закрыть.")
                return
            code = (query.get("code") or [""])[0]
            got_state = (query.get("state") or [""])[0]
            if not code or got_state != holder["state"]:
                self._send(400, "Код входа не сошёлся. Открой ссылку ещё раз.")
                return
            try:
                flow.fetch_token(code=code)
                youtube = build("youtube", "v3", credentials=flow.credentials)
                title = _require_channel(youtube)
            except Exception as exc:
                holder["error"] = str(exc)
                self._send(400, f"Не подключилось: {exc}")
                return
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(flow.credentials.to_json(), encoding="utf-8")
            holder["done"] = title
            self._send(200, f"Канал {title} подключён. Это окно можно закрыть.")

        def _send(self, status: int, text: str) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    class Server(ThreadingHTTPServer):
        allow_reuse_address = True

        def server_bind(self) -> None:
            if self.address_family == socket.AF_INET6:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            super().server_bind()

    try:
        Server.address_family = socket.AF_INET6
        server = Server(("::", 8000), Handler)
    except OSError:
        Server.address_family = socket.AF_INET
        server = Server(("127.0.0.1", 8000), Handler)
    print(auth_url, flush=True)
    lg.info("Жду подтверждение Google в браузере")
    try:
        while holder["done"] is None and holder["error"] is None:
            server.handle_request()
    finally:
        server.server_close()
    if holder["error"]:
        raise RuntimeError(str(holder["error"]))
    title = str(holder["done"])
    lg.info("Подключён канал {}", title)
    return title


def _youtube_service(client_path: Path, token_path: Path):
    """Берёт уже выданный токен и обновляет его. Новый вход сюда не открывает."""
    if not token_path.is_file():
        raise RuntimeError(
            f"Нет токена {token_path}. Обнови авторизацию YouTube, клиент лежит в {client_path}."
        )
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(str(token_path), YOUTUBE_SCOPES)
    try:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(
            f"Токен YouTube не обновился ({exc}). Обнови авторизацию, файл {token_path}."
        ) from exc
    if not creds.valid:
        raise RuntimeError(f"Токен YouTube недействителен. Обнови авторизацию, файл {token_path}.")
    return build("youtube", "v3", credentials=creds)


def _require_channel(youtube, handle: str = CHANNEL_HANDLE) -> str:
    response = youtube.channels().list(part="snippet", mine=True).execute()
    items = response.get("items") or []
    if not items:
        raise RuntimeError("У этого аккаунта Google нет канала YouTube")
    snippet = items[0].get("snippet") or {}
    custom = str(snippet.get("customUrl") or "").lstrip("@").casefold()
    title = str(snippet.get("title") or "")
    if handle not in {custom, title.casefold()}:
        raise RuntimeError(
            f"Вход выполнен на канал @{custom or title}, а нужен @{handle}"
        )
    return title or handle


def _insert_video(youtube, path: Path, row: dict) -> str:
    from googleapiclient.http import MediaFileUpload

    body = {
        "snippet": {
            "title": row["title"],
            "description": row["description"],
            "tags": short_tags(row["word"]),
            "categoryId": CATEGORY_ID,
            "defaultLanguage": "ru",
            "defaultAudioLanguage": "ru",
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": publish_at_utc(row["publish_at"]),
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=UPLOAD_CHUNK, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            lg.info("{}: залито {:.0f}%", path.name, status.progress() * 100)
    video_id = str((response or {}).get("id") or "")
    if not video_id:
        raise RuntimeError(f"YouTube не вернул id для {path.name}")
    return video_id


def upload_pending(
    folder: Path = POPULAR_DIR,
    path: Path = SHORTS_SCHEDULE_PATH,
    *,
    limit: int = 6,
    client_path: Path = YOUTUBE_CLIENT_PATH,
    token_path: Path = YOUTUBE_TOKEN_PATH,
    now: datetime | None = None,
) -> int:
    """Ставит следующие ролики в отложку. 6 за запуск — дневная квота YouTube Data API."""
    payload = load_schedule(path)
    videos = list(payload.get("videos") or [])
    if not videos:
        lg.error("Сначала собери расписание: python -m whisper_fpmi shorts-rank")
        return 1
    moment = now or datetime.now(MSK)
    _shift_past_slots(videos, moment)
    from googleapiclient.errors import HttpError

    try:
        youtube = _youtube_service(client_path, token_path)
        channel = _require_channel(youtube)
    except HttpError as exc:
        if "quotaExceeded" in str(exc):
            lg.warning("Квота YouTube на сегодня кончилась, залито 0")
            return 2
        raise
    lg.info("Канал: {}", channel)
    done = 0

    for row in sorted(videos, key=lambda item: item["rank"]):
        if done >= limit:
            break
        if row.get("youtube_id"):
            continue
        file_path = folder / row["file"]
        if not file_path.is_file():
            lg.error("Нет файла {}", row["file"])
            row["status"] = "missing"
            continue
        try:
            video_id = _insert_video(youtube, file_path, row)
        except HttpError as exc:
            write_schedule(videos, list(payload.get("skipped") or []), path)
            text = str(exc)
            if "quotaExceeded" in text or "uploadLimitExceeded" in text:
                lg.warning("Квота YouTube на сегодня кончилась, залито {}", done)
                return 0 if done else 2
            lg.error("{}", exc)
            return 1
        row["youtube_id"] = video_id
        row["status"] = "scheduled"
        done += 1
        when = datetime.fromisoformat(row["publish_at"]).strftime("%Y-%m-%d %H:%M")
        lg.info("В отложке {} → {} МСК ({})", row["file"], when, video_id)
        write_schedule(videos, list(payload.get("skipped") or []), path)
    lg.info("Поставлено в отложку: {}", done)
    return 0

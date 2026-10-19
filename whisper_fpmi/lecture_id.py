"""Повтор лекции ищется по дате и лектору, затем по названию.

Одинаковые дата и лектор почти всегда означают одну пару записей.
Название отделяет соседние лекции одного дня: 11-ю от 12-й.
Звук для этого не сравнивается.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import re
import threading

from loguru import logger as lg

from whisper_fpmi.descriptions import (
    cached_text,
    fetch_descriptions,
    fetch_youtube_descriptions,
    load_cache,
)
from whisper_fpmi.names import fold_key
from whisper_fpmi.paths import DESCRIPTIONS_PATH, PUBLISHED_PATH, SAME_LECTURE_PATH
from whisper_fpmi.timecodes import parse_event_date, parse_lecturer
from whisper_fpmi.vk import ChannelVideo

_VK_ID = re.compile(r"^-?\d+_\d+$")
_NUMBER = re.compile(r"(?<!\d)(\d{1,2})(?!\d)")
_WORD = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
_SKIP = frozenset(
    {
        "лекция",
        "лекции",
        "семинар",
        "семинара",
        "поток",
        "базовый",
        "продвинутый",
        "основной",
        "часть",
        "фпми",
        "мфти",
        "допсем",
        "консультация",
    }
)
_INDEX = re.compile(r"(?<!\d)(\d{1,2})\.\s*")
PUBLISH_GAP_DAYS = 14
_LINKS: dict[str, dict] | None = None


@dataclass(frozen=True)
class LectureHit:
    video_id: str
    title: str
    url: str
    slug: str
    lecturer: str
    event_date: str
    plain: str
    timed: str


@dataclass(frozen=True)
class LectureCopy:
    video_id: str
    title: str
    url: str
    slug: str
    same_lecture_as: str
    plain: str
    timed: str


class DescriptionBook:
    def __init__(self, texts: dict[str, str]) -> None:
        self._lock = threading.Lock()
        self._texts = dict(texts)

    def get(self, video_id: str) -> str | None:
        with self._lock:
            if video_id not in self._texts:
                return None
            return self._texts[video_id]

    def put_many(self, items: dict[str, str]) -> None:
        with self._lock:
            self._texts.update(items)


class LectureCatalog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_key: dict[tuple[str, str], list[LectureHit]] = defaultdict(list)
        self._slugs: set[str] = set()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._slugs)

    def add(self, hit: LectureHit) -> None:
        key = (lecturer_key(hit.lecturer), hit.event_date)
        if not key[0] or not hit.event_date:
            return
        with self._lock:
            if hit.slug in self._slugs:
                return
            self._by_key[key].append(hit)
            self._slugs.add(hit.slug)

    def match(self, lecturer: str | None, event_date: str | None, title: str) -> LectureHit | None:
        if not lecturer or not event_date:
            return None
        key = (lecturer_key(lecturer), event_date)
        with self._lock:
            found = list(self._by_key.get(key, ()))
        matches = [item for item in found if same_lecture_title(title, item.title)]
        if not matches:
            return None
        return max(matches, key=lambda item: _shared_titles(title, item.title))

    def match_video(self, video: ChannelVideo, book: DescriptionBook) -> LectureHit | None:
        text = book.get(video.video_id)
        if not text:
            return None
        hit = self.match(parse_lecturer(text), parse_event_date(text), video.title)
        if hit is None or hit.video_id == video.video_id:
            return None
        return hit

    def remember(self, video: ChannelVideo, slug: str, description: str) -> None:
        lecturer = parse_lecturer(description)
        event_date = parse_event_date(description)
        if not lecturer or not event_date:
            return
        plain, timed = _paths(slug)
        self.add(
            LectureHit(
                video_id=video.video_id,
                title=video.title,
                url=video.url,
                slug=slug,
                lecturer=lecturer,
                event_date=event_date,
                plain=plain,
                timed=timed,
            )
        )


def load_description_book(path=DESCRIPTIONS_PATH) -> DescriptionBook:
    cache = load_cache(path)
    texts: dict[str, str] = {}
    for video_id in cache.get("videos") or {}:
        text = cached_text(cache, video_id)
        if text is not None:
            texts[str(video_id)] = text
    return DescriptionBook(texts)


def build_catalog(
    slug_by_fold: dict[str, str],
    state: dict,
    book: DescriptionBook,
    lectures: list[dict],
) -> LectureCatalog:
    catalog = LectureCatalog()
    finals = set(slug_by_fold.values())
    for row in lectures:
        slug = str(row.get("slug") or "")
        if slug not in finals:
            continue
        description = str(row.get("description") or "")
        video_id = str(row.get("video_id") or "")
        cached = book.get(video_id) if video_id else None
        text = cached if cached else description
        lecturer = row.get("lecturer") or parse_lecturer(text)
        event_date = row.get("event_date") or parse_event_date(text)
        if not lecturer or not event_date:
            continue
        plain, timed = _paths(slug)
        catalog.add(
            LectureHit(
                video_id=video_id,
                title=str(row.get("header_title") or row.get("title") or slug),
                url=str(row.get("url") or ""),
                slug=slug,
                lecturer=str(lecturer),
                event_date=str(event_date),
                plain=str(row.get("plain") or plain),
                timed=str(row.get("timed") or timed),
            )
        )
    for video_id, record in (state.get("videos") or {}).items():
        if record.get("same_lecture_as"):
            continue
        slug = str(record.get("slug") or "")
        if slug not in finals:
            continue
        text = book.get(str(video_id)) or ""
        lecturer = parse_lecturer(text)
        event_date = parse_event_date(text)
        if not lecturer or not event_date:
            continue
        plain, timed = _paths(slug)
        catalog.add(
            LectureHit(
                video_id=str(video_id),
                title=str(record.get("title") or slug),
                url=str(record.get("url") or ""),
                slug=slug,
                lecturer=lecturer,
                event_date=event_date,
                plain=str(record.get("plain") or plain),
                timed=str(record.get("timed") or timed),
            )
        )
    return catalog


def prefetch_descriptions(
    videos: list[ChannelVideo],
    book: DescriptionBook,
    stop: threading.Event,
) -> None:
    """Докачивает описания ещё не виденных роликов, чтобы узнать дату и лектора."""
    try:
        vk_ids = [
            video.video_id
            for video in videos
            if _VK_ID.match(video.video_id) and book.get(video.video_id) is None
        ]
        if vk_ids and not stop.is_set():
            fetch_descriptions(vk_ids, DESCRIPTIONS_PATH, workers=4)
            book.put_many(_cached_many(vk_ids))
        youtube = [
            (video.video_id, video.url)
            for video in videos
            if video.url and not _VK_ID.match(video.video_id) and book.get(video.video_id) is None
        ]
        if youtube and not stop.is_set():
            fetch_youtube_descriptions(youtube, DESCRIPTIONS_PATH, workers=4, stop=stop)
            book.put_many(_cached_many([video_id for video_id, _url in youtube]))
    except Exception as exc:  # noqa: BLE001
        lg.warning("Описания для сверки лекций не докачались: {}", exc)


def load_published(path=PUBLISHED_PATH) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if value}


def save_published(dates: dict[str, str], path=PUBLISHED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_same_lectures(path=SAME_LECTURE_PATH) -> dict[str, dict]:
    global _LINKS
    if _LINKS is not None and path == SAME_LECTURE_PATH:
        return _LINKS
    if not path.is_file():
        found: dict[str, dict] = {}
    else:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        videos = raw.get("videos") if isinstance(raw, dict) else {}
        found = {str(key): value for key, value in (videos or {}).items() if isinstance(value, dict)}
    if path == SAME_LECTURE_PATH:
        _LINKS = found
    return found


def linked_slug(video_id: str) -> str | None:
    record = load_same_lectures().get(video_id) or {}
    slug = str(record.get("slug") or "")
    return slug or None


def save_same_lectures(videos: dict[str, dict], path=SAME_LECTURE_PATH) -> None:
    global _LINKS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"videos": videos}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if path == SAME_LECTURE_PATH:
        _LINKS = videos


def fetch_published_dates(
    items: list[tuple[str, str]],
    path=PUBLISHED_PATH,
    *,
    workers: int = 6,
) -> dict[str, str]:
    """Дата публикации YouTube без скачивания ролика. Уже известные не запрашивает."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import yt_dlp

    known = load_published(path)
    missing = [(video_id, url) for video_id, url in items if url and video_id not in known]
    if not missing:
        return known
    lg.info("Даты публикации YouTube: {} из {}", len(missing), len(items))
    from whisper_fpmi.download import _QuietYdl, brief_ydl_error

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 25,
        "logger": _QuietYdl(),
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }

    def one(video_id: str, url: str) -> tuple[str, str | None]:
        try:
            with yt_dlp.YoutubeDL(options) as client:
                info = client.extract_info(url, download=False) or {}
        except Exception as exc:  # noqa: BLE001
            lg.warning(
                "Дата публикации {} не прочиталась: {}",
                video_id,
                brief_ydl_error(exc),
            )
            return video_id, None
        published = iso_upload_date(info.get("upload_date") or info.get("timestamp"))
        return video_id, published

    done = 0
    blocked = 0
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futures = [pool.submit(one, video_id, url) for video_id, url in missing]
        for future in as_completed(futures):
            video_id, published = future.result()
            if published:
                known[video_id] = published
                blocked = 0
            else:
                blocked += 1
            done += 1
            if done % 40 == 0:
                save_published(known, path)
                lg.info("Даты публикации: {}/{}", done, len(missing))
            if blocked >= 12:
                lg.warning("YouTube перестал отдавать даты, останавливаю запросы")
                break
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        save_published(known, path)
    return known


def copies_from_state(state: dict) -> list[LectureCopy]:
    copies: list[LectureCopy] = []
    for video_id, record in (state.get("videos") or {}).items():
        same = str(record.get("same_lecture_as") or "")
        slug = str(record.get("slug") or "")
        if not same or not slug:
            continue
        plain, timed = _paths(slug)
        copies.append(
            LectureCopy(
                video_id=str(video_id),
                title=str(record.get("title") or ""),
                url=str(record.get("url") or ""),
                slug=slug,
                same_lecture_as=same,
                plain=str(record.get("plain") or plain),
                timed=str(record.get("timed") or timed),
            )
        )
    seen = {item.video_id for item in copies}
    for video_id, record in load_same_lectures().items():
        if video_id in seen:
            continue
        slug = str(record.get("slug") or "")
        same = str(record.get("same_lecture_as") or "")
        if not slug or not same:
            continue
        plain, timed = _paths(slug)
        copies.append(
            LectureCopy(
                video_id=video_id,
                title=str(record.get("title") or ""),
                url=str(record.get("url") or ""),
                slug=slug,
                same_lecture_as=same,
                plain=str(record.get("plain") or plain),
                timed=str(record.get("timed") or timed),
            )
        )
    return copies


def lecturer_key(name: str) -> str:
    text = name.casefold().replace("ё", "е")
    parts = re.findall(r"[a-zа-я]{3,}", text)
    return " ".join(sorted(parts))


def lecture_numbers(title: str) -> set[int]:
    found: set[int] = set()
    for item in _NUMBER.findall(title):
        value = int(item)
        if 1 <= value <= 40:
            found.add(value)
    return found


def title_tokens(title: str) -> set[str]:
    words = _WORD.findall(title.casefold().replace("ё", "е"))
    return {word for word in words if len(word) >= 5 and word not in _SKIP}


def same_lecture_title(left: str, right: str) -> bool:
    """Один курс, один номер и та же тема. «Доп. главы» не склеиваются с основным курсом."""
    left_course, left_number, left_rest = split_title(left)
    right_course, right_number, right_rest = split_title(right)
    if not left_course or left_course != right_course or left_number != right_number:
        return False
    if not left_rest and not right_rest:
        return True
    if not left_rest or not right_rest:
        return False
    left_tokens = set(course_key(left_rest).split())
    right_tokens = set(course_key(right_rest).split())
    if not left_tokens or not right_tokens:
        return fold_key(left_rest) == fold_key(right_rest)
    shared = left_tokens & right_tokens
    shorter = min(len(left_tokens), len(right_tokens))
    longer = max(len(left_tokens), len(right_tokens))
    # Короткое название может не дописывать одну фразу, но не другую лекцию.
    return len(shared) == shorter and longer - len(shared) <= 1


def split_title(title: str) -> tuple[str, int | None, str]:
    match = _INDEX.search(title)
    if match is None:
        return course_key(title), None, ""
    number = int(match.group(1))
    if not 1 <= number <= 40:
        return course_key(title), None, ""
    return course_key(title[: match.start()]), number, title[match.end() :]


def course_key(text: str) -> str:
    return " ".join(sorted(title_tokens(text)))


def publish_gap(left: str | None, right: str | None) -> int | None:
    if not left or not right:
        return None
    try:
        left_day = date.fromisoformat(left)
        right_day = date.fromisoformat(right)
    except ValueError:
        return None
    return abs((left_day - right_day).days)


def same_publication(left: str | None, right: str | None, *, days: int = PUBLISH_GAP_DAYS) -> bool:
    gap = publish_gap(left, right)
    return gap is not None and gap <= days


def iso_upload_date(value: object) -> str | None:
    if isinstance(value, (int, float)) and int(value) > 1_000_000_000:
        return datetime.fromtimestamp(int(value), timezone.utc).date().isoformat()
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return text
    return None


def _shared_titles(left: str, right: str) -> int:
    return len(title_tokens(left) & title_tokens(right))


def _paths(slug: str) -> tuple[str, str]:
    return f"result/{slug}.txt", f"download/{slug}.mp4.txt"


def _cached_many(video_ids: list[str]) -> dict[str, str]:
    cache = load_cache(DESCRIPTIONS_PATH)
    found: dict[str, str] = {}
    for video_id in video_ids:
        text = cached_text(cache, video_id)
        found[video_id] = text or ""
    return found

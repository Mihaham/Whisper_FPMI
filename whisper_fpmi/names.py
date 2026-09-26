from __future__ import annotations

from dataclasses import dataclass
import re

UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')
TAG_RE = re.compile(r"^(\[[^\]]+\])-(.+)$")
TFTDS_RE = re.compile(
    r"^(?P<course>.+?)--(?P<kind>Лекция|Семинар)-(?P<num>\d+)--(?P<title>.+)$",
    re.IGNORECASE,
)
STREAM_THEN_NUM_RE = re.compile(
    r"^(?P<course>.+?)--(?P<stream>.+?)--(?P<num>\d+)-(?P<title>.+)$"
)
NUM_THEN_STREAM_RE = re.compile(
    r"^(?P<course>.+?)-(?P<num>\d+)--(?P<stream>.+?)--(?P<title>.+)$"
)
NUM_TITLE_RE = re.compile(r"^(?P<course>.+?)-(?P<num>\d+)-(?P<title>.+)$")
NUM_ONLY_RE = re.compile(r"^(?P<course>.+?)-(?P<num>\d+)$")

STREAM_HINTS = (
    "поток",
    "семинар",
    "лекция",
    "erp",
    "экономика",
    "допсем",
    "доп-сем",
)


@dataclass(frozen=True)
class ParsedLecture:
    course: str
    title: str
    slug: str
    number: int | None = None
    stream: str | None = None
    tag: str | None = None

    @property
    def display_title(self) -> str:
        heading = humanize(self.title)
        if self.number is not None:
            return f"{self.number}. {heading}"
        return heading

    @property
    def display_course(self) -> str:
        course = humanize(self.course)
        if self.stream:
            course = f"{course} ({humanize(self.stream)})"
        if self.tag:
            course = f"{self.tag} {course}"
        return course.strip()


def format_timestamp(seconds: float) -> str:
    milliseconds = round(max(seconds, 0) * 1000.0)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def sanitize_title(title: str) -> str:
    """Имена в том же стиле, что и у старого whisper.py."""
    name = title.strip().replace(" ", "-")
    for char in "()&|":
        name = name.replace(char, "-")
    name = UNSAFE_CHARS.sub("-", name)
    name = name.strip("-.")
    return name or "untitled"


def fold_key(value: str) -> str:
    """Сжатый ключ для сравнения уже существующих расшифровок с новыми названиями VK."""
    return re.sub(r"[^a-z0-9а-яё+]+", "", value.casefold())


def humanize(value: str) -> str:
    text = value.replace("--", " — ")
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_stem(stem: str) -> ParsedLecture:
    slug = stem
    tag = None
    rest = stem
    tagged = TAG_RE.match(stem)
    if tagged:
        tag, rest = tagged.group(1), tagged.group(2)

    match = TFTDS_RE.match(rest)
    if match:
        return ParsedLecture(
            course=_normalize_course(match.group("course")),
            stream=match.group("kind"),
            number=int(match.group("num")),
            title=match.group("title"),
            slug=slug,
            tag=tag,
        )

    match = STREAM_THEN_NUM_RE.match(rest)
    if match and _is_stream(match.group("stream")):
        return ParsedLecture(
            course=_normalize_course(match.group("course")),
            stream=match.group("stream"),
            number=int(match.group("num")),
            title=match.group("title"),
            slug=slug,
            tag=tag,
        )

    match = NUM_THEN_STREAM_RE.match(rest)
    if match and _is_stream(match.group("stream")):
        return ParsedLecture(
            course=_normalize_course(match.group("course")),
            stream=match.group("stream"),
            number=int(match.group("num")),
            title=match.group("title"),
            slug=slug,
            tag=tag,
        )

    match = NUM_TITLE_RE.match(rest)
    if match:
        course, stream = _split_trailing_stream(match.group("course"))
        return ParsedLecture(
            course=_normalize_course(course),
            stream=stream,
            number=int(match.group("num")),
            title=match.group("title"),
            slug=slug,
            tag=tag,
        )

    match = NUM_ONLY_RE.match(rest)
    if match:
        course, stream = _split_trailing_stream(match.group("course"))
        return ParsedLecture(
            course=_normalize_course(course),
            stream=stream,
            number=int(match.group("num")),
            title=match.group("course"),
            slug=slug,
            tag=tag,
        )

    course, stream = _split_trailing_stream(rest or "Разное")
    return ParsedLecture(
        course=_normalize_course(course),
        stream=stream,
        title=rest or stem,
        slug=slug,
        tag=tag,
    )


def _is_stream(value: str) -> bool:
    lowered = value.lower()
    return any(hint in lowered for hint in STREAM_HINTS)


TRAILING_STREAM_RE = re.compile(
    r"^(?P<course>.+?)-(?P<stream>(?:базовый|продвинутый|основной)-поток)$",
    re.IGNORECASE,
)


def _split_trailing_stream(course: str) -> tuple[str, str | None]:
    match = TRAILING_STREAM_RE.match(course)
    if match:
        return match.group("course"), match.group("stream")
    return course, None


def _normalize_course(course: str) -> str:
    if course.startswith("С++"):
        return "C++" + course[3:]
    return course

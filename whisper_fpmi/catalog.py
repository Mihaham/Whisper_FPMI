from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

from whisper_fpmi.names import ParsedLecture, parse_stem
from whisper_fpmi.paths import CATALOG_PATH, RESULT_DIR, ROOT, TIMED_DIR


def collect_lectures(result_dir: Path | None = None) -> list[ParsedLecture]:
    folder = result_dir or RESULT_DIR
    lectures = [
        parse_stem(path.stem)
        for path in sorted(folder.glob("*.txt"))
        if path.is_file()
    ]
    lectures.sort(key=_sort_key)
    return lectures


def group_by_course(lectures: list[ParsedLecture]) -> dict[str, list[ParsedLecture]]:
    grouped: dict[str, list[ParsedLecture]] = defaultdict(list)
    for lecture in lectures:
        grouped[lecture.display_course].append(lecture)
    ordered: dict[str, list[ParsedLecture]] = {}
    for course in sorted(grouped, key=str.casefold):
        ordered[course] = sorted(grouped[course], key=_sort_key)
    return ordered


def timed_path(lecture: ParsedLecture, timed_dir: Path | None = None) -> Path | None:
    candidate = (timed_dir or TIMED_DIR) / f"{lecture.slug}.mp4.txt"
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate
    return None


def render_catalog(lectures: list[ParsedLecture] | None = None) -> str:
    if lectures is None:
        lectures = collect_lectures()
    grouped = group_by_course(lectures)
    today = date.today().isoformat()
    lines = [
        "# Каталог расшифровок ФПМИ",
        "",
        "Список уже сохранённых в репозитории текстов лекций.",
        "Источник новых записей: [vkvideo.ru/@lectorium_fpmi](https://vkvideo.ru/@lectorium_fpmi).",
        "",
        f"- Расшифровок: **{len(lectures)}**",
        f"- Курсов: **{len(grouped)}**",
        f"- Обновлено: {today}",
        "",
    ]
    if (ROOT / "assets" / "wordcloud.svg").exists():
        lines.append("![Облако слов](assets/wordcloud.svg)")
        lines.append("")
    if (ROOT / "assets" / "term-graph.html").exists():
        lines.append("[Граф терминов по курсам](assets/term-graph.html)")
        lines.append("")
    lines.extend(
        [
            "Формат записи: **номер. тема** — ссылка на сплошной текст и, если есть, на текст с таймкодами.",
            "",
            "## Содержание",
            "",
        ]
    )
    for course, items in grouped.items():
        anchor = _anchor(course)
        lines.append(f"- [{course}](#{anchor}) ({len(items)})")
    lines.append("")

    for course, items in grouped.items():
        lines.append(f'<a id="{_anchor(course)}"></a>')
        lines.append(f"## {course}")
        lines.append("")
        for lecture in items:
            plain = _rel(RESULT_DIR / f"{lecture.slug}.txt")
            parts = [f"- [{lecture.display_title}]({plain})"]
            timed = timed_path(lecture)
            if timed:
                parts.append(f"[таймкоды]({_rel(timed)})")
            lines.append(" · ".join(parts))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_catalog(path: Path | None = None) -> Path:
    target = path or CATALOG_PATH
    target.write_text(render_catalog(), encoding="utf-8")
    return target


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _anchor(title: str) -> str:
    slug = title.casefold()
    slug = slug.replace(" ", "-")
    slug = "".join(char if char.isalnum() or char in "-_" else "" for char in slug)
    slug = slug.strip("-")
    return slug or "course"


def _sort_key(lecture: ParsedLecture) -> tuple:
    return (
        lecture.display_course.casefold(),
        lecture.number if lecture.number is not None else 10**9,
        lecture.display_title.casefold(),
    )

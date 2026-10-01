"""Облако слов по содержательным леммам."""

from __future__ import annotations

from collections import Counter
import html
import json
import math
from pathlib import Path

from whisper_fpmi.lang import analyze, get_lemmatizer
from whisper_fpmi.paths import WORDCLOUD_JSON, WORDCLOUD_SVG
from whisper_fpmi.transcript import corpus_hash

_COLORS = ("#8ecaff", "#9be7c4", "#f2c14e", "#f09a78", "#d2b0ff", "#f7f4ea")
CLOUD_VERSION = 3
# Слова, которыми вслух читают формулу, а не называют тему.
_DICTATION = {
    "плюс",
    "минус",
    "равно",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
    "десять",
    "первый",
    "второй",
    "третий",
    "большой",
    "маленький",
    "икс",
    "игрек",
    "ноль",
    "нуль",
    "говорить",
    "сказать",
    "написать",
    "писать",
    "знать",
    "дело",
    "получаться",
    "получить",
    "посмотреть",
    "смотреть",
    "видеть",
    "понимать",
    "хотеть",
    "давать",
    "дать",
    "взять",
    "идти",
    "делать",
    "сделать",
    "думать",
    "казаться",
    "являться",
    "называться",
    "означать",
}


def top_counts(counter: Counter[str] | dict[str, int], limit: int = 70) -> list[tuple[str, int]]:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    picked = [
        (word, count)
        for word, count in items
        if count > 0 and word and word not in _DICTATION
    ]
    return picked[:limit]


def render_svg(pairs: list[tuple[str, int]], width: int = 1100, height: int = 720) -> str:
    if not pairs:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<rect width="100%" height="100%" fill="#10151c"/>'
            '<text x="40" y="80" fill="#e7ecf3" font-size="28" '
            'font-family="Segoe UI, Arial, sans-serif">Нет слов</text></svg>'
        )
    max_count = pairs[0][1] or 1
    placed: list[tuple[float, float, float, float]] = []
    tags: list[str] = []
    cx, cy = width / 2, height / 2
    for index, (word, count) in enumerate(pairs):
        weight = count / max_count
        size = 15 + weight * 62
        box_w = size * 0.58 * len(word) + 8
        box_h = size * 1.15
        spot = _place(box_w, box_h, cx, cy, placed, width, height)
        if spot is None:
            continue
        x, y = spot
        placed.append((x, y, box_w, box_h))
        color = _COLORS[index % len(_COLORS)] if weight < 0.85 else _COLORS[0]
        tags.append(
            f'<text x="{x:.1f}" y="{y + box_h * 0.78:.1f}" fill="{color}" '
            f'font-size="{size:.1f}" font-family="Segoe UI, Arial, sans-serif">'
            f"{html.escape(word)}</text>"
        )
    body = "\n".join(tags)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="100%" height="100%" fill="#10151c"/>\n{body}\n</svg>\n'
    )


def write_wordcloud(
    counter: Counter[str] | dict[str, int],
    source_hash: str,
    lemmatizer_name: str,
    *,
    svg_path: Path = WORDCLOUD_SVG,
    json_path: Path = WORDCLOUD_JSON,
    limit: int = 70,
) -> Path:
    pairs = top_counts(counter, limit)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(render_svg(pairs), encoding="utf-8")
    payload = {
        "source_hash": source_hash,
        "lemmatizer": lemmatizer_name,
        "version": CLOUD_VERSION,
        "words": [{"text": word, "count": count} for word, count in pairs],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return svg_path


def is_fresh(result_dir: Path, json_path: Path = WORDCLOUD_JSON, svg_path: Path = WORDCLOUD_SVG) -> tuple[bool, str]:
    if not json_path.is_file() or not svg_path.is_file():
        return False, "облако ещё не собрано"
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "wordcloud.json повреждён"
    digest = corpus_hash(result_dir)
    if meta.get("source_hash") != digest:
        return False, "тексты лекций изменились, облако устарело"
    _lemmatize, name = get_lemmatizer()
    if meta.get("lemmatizer") != name:
        return False, "сменился способ лемматизации, облако надо пересобрать"
    if meta.get("version") != CLOUD_VERSION:
        return False, "облако собрано старым фильтром слов"
    return True, "облако совпадает с текстами"


def counts_from_texts(texts: list[str], lemmatize) -> Counter[str]:
    total: Counter[str] = Counter()
    for text in texts:
        total.update(analyze(text, lemmatize).counts)
    return total


def _place(box_w, box_h, cx, cy, placed, width, height):
    for step in range(700):
        angle = step * 0.45
        radius = step * 2.05
        x = cx + math.cos(angle) * radius - box_w / 2
        y = cy + math.sin(angle) * radius * 0.62 - box_h / 2
        if x < 8 or y < 8 or x + box_w > width - 8 or y + box_h > height - 8:
            continue
        if any(_hits(x, y, box_w, box_h, other) for other in placed):
            continue
        return x, y
    return None


def _hits(x, y, w, h, other, pad: float = 4) -> bool:
    ox, oy, ow, oh = other
    return not (
        x + w + pad < ox
        or ox + ow + pad < x
        or y + h + pad < oy
        or oy + oh + pad < y
    )

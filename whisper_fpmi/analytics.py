"""Сборка датасета, облака, графа, индекса и метрик."""

from __future__ import annotations

from collections import Counter
import json

from loguru import logger as lg
from tqdm import tqdm

from whisper_fpmi.cloud import is_fresh, write_wordcloud
from whisper_fpmi.dataset import make_record, read_jsonl, write_csv, write_jsonl
from whisper_fpmi.descriptions import fetch_descriptions
from whisper_fpmi.evaluate import retrieval_report
from whisper_fpmi.lang import get_lemmatizer
from whisper_fpmi.names import fold_key, sanitize_title
from whisper_fpmi.paths import (
    DESCRIPTIONS_PATH,
    LECTURES_CSV,
    LECTURES_PATH,
    METRICS_PATH,
    RESULT_DIR,
    ROOT,
    SEARCH_INDEX_PATH,
    TERM_GRAPH_HTML,
    TIMED_DIR,
)
from whisper_fpmi.rag import answer_query
from whisper_fpmi.search import (
    Hit,
    Index,
    build_chunks,
    build_index,
    load_index,
    save_index,
    search,
    snippet,
    vk_link,
)
from whisper_fpmi.state import load_state
from whisper_fpmi.terms import build_graph, graph_stats, write_html
from whisper_fpmi.timecodes import format_clock_full
from whisper_fpmi.transcript import load_lectures


def build_all(*, offline: bool = False, refresh: bool = False, workers: int = 4) -> dict:
    lemmatize, name = get_lemmatizer()
    lg.info("Лемматизатор: {}", name)
    lectures, digest = load_lectures(RESULT_DIR, TIMED_DIR)
    lg.info("Лекций в result/: {}", len(lectures))
    durations = {} if offline else _attach_channel(lectures)
    _attach_state(lectures)
    descriptions = _descriptions(lectures, offline=offline, refresh=refresh, workers=workers)

    records = []
    counts: Counter[str] = Counter()
    rows = []
    chunks = []
    for lecture in tqdm(lectures, desc="разбор", unit="лекция"):
        description = descriptions.get(lecture.video_id or "", "")
        record, stats = make_record(
            lecture,
            description=description,
            vk_duration=durations.get(lecture.video_id or ""),
            lemmatize=lemmatize,
            root=ROOT,
        )
        records.append(record)
        counts.update(stats.counts)
        rows.append(
            {
                "course": record["course"],
                "slug": record["slug"],
                "title": record["title"],
                "number": record["number"],
                "url": record["url"],
                "counts": stats.counts,
            }
        )
        chunks.extend(
            build_chunks(
                lecture,
                title=record["title"],
                course=record["course"],
                lemmatize=lemmatize,
            )
        )

    write_jsonl(records, LECTURES_PATH)
    write_csv(records, LECTURES_CSV)
    cloud = write_wordcloud(counts, digest, name)
    graph = build_graph(rows)
    page = write_html(graph, TERM_GRAPH_HTML)
    index = build_index(chunks, lemmatize, name)
    save_index(index, SEARCH_INDEX_PATH)
    lg.info("Считаю метрики поиска и RAG")
    retrieval = retrieval_report(index, records, lemmatize)
    metrics = _summary(records, name, graph, index, retrieval, cloud, page)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def ensure_index() -> tuple[Index, object]:
    lemmatize, name = get_lemmatizer()
    index = load_index(SEARCH_INDEX_PATH)
    if index is not None and index.lemmatizer == name and index.chunks:
        return index, lemmatize
    lg.info("Индекса нет или он от другого лемматизатора, собираю заново")
    from whisper_fpmi.names import parse_stem

    lectures, _digest = load_lectures(RESULT_DIR, TIMED_DIR)
    chunks = []
    for lecture in tqdm(lectures, desc="индекс", unit="лекция"):
        parsed = parse_stem(lecture.slug)
        chunks.extend(
            build_chunks(
                lecture,
                title=parsed.display_title,
                course=parsed.display_course,
                lemmatize=lemmatize,
            )
        )
    index = build_index(chunks, lemmatize, name)
    save_index(index, SEARCH_INDEX_PATH)
    return index, lemmatize


def format_hits(hits: list[Hit], query: str) -> str:
    if not hits:
        return "Ничего не нашлось."
    lines = []
    for hit in hits:
        chunk = hit.chunk
        link = vk_link(chunk.url, chunk.start)
        lines.append(
            f"{hit.rank}. {format_clock_full(chunk.start)}  {chunk.course} — {chunk.title}"
            f"  ({hit.score:.2f})"
        )
        if link:
            lines.append(f"   {link}")
        lines.append(f"   {snippet(chunk.text, query)}")
    return "\n".join(lines)


def _descriptions(lectures, *, offline: bool, refresh: bool, workers: int) -> dict[str, str]:
    ids = [lecture.video_id for lecture in lectures if lecture.video_id]
    if offline:
        from whisper_fpmi.descriptions import cached_text, load_cache

        cache = load_cache(DESCRIPTIONS_PATH)
        return {video_id: cached_text(cache, video_id) or "" for video_id in ids}
    return fetch_descriptions(ids, DESCRIPTIONS_PATH, workers=workers, refresh=refresh)


def _attach_state(lectures) -> None:
    state = load_state()
    slug_to_id = {}
    for video_id, record in (state.get("videos") or {}).items():
        slug = record.get("slug")
        if slug:
            slug_to_id.setdefault(slug, video_id)
    for lecture in lectures:
        if lecture.video_id:
            continue
        lecture.video_id = slug_to_id.get(lecture.slug)


def _attach_channel(lectures) -> dict[str, float | None]:
    from whisper_fpmi.vk import list_channel_videos

    try:
        videos = list_channel_videos()
    except (OSError, RuntimeError) as exc:
        lg.warning("Список канала недоступен: {}", exc)
        return {}
    by_fold = {}
    durations: dict[str, float | None] = {}
    for video in videos:
        durations[video.video_id] = video.duration
        by_fold.setdefault(fold_key(sanitize_title(video.title)), video)
    for lecture in lectures:
        if lecture.video_id:
            continue
        video = by_fold.get(fold_key(lecture.slug))
        if video:
            lecture.video_id = video.video_id
    lg.info("Сопоставлено с каналом, роликов в списке: {}", len(videos))
    return durations


def _summary(records, lemmatizer, graph, index, retrieval, cloud, page) -> dict:
    described = [row for row in records if (row.get("description") or "").strip()]
    with_codes = [row for row in records if row.get("timecode_count")]
    return {
        "lectures": len(records),
        "with_video_id": sum(1 for row in records if row.get("video_id")),
        "with_description": len(described),
        "with_lecturer": sum(1 for row in records if row.get("lecturer")),
        "with_timecodes": len(with_codes),
        "timecodes": sum(row.get("timecode_count") or 0 for row in records),
        "timecodes_with_hour_field": sum(row.get("timecodes_with_hour_field") or 0 for row in records),
        "timecodes_past_one_hour": sum(row.get("timecodes_past_one_hour") or 0 for row in records),
        "lemmatizer": lemmatizer,
        "wordcloud": cloud.relative_to(ROOT).as_posix(),
        "term_graph": page.relative_to(ROOT).as_posix(),
        "graph": graph_stats(graph),
        "chunks": index.size,
        "retrieval": retrieval,
    }


def dataset_rows() -> list[dict]:
    return read_jsonl(LECTURES_PATH)


def run_search(query: str, *, limit: int = 8, course: str | None = None) -> list[Hit]:
    index, lemmatize = ensure_index()
    return search(index, query, lemmatize, limit=limit, course=course)


def run_rag(query: str, *, limit: int = 5, course: str | None = None):
    index, lemmatize = ensure_index()
    return answer_query(index, query, lemmatize, limit=limit, course=course)


def rebuild_cloud() -> None:
    from whisper_fpmi.cloud import counts_from_texts

    lectures, digest = load_lectures(RESULT_DIR, TIMED_DIR)
    lemmatize, name = get_lemmatizer()
    counts = counts_from_texts([lecture.text for lecture in lectures], lemmatize)
    write_wordcloud(counts, digest, name)


def cloud_check() -> tuple[bool, str]:
    return is_fresh(RESULT_DIR)

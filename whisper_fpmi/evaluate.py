"""Метрики поиска и RAG по главам из описаний и по названиям лекций."""

from __future__ import annotations

from dataclasses import dataclass

from whisper_fpmi.lang import stem_token, usable_phrase
from whisper_fpmi.rag import extractive_answer, grounded_fraction
from whisper_fpmi.search import Index, search

_HITS = (1, 5, 10)


@dataclass(frozen=True)
class Gold:
    query: str
    slug: str
    course: str
    start: float
    end: float
    kind: str


def chapter_queries(records: list[dict], lemmatize=stem_token) -> tuple[list[Gold], int]:
    gold: list[Gold] = []
    skipped = 0
    for record in records:
        codes = record.get("timecodes") or []
        if not codes:
            continue
        duration = record.get("duration_sec") or 0
        for index, code in enumerate(codes):
            title = (code.get("title") or "").strip()
            if not usable_phrase(title, lemmatize):
                skipped += 1
                continue
            start = float(code["seconds"])
            if index + 1 < len(codes):
                end = float(codes[index + 1]["seconds"])
            elif duration and duration > start:
                end = float(duration)
            else:
                end = start + 15 * 60
            if end <= start:
                end = start + 60
            gold.append(
                Gold(
                    query=title,
                    slug=record["slug"],
                    course=record.get("course") or "",
                    start=start,
                    end=end,
                    kind="chapter",
                )
            )
    return gold, skipped


def title_queries(records: list[dict]) -> list[Gold]:
    gold = []
    for record in records:
        title = (record.get("title") or "").strip()
        course = record.get("course") or ""
        if not title:
            continue
        gold.append(
            Gold(
                query=f"{course} {title}",
                slug=record["slug"],
                course=course,
                start=0.0,
                end=10**9,
                kind="title",
            )
        )
    return gold


def retrieval_report(index: Index, records: list[dict], lemmatize=stem_token) -> dict:
    chapters, skipped = chapter_queries(records, lemmatize)
    titles = [item for item in title_queries(records) if item.slug in _slugs(index)]
    chapters = [item for item in chapters if item.slug in _slugs(index)]
    report = {
        "chapters": _pack(chapters, index, lemmatize, boundary=True),
        "chapters_in_course": _pack(chapters, index, lemmatize, boundary=True, by_course=True),
        "titles": _pack(titles, index, lemmatize, boundary=False),
        "chapter_queries_skipped": skipped,
    }
    report["rag"] = _rag_scores(_sample(chapters, 500), index, lemmatize)
    return report


def _pack(golds, index, lemmatize, *, boundary: bool, by_course: bool = False) -> dict:
    if not golds:
        return {"queries": 0}
    ranks = []
    hits = {k: 0 for k in _HITS}
    boundary_hits = 0
    for gold in golds:
        found = search(
            index,
            gold.query,
            lemmatize,
            limit=10,
            course=gold.course if by_course else None,
        )
        rank = _first_rank(found, gold)
        ranks.append(rank)
        for k in _HITS:
            if rank is not None and rank <= k:
                hits[k] += 1
        if boundary and any(_boundary(hit.chunk, gold) for hit in found[:5]):
            boundary_hits += 1
    total = len(golds)
    packed = {
        "queries": total,
        "mrr_at_10": round(sum((1 / rank) if rank else 0 for rank in ranks) / total, 4),
    }
    for k in _HITS:
        packed[f"hit_rate_at_{k}"] = round(hits[k] / total, 4)
    if boundary:
        packed["boundary_hit_at_90s_at_5"] = round(boundary_hits / total, 4)
    return packed


def _rag_scores(golds: list[Gold], index: Index, lemmatize) -> dict:
    if not golds:
        return {"queries": 0}
    support_hits = 0
    grounded = 0.0
    for gold in golds:
        found = search(index, gold.query, lemmatize, limit=5)
        answer = extractive_answer(gold.query, found, lemmatize)
        if _support_hit(answer, gold):
            support_hits += 1
        grounded += grounded_fraction(answer.answer, answer.citations, lemmatize)
    total = len(golds)
    return {
        "queries": total,
        "support_hit_rate": round(support_hits / total, 4),
        "grounded": round(grounded / total, 4),
        "mode": "extractive",
    }


def _support_hit(answer, gold: Gold) -> bool:
    if answer.support_slug != gold.slug:
        return False
    if gold.kind == "title":
        return True
    return answer.support_start < gold.end and gold.start < answer.support_end


def _first_rank(hits, gold: Gold) -> int | None:
    for hit in hits:
        if _relevant(hit.chunk, gold):
            return hit.rank
    return None


def _relevant(chunk, gold: Gold) -> bool:
    if chunk.slug != gold.slug:
        return False
    if gold.kind == "title":
        return True
    if chunk.end <= chunk.start:
        return False
    return chunk.start < gold.end and gold.start < chunk.end


def _boundary(chunk, gold: Gold) -> bool:
    return chunk.slug == gold.slug and abs(chunk.start - gold.start) <= 90


def _slugs(index: Index) -> set[str]:
    return {chunk.slug for chunk in index.chunks}


def _sample(items: list[Gold], limit: int) -> list[Gold]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]

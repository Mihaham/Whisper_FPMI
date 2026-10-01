"""Нарезка лекций и поиск BM25 без нейросети."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import pickle
import re
from pathlib import Path

from whisper_fpmi.lang import analyze, fold, stem_token
from whisper_fpmi.timecodes import format_clock
from whisper_fpmi.transcript import LectureText, Segment

CHUNK_WORDS = 140
CHUNK_OVERLAP = 40
_SENTENCE = re.compile(r"[^.!?…]+[.!?…]+|[^.!?…]+$")


@dataclass
class Chunk:
    slug: str
    course: str
    title: str
    url: str
    start: float
    end: float
    text: str
    title_lemmas: tuple[str, ...] = ()


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int = 0


@dataclass
class Index:
    chunks: list[Chunk]
    postings: dict[str, list[tuple[int, int]]]
    doc_len: list[int]
    avgdl: float
    k1: float = 1.2
    b: float = 0.75
    lemmatizer: str = "suffix"
    df: dict[str, int] = field(default_factory=dict)
    title_postings: dict[str, list[int]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.chunks)


def build_chunks(
    lecture: LectureText,
    *,
    title: str,
    course: str,
    lemmatize=stem_token,
    window: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    title_lemmas = tuple(analyze(title, lemmatize).counts)
    pieces = _windows(_explode(lecture.segments), window, overlap)
    if not pieces and lecture.text.strip():
        pieces = [Segment(0.0, 0.0, lecture.text.strip())]
    chunks = []
    for piece in pieces:
        if not piece.text.strip():
            continue
        chunks.append(
            Chunk(
                slug=lecture.slug,
                course=course,
                title=title,
                url=lecture.source_url,
                start=piece.start,
                end=piece.end,
                text=piece.text.strip(),
                title_lemmas=title_lemmas,
            )
        )
    return chunks


def build_index(chunks: list[Chunk], lemmatize=stem_token, lemmatizer_name: str = "suffix") -> Index:
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    title_postings: dict[str, list[int]] = defaultdict(list)
    doc_len: list[int] = []
    for doc_id, chunk in enumerate(chunks):
        counts = analyze(chunk.text, lemmatize).counts
        length = int(sum(counts.values()))
        doc_len.append(length)
        for lemma, freq in counts.items():
            postings[lemma].append((doc_id, freq))
        for lemma in chunk.title_lemmas:
            title_postings[lemma].append(doc_id)
    avgdl = (sum(doc_len) / len(doc_len)) if doc_len else 0.0
    df = {term: len(items) for term, items in postings.items()}
    return Index(
        chunks=chunks,
        postings=dict(postings),
        doc_len=doc_len,
        avgdl=avgdl,
        lemmatizer=lemmatizer_name,
        df=df,
        title_postings=dict(title_postings),
    )


def search(
    index: Index,
    query: str,
    lemmatize=stem_token,
    *,
    limit: int = 8,
    course: str | None = None,
) -> list[Hit]:
    terms = _query_terms(query, lemmatize)
    if not terms or not index.chunks:
        return []
    course_key = course.casefold().strip() if course else ""
    scores: dict[int, float] = defaultdict(float)
    total = index.size
    avgdl = index.avgdl or 1.0
    for term, qtf in terms.items():
        posting = index.postings.get(term)
        if not posting:
            continue
        df = index.df.get(term, len(posting))
        idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
        for doc_id, freq in posting:
            chunk = index.chunks[doc_id]
            if course_key and course_key not in chunk.course.casefold():
                continue
            length = index.doc_len[doc_id] or 1
            denom = freq + index.k1 * (1 - index.b + index.b * length / avgdl)
            scores[doc_id] += idf * freq * (index.k1 + 1) / denom * qtf
    bonus_docs: set[int] = set()
    for term in terms:
        bonus_docs.update(index.title_postings.get(term, ()))
    for doc_id in bonus_docs:
        chunk = index.chunks[doc_id]
        if course_key and course_key not in chunk.course.casefold():
            continue
        bonus = _title_bonus(chunk.title_lemmas, terms, _course_lemmas(chunk.course, lemmatize))
        if bonus:
            scores[doc_id] += bonus
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    hits = []
    for rank, (doc_id, score) in enumerate(ranked[:limit], start=1):
        if score <= 0:
            continue
        hits.append(Hit(chunk=index.chunks[doc_id], score=score, rank=rank))
    return hits


def save_index(index: Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(index, protocol=4))


def load_index(path: Path) -> Index | None:
    if not path.is_file():
        return None
    return pickle.loads(path.read_bytes())


def snippet(text: str, query: str, width: int = 240) -> str:
    folded = text.casefold()
    position = -1
    for token in query.split():
        needle = fold(token)
        if len(needle) < 3:
            continue
        position = folded.find(needle)
        if position >= 0:
            break
    if position < 0:
        position = 0
    start = max(0, position - 70)
    end = min(len(text), start + width)
    clip = " ".join(text[start:end].split())
    if start > 0:
        clip = "…" + clip
    if end < len(text):
        clip = clip + "…"
    return clip


def vk_link(url: str, seconds: float) -> str:
    if not url:
        return ""
    parts = format_clock(max(seconds, 0)).split(":")
    if len(parts) == 3:
        stamp = f"{int(parts[0])}h{int(parts[1])}m{int(parts[2])}s"
    else:
        stamp = f"{int(parts[0])}m{int(parts[1])}s"
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}t={stamp}"


_COURSE_LEMMAS: dict[str, set[str]] = {}


def _course_lemmas(course: str, lemmatize) -> set[str]:
    cached = _COURSE_LEMMAS.get(course)
    if cached is None:
        cached = set(analyze(course, lemmatize).counts)
        _COURSE_LEMMAS[course] = cached
    return cached


def _title_bonus(title_lemmas: tuple[str, ...], terms: dict[str, int], course_lemmas: set[str]) -> float:
    """Сильный бонус, если запрос — это название лекции, а не глава с лишними словами."""
    title = set(title_lemmas)
    if not title:
        return 0.0
    overlap = sum(1 for term in terms if term in title)
    if not overlap:
        return 0.0
    cover_title = overlap / len(title)
    extra_topic = sum(1 for term in terms if term not in title and term not in course_lemmas)
    if overlap >= 2 and cover_title >= 0.75 and extra_topic <= 1:
        return 24.0 * cover_title + 8.0 * overlap
    if overlap == 1 and cover_title == 1.0 and extra_topic <= 1:
        return 18.0
    return 1.15 * overlap


def _query_terms(query: str, lemmatize) -> dict[str, int]:
    counts = analyze(query, lemmatize).counts
    if counts:
        return counts
    folded = [fold(token) for token in query.split()]
    return {token: folded.count(token) for token in folded if token}


def _explode(segments: list[Segment]) -> list[Segment]:
    pieces: list[Segment] = []
    for segment in segments:
        words = analyze(segment.text).tokens
        if words <= CHUNK_WORDS:
            pieces.append(segment)
            continue
        sentences = [part.strip() for part in _SENTENCE.findall(segment.text) if part.strip()]
        if len(sentences) < 2:
            pieces.append(segment)
            continue
        span = max(segment.end - segment.start, 0.0)
        total = max(len(segment.text), 1)
        cursor = 0
        for sentence in sentences:
            pos = segment.text.find(sentence, cursor)
            if pos < 0:
                pos = cursor
            start = segment.start + span * (pos / total)
            end = segment.start + span * ((pos + len(sentence)) / total)
            pieces.append(Segment(start, max(end, start), sentence))
            cursor = pos + len(sentence)
    return pieces


def _windows(segments: list[Segment], window: int, overlap: int) -> list[Segment]:
    if not segments:
        return []
    lengths = [max(analyze(segment.text).tokens, 1) for segment in segments]
    chunks: list[Segment] = []
    start = 0
    total = len(segments)
    while start < total:
        words = 0
        end = start
        while end < total and (words < window or end == start):
            if end > start and segments[end].start - segments[end - 1].end > 30:
                break
            words += lengths[end]
            end += 1
            if words >= window:
                break
        group = segments[start:end]
        text = " ".join(segment.text.strip() for segment in group if segment.text.strip())
        chunks.append(Segment(group[0].start, group[-1].end, text))
        if end >= total:
            break
        new_start = end
        kept = 0
        while new_start > start and kept < overlap:
            new_start -= 1
            kept += lengths[new_start]
        if new_start <= start:
            new_start = start + 1
        start = new_start
    return chunks

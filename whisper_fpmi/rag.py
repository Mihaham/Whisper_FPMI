"""Ответ по найденным фрагментам: цитаты и, если задан, внешний LLM."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import urllib.request

from whisper_fpmi.lang import analyze, stem_token
from whisper_fpmi.search import Hit, Index, search, snippet, vk_link
from whisper_fpmi.timecodes import format_clock_full

_SENTENCE = re.compile(r"[^.!?…]+[.!?…]+|[^.!?…]+$")


@dataclass
class RagAnswer:
    query: str
    answer: str
    mode: str
    citations: list[dict]
    support_slug: str
    support_start: float
    support_end: float

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "mode": self.mode,
            "support_slug": self.support_slug,
            "support_start": self.support_start,
            "support_end": self.support_end,
            "citations": self.citations,
        }


def answer_query(
    index: Index,
    query: str,
    lemmatize=stem_token,
    *,
    limit: int = 5,
    course: str | None = None,
    generate: bool | None = None,
) -> RagAnswer:
    hits = search(index, query, lemmatize, limit=limit, course=course)
    draft = extractive_answer(query, hits, lemmatize)
    use_llm = generate if generate is not None else _llm_configured()
    if not use_llm or not hits:
        return draft
    generated = _complete(query, draft.citations)
    if not generated:
        return draft
    draft.answer = generated
    draft.mode = "llm"
    return draft


def extractive_answer(query: str, hits: list[Hit], lemmatize=stem_token) -> RagAnswer:
    citations = [_citation(hit, query) for hit in hits]
    query_lemmas = set(analyze(query, lemmatize).counts)
    best: list[tuple[int, int, str]] = []
    for index, hit in enumerate(hits, start=1):
        for sentence in _sentences(hit.chunk.text):
            lemmas = set(analyze(sentence, lemmatize).counts)
            overlap = len(query_lemmas & lemmas)
            if overlap <= 0:
                continue
            best.append((overlap, -index, sentence.strip()))
    best.sort(key=lambda item: (-item[0], -item[1]))
    chosen: list[tuple[int, str]] = []
    seen: set[str] = set()
    for _overlap, negative_index, sentence in best:
        if sentence in seen:
            continue
        seen.add(sentence)
        chosen.append((-negative_index, sentence))
        if len(chosen) >= 3:
            break
    if not chosen and hits:
        text = hits[0].chunk.text.strip()
        clip = text[:320].rsplit(" ", 1)[0] if len(text) > 320 else text
        chosen = [(1, clip)]
    lines = [f"{sentence} [{index}]" for index, sentence in chosen]
    support = hits[0] if hits else None
    if chosen and hits:
        support = hits[chosen[0][0] - 1]
    return RagAnswer(
        query=query,
        answer="\n".join(lines),
        mode="extractive",
        citations=citations,
        support_slug=support.chunk.slug if support else "",
        support_start=support.chunk.start if support else 0.0,
        support_end=support.chunk.end if support else 0.0,
    )


def grounded_fraction(answer: str, citations: list[dict], lemmatize=stem_token) -> float:
    """Доля содержательных лемм ответа, которые есть в процитированных фрагментах."""
    answer_lemmas = set(analyze(answer, lemmatize).counts)
    if not answer_lemmas:
        return 1.0
    source: set[str] = set()
    for citation in citations:
        source.update(analyze(citation.get("text", ""), lemmatize).counts)
    if not source:
        return 0.0
    return len(answer_lemmas & source) / len(answer_lemmas)


def _citation(hit: Hit, query: str) -> dict:
    chunk = hit.chunk
    return {
        "rank": hit.rank,
        "score": round(hit.score, 4),
        "slug": chunk.slug,
        "course": chunk.course,
        "title": chunk.title,
        "start": chunk.start,
        "end": chunk.end,
        "clock": format_clock_full(chunk.start),
        "url": vk_link(chunk.url, chunk.start),
        "snippet": snippet(chunk.text, query),
        "text": chunk.text,
    }


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.findall(text) if part.strip()]


def _llm_configured() -> bool:
    return bool(os.environ.get("FPMI_LLM_URL") or os.environ.get("OPENAI_API_KEY"))


def _complete(query: str, citations: list[dict]) -> str | None:
    endpoint = os.environ.get("FPMI_LLM_URL") or "https://api.openai.com/v1/chat/completions"
    key = os.environ.get("FPMI_LLM_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    model = os.environ.get("FPMI_LLM_MODEL", "gpt-4o-mini")
    context = []
    for citation in citations:
        context.append(
            f"[{citation['rank']}] {citation['course']} — {citation['title']} "
            f"@ {citation['clock']}\n{citation['text']}"
        )
    prompt = (
        "Ответь на вопрос только по фрагментам лекций. "
        "Каждое утверждение сопроводи номером фрагмента в квадратных скобках. "
        "Если ответа во фрагментах нет, так и скажи.\n\n"
        f"Вопрос: {query}\n\n" + "\n\n".join(context)
    )
    body = json.dumps(
        {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Ты отвечаешь по конспектам лекций ФПМИ."},
                {"role": "user", "content": prompt},
            ],
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(endpoint, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    choices = payload.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    text = (message.get("content") or "").strip()
    return text or None

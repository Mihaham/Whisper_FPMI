"""Локальная страница поиска, RAG и графа терминов."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import parse_qs, urlparse

from whisper_fpmi.paths import METRICS_PATH, TERM_GRAPH_HTML, WORDCLOUD_SVG

_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Поиск по лекциям ФПМИ</title>
<style>
  body { margin: 0; background: #10151c; color: #f7f4ea; font: 16px/1.45 Segoe UI, sans-serif; }
  main { max-width: 860px; margin: 0 auto; padding: 24px 16px 64px; }
  form { display: flex; gap: 8px; flex-wrap: wrap; }
  input { flex: 1; min-width: 180px; background: #182230; color: #f7f4ea;
    border: 1px solid #31445c; border-radius: 8px; padding: 10px 12px; font: inherit; }
  button, a.button { background: #2a6f97; color: white; border: 0; border-radius: 8px;
    padding: 10px 14px; font: inherit; cursor: pointer; text-decoration: none; }
  article { margin-top: 18px; padding-top: 8px; border-top: 1px solid #243140; }
  .meta { color: #8ecaff; }
  img { max-width: 100%; margin-top: 28px; border-radius: 12px; }
  pre { white-space: pre-wrap; }
</style>
</head>
<body>
<main>
  <h1>Лекции ФПМИ</h1>
  <p><a href="/graph">Граф терминов</a></p>
  <form id="search">
    <input name="q" placeholder="вопрос или фраза из лекции" required/>
    <input name="course" placeholder="курс, если известен"/>
    <button type="submit">Найти</button>
    <button type="button" id="ask">Ответ</button>
  </form>
  <div id="out"></div>
  <img src="/wordcloud.svg" alt="Облако слов"/>
</main>
<script>
const out = document.getElementById("out");
function query() {
  const data = new FormData(document.getElementById("search"));
  const params = new URLSearchParams();
  for (const [key, value] of data.entries()) {
    if (value) params.set(key, value);
  }
  return params;
}
function showHits(hits) {
  if (!hits.length) {
    out.textContent = "Ничего не нашлось.";
    return;
  }
  out.innerHTML = hits.map((hit) => {
    const link = hit.url ? '<a href="' + hit.url + '">' + hit.clock + "</a>" : hit.clock;
    return "<article><div class=meta>" + link + " · " + escapeHtml(hit.course) +
      " — " + escapeHtml(hit.title) + "</div><p>" + escapeHtml(hit.snippet) + "</p></article>";
  }).join("");
}
function escapeHtml(value) {
  return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
document.getElementById("search").addEventListener("submit", async (event) => {
  event.preventDefault();
  const response = await fetch("/api/search?" + query());
  const payload = await response.json();
  showHits(payload.hits || []);
});
document.getElementById("ask").addEventListener("click", async () => {
  const response = await fetch("/api/rag?" + query());
  const payload = await response.json();
  const answer = document.createElement("article");
  answer.innerHTML = "<pre>" + escapeHtml(payload.answer || "") + "</pre>";
  out.innerHTML = "";
  out.appendChild(answer);
  showHits(payload.citations || []);
  out.prepend(answer);
});
</script>
</body>
</html>
"""


def serve(port: int = 8765) -> None:
    from whisper_fpmi.analytics import ensure_index
    from whisper_fpmi.rag import answer_query
    from whisper_fpmi.search import search

    index, lemmatize = ensure_index()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._bytes(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/graph":
                self._file(TERM_GRAPH_HTML, "text/html; charset=utf-8")
                return
            if parsed.path == "/wordcloud.svg":
                self._file(WORDCLOUD_SVG, "image/svg+xml")
                return
            if parsed.path == "/api/metrics":
                self._file(METRICS_PATH, "application/json; charset=utf-8")
                return
            params = {key: values[0] for key, values in parse_qs(parsed.query).items() if values}
            if parsed.path == "/api/search":
                hits = search(
                    index,
                    params.get("q", ""),
                    lemmatize,
                    limit=_limit(params.get("limit"), 8),
                    course=params.get("course") or None,
                )
                payload = [_public_hit(hit, params.get("q", "")) for hit in hits]
                self._json({"hits": payload})
                return
            if parsed.path == "/api/rag":
                answer = answer_query(
                    index,
                    params.get("q", ""),
                    lemmatize,
                    limit=_limit(params.get("limit"), 5),
                    course=params.get("course") or None,
                )
                body = answer.as_dict()
                for citation in body["citations"]:
                    citation.pop("text", None)
                self._json(body)
                return
            self._bytes(404, b"not found", "text/plain; charset=utf-8")

        def log_message(self, fmt, *args):
            return

        def _file(self, path, content_type):
            if not path.is_file():
                self._bytes(404, "файл ещё не собран\n".encode("utf-8"), "text/plain; charset=utf-8")
                return
            self._bytes(200, path.read_bytes(), content_type)

        def _json(self, payload):
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._bytes(200, raw, "application/json; charset=utf-8")

        def _bytes(self, status, raw, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"http://127.0.0.1:{port}")
    server.serve_forever()


def _public_hit(hit, query: str) -> dict:
    from whisper_fpmi.search import snippet, vk_link
    from whisper_fpmi.timecodes import format_clock_full

    chunk = hit.chunk
    return {
        "rank": hit.rank,
        "score": round(hit.score, 4),
        "course": chunk.course,
        "title": chunk.title,
        "clock": format_clock_full(chunk.start),
        "url": vk_link(chunk.url, chunk.start),
        "snippet": snippet(chunk.text, query),
    }


def _limit(value: str | None, default: int) -> int:
    try:
        parsed = int(value) if value else default
    except ValueError:
        return default
    return min(max(parsed, 1), 20)

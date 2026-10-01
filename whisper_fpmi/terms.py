"""Граф терминов курса: вершины — леммы, рёбра — совместная встречаемость."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import json
import math
from pathlib import Path

_JS = Path(__file__).with_name("term_graph.js")


def build_graph(
    rows: list[dict],
    *,
    per_course: int = 40,
    min_jaccard: float = 0.2,
    max_edges_per_node: int = 4,
) -> dict:
    """rows: course, slug, title, number, url, counts(dict lemma->tf)."""
    global_df: Counter[str] = Counter()
    for row in rows:
        global_df.update(row["counts"].keys())
    corpus = max(len(rows), 1)
    by_course: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_course[row["course"]].append(row)

    courses = []
    for name in sorted(by_course, key=str.casefold):
        lectures = by_course[name]
        courses.append(
            _course_graph(
                name,
                lectures,
                global_df,
                corpus,
                per_course=per_course,
                min_jaccard=min_jaccard,
                max_edges_per_node=max_edges_per_node,
            )
        )
    return {"courses": [course for course in courses if course["terms"]]}


def render_html(graph: dict) -> str:
    payload = json.dumps(graph, ensure_ascii=False).replace("<", "\\u003c")
    script = _JS.read_text(encoding="utf-8")
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Термины курсов ФПМИ</title>
<style>
  body {{ margin: 0; background: #10151c; color: #f7f4ea;
    font: 15px/1.45 Segoe UI, sans-serif; }}
  header {{ display: flex; gap: 12px; align-items: center; padding: 12px 16px;
    border-bottom: 1px solid #243140; flex-wrap: wrap; }}
  select, input {{ background: #182230; color: #f7f4ea; border: 1px solid #31445c;
    border-radius: 8px; padding: 8px 10px; font: inherit; }}
  main {{ display: grid; grid-template-columns: 1fr 320px; min-height: calc(100vh - 58px); }}
  canvas {{ width: 100%; height: calc(100vh - 58px); display: block; }}
  aside {{ border-left: 1px solid #243140; padding: 12px 16px; overflow: auto; }}
  a {{ color: #8ecaff; }}
  h2 {{ font-size: 16px; margin: 0 0 8px; }}
  ol {{ padding-left: 18px; }}
  @media (max-width: 800px) {{
    main {{ grid-template-columns: 1fr; }}
    canvas {{ height: 70vh; }}
    aside {{ border-left: 0; border-top: 1px solid #243140; }}
  }}
</style>
</head>
<body>
<header>
  <strong>Термины курса</strong>
  <select id="course"></select>
  <input id="filter" placeholder="фильтр термина"/>
  <span id="stats"></span>
</header>
<main>
  <canvas id="graph"></canvas>
  <aside id="side"></aside>
</main>
<script id="graph-data" type="application/json">{payload}</script>
<script>
{script}
boot(JSON.parse(document.getElementById("graph-data").textContent));
</script>
</body>
</html>
"""


def write_html(graph: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(graph), encoding="utf-8")
    return path


def graph_stats(graph: dict) -> dict:
    courses = graph.get("courses") or []
    terms = sum(len(course["terms"]) for course in courses)
    edges = sum(len(course["edges"]) for course in courses)
    weights = [edge["jaccard"] for course in courses for edge in course["edges"]]
    mean = sum(weights) / len(weights) if weights else 0.0
    return {
        "courses": len(courses),
        "terms": terms,
        "edges": edges,
        "mean_jaccard": round(mean, 4),
    }


def _course_graph(name, lectures, global_df, corpus, *, per_course, min_jaccard, max_edges_per_node):
    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()
    where: dict[str, list[str]] = defaultdict(list)
    for lecture in lectures:
        counts = lecture["counts"]
        tf.update(counts)
        for lemma in counts:
            df[lemma] += 1
            where[lemma].append(lecture["slug"])
    n = len(lectures)
    min_tf = 3 if n >= 3 else 2
    min_df = 2 if n >= 4 else 1
    ranked = []
    for lemma, total in tf.items():
        if total < min_tf or df[lemma] < min_df:
            continue
        if _too_common(global_df[lemma], corpus):
            continue
        idf = math.log((corpus + 1) / (global_df[lemma] + 1)) + 1
        ranked.append((total * idf, total, lemma))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    chosen = [lemma for _score, _total, lemma in ranked[:per_course]]
    chosen_set = set(chosen)
    co: Counter[tuple[str, str]] = Counter()
    for lecture in lectures:
        present = sorted(lemma for lemma in lecture["counts"] if lemma in chosen_set)
        for left, right in combinations(present, 2):
            co[(left, right)] += 1
    edges = _edges(co, df, chosen_set, min_jaccard, max_edges_per_node)
    return {
        "name": name,
        "lectures": [
            {
                "slug": lecture["slug"],
                "title": lecture["title"],
                "number": lecture["number"],
                "url": lecture["url"],
            }
            for lecture in sorted(lectures, key=lambda item: (item["number"] is None, item["number"] or 0, item["title"]))
        ],
        "terms": [
            {
                "lemma": lemma,
                "count": int(tf[lemma]),
                "df": int(df[lemma]),
                "lectures": where[lemma],
            }
            for lemma in chosen
        ],
        "edges": edges,
    }


def _too_common(df: int, corpus: int) -> bool:
    if corpus < 30:
        return False
    return df / corpus >= 0.4


def _edges(co, df, chosen, min_jaccard, max_edges_per_node):
    raw = []
    for (left, right), weight in co.items():
        union = df[left] + df[right] - weight
        if union <= 0:
            continue
        jaccard = weight / union
        if jaccard < min_jaccard:
            continue
        raw.append((jaccard, weight, left, right))
    raw.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    used: Counter[str] = Counter()
    edges = []
    for jaccard, weight, left, right in raw:
        if left not in chosen or right not in chosen:
            continue
        if used[left] >= max_edges_per_node or used[right] >= max_edges_per_node:
            continue
        used[left] += 1
        used[right] += 1
        edges.append(
            {
                "source": left,
                "target": right,
                "weight": int(weight),
                "jaccard": round(jaccard, 4),
            }
        )
    return edges

from whisper_fpmi.cloud import render_svg, top_counts
from whisper_fpmi.dataset import make_record, write_csv
from whisper_fpmi.descriptions import description_from_payload
from whisper_fpmi.evaluate import retrieval_report
from whisper_fpmi.lang import analyze, stem_token, usable_phrase
from whisper_fpmi.rag import extractive_answer, grounded_fraction
from whisper_fpmi.search import Chunk, build_chunks, build_index, search, vk_link
from whisper_fpmi.terms import build_graph, render_html
from whisper_fpmi.timecodes import parse_description_timecodes, parse_lecturer
from whisper_fpmi.transcript import LectureText, Segment, parse_clock, parse_segments


def test_hour_timecode_is_not_minutes():
    text = "\n".join(
        [
            "Таймкоды:",
            "00:00:00 - Начало",
            "1:05:30 - После часа",
            "01:16:02 — Ещё одна глава",
            "2:15:04 - Второй час",
            "65:21 Долгие минуты",
            "12:04 Короткая метка",
            "Длительность: 1:16:38",
            "Семинарист: Евстигнеев Георгий Дмитриевич",
            "Дата допсема: 06.04.2026",
        ]
    )
    codes = parse_description_timecodes(text)
    by_title = {item.title: item.seconds for item in codes}
    assert by_title["После часа"] == 1 * 3600 + 5 * 60 + 30
    assert by_title["Ещё одна глава"] == 1 * 3600 + 16 * 60 + 2
    assert by_title["Второй час"] == 2 * 3600 + 15 * 60 + 4
    assert by_title["Долгие минуты"] == 65 * 60 + 21
    assert by_title["Короткая метка"] == 12 * 60 + 4
    assert "1:16:38" not in by_title
    assert all(item.hour_field for item in codes if item.title == "После часа")
    assert parse_lecturer(text) == "Евстигнеев Георгий Дмитриевич"


def test_html_description_keeps_hour_marks():
    payload = {
        "payload": [
            0,
            {
                "desc": "Таймкоды:<br>1:05:30 - После часа<br>Лектор: Иванов Иван",
            },
        ]
    }
    text = description_from_payload(payload)
    codes = parse_description_timecodes(text)
    assert codes[0].seconds == 3930
    assert codes[0].title == "После часа"
    assert parse_lecturer(text) == "Иванов Иван"


def test_fillers_are_separate_from_content():
    stats = analyze("Ну, давайте, это функция и предел. Как бы предел.")
    assert stats.filler_counts.get("ну") == 1
    assert stats.filler_counts.get("давайте") == 1
    assert stats.filler_counts.get("как бы") == 1
    assert stats.function_words >= 1
    assert "предел" in stats.counts or any("предел" in key for key in stats.counts)
    assert "ну" not in stats.counts
    assert usable_phrase("Начало") is False
    assert usable_phrase("частично определенные функции") is True


def test_clock_with_hours_and_millis():
    assert parse_clock("01:02:03.250") == 3723.25
    segments = parse_segments("[01:05:30.000 --> 01:06:00.000]  после часа про жорданову форму")
    assert segments[0].start == 3930


def test_dataset_record_stores_hour_timecodes(tmp_path):
    plain = tmp_path / "Алгебра-5-Жорданова-форма.txt"
    plain.write_text(
        "\n".join(
            [
                "# Название: Алгебра 5. Жорданова форма",
                "# Источник: https://vk.com/video-206078025_456241617",
                "# Модель: whisper large-v3",
                "# Язык: ru",
                "# Дата: 2026-04-06T00:00:00Z",
                "",
                "Жорданова форма и матрица. Ну, давайте разберём жорданову клетку.",
            ]
        ),
        encoding="utf-8",
    )
    lecture = LectureText(
        slug=plain.stem,
        plain_path=plain,
        timed_path=None,
        header={
            "Название": "Алгебра 5. Жорданова форма",
            "Источник": "https://vk.com/video-206078025_456241617",
            "Модель": "whisper large-v3",
            "Язык": "ru",
            "Дата": "2026-04-06T00:00:00Z",
        },
        text="Жорданова форма и матрица. Ну, давайте разберём жорданову клетку.",
        segments=[Segment(0, 30, "Жорданова форма"), Segment(3930, 4000, "клетка после часа")],
        video_id="-206078025_456241617",
    )
    record, _stats = make_record(
        lecture,
        description="Таймкоды:\n1:05:30 - Жорданова клетка\nЛектор: Петров Пётр",
        vk_duration=4000,
        root=tmp_path,
    )
    assert record["timecode_count"] == 1
    assert record["timecodes"][0]["seconds"] == 3930
    assert record["timecodes_past_one_hour"] == 1
    assert record["lecturer"] == "Петров Пётр"
    assert record["duration_sec"] == 4000
    assert record["fillers"] >= 1
    assert record["video_id"] == "-206078025_456241617"


def test_search_finds_the_chapter_and_not_the_other_course():
    algebra = LectureText(
        slug="algebra",
        plain_path=None,
        timed_path=None,
        header={"Источник": "https://vk.com/video-1_1"},
        text="",
        segments=[
            Segment(0, 40, "вступление про поле"),
            Segment(3930, 4200, "жорданова клетка и жорданова форма матрицы"),
        ],
    )
    logic = LectureText(
        slug="logic",
        plain_path=None,
        timed_path=None,
        header={"Источник": "https://vk.com/video-1_2"},
        text="",
        segments=[Segment(10, 80, "разрешимость и перечислимость множеств")],
    )
    chunks = []
    chunks.extend(build_chunks(algebra, title="5. Жорданова форма", course="Алгебра"))
    chunks.extend(build_chunks(logic, title="3. Вычислимость", course="Матлогика"))
    index = build_index(chunks, stem_token, "suffix")
    hits = search(index, "жорданова клетка", stem_token, limit=3)
    assert hits[0].chunk.slug == "algebra"
    assert hits[0].chunk.start >= 3600
    assert vk_link(hits[0].chunk.url, hits[0].chunk.start).endswith("t=1h5m30s")


def test_rag_citation_is_grounded_in_the_hit():
    chunk = Chunk(
        slug="algebra",
        course="Алгебра",
        title="5. Жорданова форма",
        url="https://vk.com/video-1_1",
        start=3930,
        end=4200,
        text="Жорданова клетка стоит на диагонали матрицы.",
        title_lemmas=("жордан",),
    )
    index = build_index([chunk], stem_token, "suffix")
    hits = search(index, "жорданова клетка", stem_token)
    answer = extractive_answer("жорданова клетка", hits, stem_token)
    assert answer.support_slug == "algebra"
    assert answer.support_start == 3930
    assert "[" in answer.answer
    assert grounded_fraction(answer.answer, answer.citations, stem_token) == 1


def test_retrieval_metrics_on_a_labeled_chapter():
    lecture = LectureText(
        slug="algebra",
        plain_path=None,
        timed_path=None,
        header={},
        text="",
        segments=[Segment(3930, 4200, "жорданова клетка и жорданова форма матрицы")],
    )
    chunks = build_chunks(lecture, title="5. Жорданова форма", course="Алгебра")
    index = build_index(chunks, stem_token, "suffix")
    records = [
        {
            "slug": "algebra",
            "course": "Алгебра",
            "title": "5. Жорданова форма",
            "duration_sec": 4200,
            "timecodes": [{"seconds": 3930, "title": "Жорданова клетка", "hour_field": True}],
        }
    ]
    report = retrieval_report(index, records, stem_token)
    assert report["chapters"]["hit_rate_at_1"] == 1
    assert report["chapters"]["boundary_hit_at_90s_at_5"] == 1
    assert report["titles"]["hit_rate_at_1"] == 1
    assert report["rag"]["support_hit_rate"] == 1
    assert report["rag"]["grounded"] == 1


def test_term_graph_connects_words_of_one_course():
    rows = [
        {
            "course": "Алгебра",
            "slug": "a1",
            "title": "1. Клетки",
            "number": 1,
            "url": "https://vk.com/video-1_1",
            "counts": {"жордан": 4, "матрица": 5, "клетка": 3},
        },
        {
            "course": "Алгебра",
            "slug": "a2",
            "title": "2. Форма",
            "number": 2,
            "url": "",
            "counts": {"жордан": 6, "матрица": 4, "оператор": 5},
        },
        {
            "course": "Логика",
            "slug": "l1",
            "title": "1. Модели",
            "number": 1,
            "url": "",
            "counts": {"модель": 4, "формула": 4},
        },
    ]
    graph = build_graph(rows, per_course=10, min_jaccard=0.2)
    algebra = next(course for course in graph["courses"] if course["name"] == "Алгебра")
    lemmas = {term["lemma"] for term in algebra["terms"]}
    assert "жордан" in lemmas and "матрица" in lemmas
    assert any(edge["source"] == "жордан" and edge["target"] == "матрица" for edge in algebra["edges"])
    page = render_html(graph)
    assert "Алгебра" in page
    assert "boot(" in page
    assert "<script" in page


def test_csv_keeps_hour_timecode_and_commas(tmp_path):
    import csv

    path = tmp_path / "lectures.csv"
    write_csv(
        [
            {
                "slug": "algebra",
                "title": "5. Жорданова форма",
                "course": "Алгебра, геометрия",
                "lecturer": "Иванов Иван",
                "timecodes": [
                    {"seconds": 3930, "clock": "01:05:30", "title": "После часа, клетка", "hour_field": True}
                ],
                "timecode_count": 1,
                "timecodes_past_one_hour": 1,
                "filler_counts": {"значит": 2, "давайте": 1},
                "description": "строка\nвторая",
                "duration_sec": 4000,
            }
        ],
        path,
    )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["course"] == "Алгебра, геометрия"
    assert rows[0]["timecodes"] == "01:05:30 После часа, клетка"
    assert rows[0]["filler_counts"] == "давайте=1; значит=2"
    assert "вторая" in rows[0]["description"]
    assert rows[0]["duration_sec"] == "4000"


def test_wordcloud_skips_empty_and_draws_the_top_word():
    svg = render_svg(top_counts({"жордан": 10, "матрица": 3}))
    assert "жордан" in svg
    assert "матрица" in svg
    assert "#10151c" in svg

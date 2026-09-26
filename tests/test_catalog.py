from whisper_fpmi.catalog import group_by_course, render_catalog
from whisper_fpmi.names import format_timestamp, parse_stem


def test_format_timestamp_under_hour_matches_whisper_cli():
    assert format_timestamp(0) == "00:00.000"
    assert format_timestamp(12) == "00:12.000"
    assert format_timestamp(62.5) == "01:02.500"


def test_format_timestamp_with_hours():
    assert format_timestamp(3661.25) == "01:01:01.250"


def test_group_by_course_keeps_related_lectures_together():
    lectures = [
        parse_stem("C++-2-Ссылки-и-константы"),
        parse_stem("C++-1-stdmap-stdlist"),
        parse_stem("Highload-6-Очередь"),
    ]
    grouped = group_by_course(lectures)
    assert list(grouped["C++"])[0].number == 1
    assert "Highload" in grouped


def test_render_catalog_contains_links_to_result(tmp_path, monkeypatch):
    result = tmp_path / "result"
    result.mkdir()
    (result / "C++-1-stdmap-stdlist.txt").write_text("текст", encoding="utf-8")
    monkeypatch.setattr("whisper_fpmi.catalog.RESULT_DIR", result)
    monkeypatch.setattr("whisper_fpmi.catalog.TIMED_DIR", tmp_path / "download")
    monkeypatch.setattr("whisper_fpmi.catalog.ROOT", tmp_path)
    markdown = render_catalog()
    assert "C++-1-stdmap-stdlist.txt" in markdown
    assert "## C++" in markdown

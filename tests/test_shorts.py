from pathlib import Path
import json
import shutil
import subprocess
import threading
import time

from whisper_fpmi.lang import stem_token
from whisper_fpmi.shorts import (
    BEST_FORMAT,
    STYLES,
    BuildProgress,
    Clip,
    Hit,
    _transient,
    clip_window,
    collect,
    cue_word,
    ensure_index,
    ensure_space,
    load_popular_words,
    level_gain,
    even_sample,
    fetch_sources,
    find_font,
    make_settings,
    page_url,
    phrase_spans,
    plan_clips,
    plan_shorts,
    vk_url_from_id,
    preview_payload,
    query_tokens,
    render_short,
)


def _timed(directory: Path, slug: str, lines: list[str], title: str, url: str) -> None:
    path = directory / f"{slug}.mp4.txt"
    header = [
        f"# Название: {title}",
        f"# Источник: {url}",
        "# Модель: whisper large-v3",
        "",
    ]
    path.write_text("\n".join(header + lines) + "\n", encoding="utf-8")


def _catalog(tmp_path: Path):
    timed = tmp_path / "download"
    timed.mkdir()
    _timed(
        timed,
        "Алгебра-и-геометрия-1.-Жордан",
        [
            "[00:01.000 --> 00:01.200]  Жорданова",
            "[00:01.200 --> 00:01.500]  форма.",
            "[00:08.000 --> 00:08.200]  алгоритм",
            "[00:12.000 --> 00:12.300]  алгоритмов",
            "[00:20.000 --> 00:20.200]  и",
            "[00:21.000 --> 00:21.200]  и",
            "[00:30.000 --> 00:30.400]  просто",
            "[00:30.400 --> 00:30.700]  форма",
        ],
        "Жорданова форма",
        "https://vk.com/video-1_2",
    )
    _timed(
        timed,
        "Анализ-1.-Предел",
        [
            "[00:03.000 --> 00:03.200]  алгоритм",
            "[00:04.000 --> 00:04.500]  предел",
        ],
        "Предел",
        "https://vk.com/video-1_3",
    )
    return ensure_index(
        timed,
        tmp_path / "shorts.sqlite",
        lemmatize=stem_token,
        lemmatizer_name="stem",
    )


def test_cue_word_strips_punctuation_and_keeps_hyphen():
    assert cue_word("день.") == "день"
    assert cue_word("каким-то") == "каким-то"
    assert cue_word("это уже фраза") is None
    assert query_tokens("  Жорданова   форма. ") == ["жорданова", "форма"]


def test_even_sample_keeps_ends():
    assert even_sample([1, 2, 3, 4, 5], 3) == [1, 3, 5]
    assert even_sample([1, 2, 3, 4], 1) == [3]
    assert even_sample([1, 2], 5) == [1, 2]


def test_clip_window_stays_inside_style_limits():
    tight = STYLES["tight"]
    start, end = clip_window(10.0, 10.2, tight)
    assert round(start, 2) == 9.86
    assert round(end, 2) == 10.48
    start, end = clip_window(5.0, 8.0, tight)
    assert end - start == tight.max_clip


def test_phrase_requires_words_in_order_and_close():
    cues = [
        (1.0, 1.2, "жорданова"),
        (1.2, 1.5, "форма"),
        (10.0, 10.2, "форма"),
        (20.0, 20.2, "жорданова"),
        (23.0, 23.2, "форма"),
    ]
    spans = phrase_spans(cues, ["жорданова", "форма"], lambda word: word)
    assert spans == [(1.0, 1.5)]


def test_plan_keeps_one_lecture_together():
    hits = []
    for slug in ("a", "b", "c"):
        for start in (1.0, 20.0, 40.0, 60.0):
            hits.append(Hit(slug, slug, slug, "", start, start + 0.2, "граф"))
    hits.append(Hit("a", "a", "a", "", 1.3, 1.5, "граф"))
    clips = plan_clips(
        hits,
        style=STYLES["tight"],
        target_seconds=30,
        max_per_lecture=2,
        seed=1,
        intro_seconds=0,
    )
    assert len(clips) == 6
    assert {clip.hit.slug for clip in clips} == {"a", "b", "c"}
    assert all(clip.hit.slug != "a" or clip.hit.start != 1.3 for clip in clips)
    order = []
    for clip in clips:
        if not order or order[-1] != clip.hit.slug:
            order.append(clip.hit.slug)
    assert order[0] == "a"
    assert len(order) == 3


def test_same_lecturer_stays_together_and_louder_word_comes_first():
    hits = []
    for slug in ("low", "high", "other"):
        hits.append(Hit(slug, slug, "Курс", "", 10.0, 10.2, "граф"))
        hits.append(Hit(slug, slug, "Курс", "", 40.0, 40.2, "граф"))
    videos = plan_shorts(
        hits,
        style=STYLES["tight"],
        target_seconds=30,
        max_per_lecture=2,
        seed=1,
        intro_seconds=0,
        counts={"low": 3, "high": 10, "other": 8},
        lecturers={"low": "Иванов", "high": "Иванов", "other": "Петров"},
        max_videos=1,
    )
    order = []
    for clip in videos[0]:
        if not order or order[-1] != clip.hit.slug:
            order.append(clip.hit.slug)
    assert order == ["high", "low", "other"]


def test_level_gain_meets_one_loudness():
    assert level_gain(-28) == 12
    assert level_gain(-10) == -6
    assert level_gain(-60) == 0
    assert level_gain(-1) == -15
    assert level_gain(-40) == 18


def test_plan_shorts_puts_frequent_lectures_first_and_caps_at_five():
    hits = []
    for index in range(8):
        slug = f"l{index:02d}"
        hits.append(Hit(slug, slug, "Курс", "", 10.0, 10.2, "граф"))
        hits.append(Hit(slug, slug, "Курс", "", 40.0, 40.2, "граф"))
    for start in (5.0, 15.0, 25.0, 35.0):
        hits.append(Hit("l99", "l99", "Курс", "", start, start + 0.2, "граф"))
    videos = plan_shorts(
        hits,
        style=STYLES["tight"],
        target_seconds=2,
        max_per_lecture=4,
        seed=1,
        intro_seconds=0,
        counts={"l99": 40},
    )
    assert len(videos) == 5
    assert videos[0][0].hit.slug == "l99"


def test_plan_shorts_offers_several_full_videos():
    hits = []
    for index in range(6):
        slug = f"lecture-{index}"
        hits.append(Hit(slug, slug, "Курс", "", 10.0, 10.2, "граф"))
        hits.append(Hit(slug, slug, "Курс", "", 30.0, 30.2, "граф"))
    videos = plan_shorts(
        hits,
        style=STYLES["tight"],
        target_seconds=3,
        max_per_lecture=2,
        seed=1,
        intro_seconds=0,
    )
    assert len(videos) >= 2
    for video in videos:
        order = []
        for clip in video:
            if not order or order[-1] != clip.hit.slug:
                order.append(clip.hit.slug)
        assert len(order) == len(set(order))


def test_index_counts_forms_and_skips_function_words(tmp_path: Path):
    catalog = _catalog(tmp_path)
    forms = collect(catalog, make_settings("алгоритмов", mode="forms"))
    assert forms.total == 3
    assert forms.lectures == 2
    assert {hit.label for hit in forms.hits} == {"алгоритм", "алгоритмов"}
    exact = collect(catalog, make_settings("алгоритмов", mode="exact"))
    assert exact.total == 1
    assert len(exact.hits) == 1
    stop = collect(catalog, make_settings("и"))
    assert stop.total == 2
    assert stop.hits == []
    assert stop.indexed is False


def test_course_filter_and_phrase(tmp_path: Path):
    catalog = _catalog(tmp_path)
    algebra = collect(catalog, make_settings("алгоритм", mode="forms", course="Алгебра"))
    assert algebra.lectures == 1
    assert {hit.slug for hit in algebra.hits} == {"Алгебра-и-геометрия-1.-Жордан"}
    phrase = collect(catalog, make_settings("жорданова форма", mode="exact"))
    assert phrase.total == 1
    assert phrase.hits[0].start == 1.0
    assert phrase.hits[0].end == 1.5
    payload = preview_payload(catalog, make_settings("алгоритм", mode="exact", seconds=20, seed=2))
    assert payload["planned"] >= 1
    assert payload["clips"]


def test_new_transcript_rebuilds_index(tmp_path: Path):
    catalog = _catalog(tmp_path)
    before = collect(catalog, make_settings("предел", mode="exact")).total
    assert before == 1
    _timed(
        catalog.timed_dir,
        "Анализ-2.-Ещё",
        ["[00:02.000 --> 00:02.200]  предел"],
        "Ещё предел",
        "https://vk.com/video-1_4",
    )
    rebuilt = ensure_index(
        catalog.timed_dir,
        catalog.path,
        lemmatize=stem_token,
        lemmatizer_name="stem",
    )
    assert collect(rebuilt, make_settings("предел", mode="exact")).total == 2


def test_vk_id_becomes_page_url():
    assert vk_url_from_id("-206078025_456239146") == "https://vk.com/video-206078025_456239146"
    assert vk_url_from_id("dQw4w9WgXcQ") == ""
    assert page_url("", "-1_2") == "https://vk.com/video-1_2"
    assert page_url("https://vk.com/video-1_2", "") == "https://vk.com/video-1_2"
    assert _transient(RuntimeError("Read timed out"))
    assert not _transient(RuntimeError("Requested format is not available"))


def test_progress_reports_download_cut_and_eta():
    progress = BuildProgress()
    hit = Hit("a", "Лекция", "Курс", "https://vk.com/video-1_2", 1.0, 1.2, "граф")
    progress.begin_downloads([hit])
    progress.tick_download("a", "video", 50, 100, "B", False)
    progress.started = time.monotonic() - 30
    snap = progress.snapshot()
    assert snap["phase"] == "download"
    assert 0 < snap["percent"] < 90
    assert snap["eta_seconds"] > 0
    assert "50 Б" in snap["items"][0]["detail"]
    progress.finish_download("a", False, "VK не ответил вовремя")
    assert progress.snapshot()["items"][0]["detail"] == "VK не ответил вовремя"
    progress.begin_cut(4)
    progress.tick_cut(2, "Лекция")
    assert progress.snapshot()["phase"] == "cut"
    progress.begin_assemble()
    assert progress.snapshot()["percent"] >= 90
    progress.finish()
    assert progress.snapshot()["percent"] == 100


def test_missing_lectures_download_at_once(tmp_path: Path):
    folder = tmp_path / "sources"
    folder.mkdir()
    (folder / "ready.mp4").write_bytes(b"already")
    hits = [
        Hit("ready", "Уже есть", "Курс", "https://vk.com/video-1_1", 1.0, 1.2, "граф"),
        Hit("one", "Один", "Курс", "https://vk.com/video-1_2", 2.0, 2.2, "граф"),
        Hit("two", "Два", "Курс", "https://vk.com/video-1_3", 3.0, 3.2, "граф"),
        Hit("one", "Один", "Курс", "https://vk.com/video-1_2", 8.0, 8.2, "граф"),
    ]
    clips = [Clip(hit, hit.start, hit.end) for hit in hits]
    started: list[str] = []
    barrier = threading.Barrier(2)

    def fake(hit: Hit, dest: Path, cookies: str | None) -> Path:
        assert cookies is None
        assert "height" not in BEST_FORMAT
        started.append(hit.slug)
        barrier.wait(timeout=2)
        path = dest / f"{hit.slug}.mp4"
        path.write_bytes(b"video")
        return path

    ready = fetch_sources(clips, folder, None, download=fake)
    assert set(started) == {"one", "two"}
    assert ready["ready"].read_bytes() == b"already"
    assert ready["one"].read_bytes() == b"video"
    assert ready["two"].read_bytes() == b"video"


def test_popular_words_follow_the_cloud_order(tmp_path: Path):
    path = tmp_path / "cloud.json"
    path.write_text(
        json.dumps({"words": [{"text": "функция", "count": 10}, {"text": "точка"}, {"text": ""}]}),
        encoding="utf-8",
    )
    assert load_popular_words(path) == ["функция", "точка"]


def test_low_disk_removes_only_sources_that_will_not_be_reused(tmp_path: Path):
    folder = tmp_path / "sources"
    folder.mkdir()
    keep = folder / "keep.mp4"
    drop = folder / "drop.mp4"
    keep.write_bytes(b"keep")
    drop.write_bytes(b"drop-me")
    assert ensure_space(folder, {"keep"}, reserve=100, free=lambda: 10**12) == []
    assert drop.exists()
    removed = ensure_space(folder, {"keep"}, reserve=100, free=lambda: 1)
    assert removed == [drop]
    assert keep.exists()
    assert not drop.exists()


def test_render_from_local_file_when_ffmpeg_is_available(tmp_path: Path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None or find_font() is None:
        return
    catalog = _catalog(tmp_path)
    slug = "Анализ-1.-Предел"
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source = source_dir / f"{slug}.mp4"
    probe = shutil.which("ffmpeg")
    code = subprocess.run(
        [
            probe, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=5",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=5",
            "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(source),
        ],
        check=False,
    ).returncode
    assert code == 0
    found = collect(catalog, make_settings("предел", mode="exact"))
    clips = plan_clips(
        found.hits,
        style=STYLES["tight"],
        target_seconds=12,
        max_per_lecture=1,
        seed=1,
        intro_seconds=1.7,
    )
    assert clips
    output = tmp_path / "out.mp4"
    rendered = render_short(clips, label="ПРЕДЕЛ", meta_line="1 фрагмент", output=output, source_dir=source_dir)
    assert rendered.path.is_file()
    assert rendered.path.stat().st_size > 10_000
    assert rendered.clips == 1

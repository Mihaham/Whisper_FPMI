from pathlib import Path

from whisper_fpmi.pipeline import _already_done
from whisper_fpmi.transcript import assess_timed, classify_corpus, write_retranscribe_list
from whisper_fpmi.vk import ChannelVideo


def _word_file(path: Path, *, model: str = "whisper large-v3") -> None:
    lines = [
        "# Название: Тест",
        f"# Модель: {model}",
        "",
    ]
    for index in range(12):
        lines.append(f"[00:{index:02d}.000 --> 00:{index:02d}.400]  слово")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _phrase_file(path: Path, *, header: bool) -> None:
    lines = []
    if header:
        lines.extend(
            [
                "# Название: Тест",
                "# Модель: whisper large-v3",
                "",
            ]
        )
    for index in range(12):
        lines.append(
            f"[00:{index:02d}.000 --> 00:{index:02d}.800]  это уже целая фраза лекции"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_word_level_large_v3_is_final(tmp_path: Path):
    path = tmp_path / "lecture.mp4.txt"
    _word_file(path)
    assert assess_timed(path) == "final"


def test_phrase_timestamps_need_retranscribe(tmp_path: Path):
    path = tmp_path / "lecture.mp4.txt"
    _phrase_file(path, header=True)
    assert assess_timed(path) == "segment-timestamps"


def test_missing_model_header_needs_retranscribe(tmp_path: Path):
    path = tmp_path / "lecture.mp4.txt"
    _phrase_file(path, header=False)
    assert assess_timed(path) == "no-model"


def test_other_model_needs_retranscribe(tmp_path: Path):
    path = tmp_path / "lecture.mp4.txt"
    _word_file(path, model="whisper medium")
    assert assess_timed(path) == "other-model"


def test_classify_corpus_keeps_only_final(tmp_path: Path):
    result = tmp_path / "result"
    timed = tmp_path / "download"
    result.mkdir()
    timed.mkdir()
    (result / "good.txt").write_text("текст", encoding="utf-8")
    (result / "old.txt").write_text("текст", encoding="utf-8")
    _word_file(timed / "good.mp4.txt")
    _phrase_file(timed / "old.mp4.txt", header=False)

    done, redo = classify_corpus(result, timed)
    assert done == {"good"}
    assert redo == [("old", "no-model")]

    listing = tmp_path / "retranscribe.txt"
    write_retranscribe_list(redo, listing)
    text = listing.read_text(encoding="utf-8")
    assert "old\tнет шапки модели" in text
    assert "good" not in text.splitlines()[-1]


def test_already_done_rejects_old_transcript(tmp_path: Path, monkeypatch):
    timed = tmp_path / "download"
    timed.mkdir()
    _phrase_file(timed / "old-lecture.mp4.txt", header=False)
    monkeypatch.setattr("whisper_fpmi.pipeline.TIMED_DIR", timed)
    video = ChannelVideo("1_2", "old lecture", "https://vk.com/video1_2")
    state = {"videos": {"1_2": {"slug": "old-lecture"}}}
    assert _already_done(video, state, set()) is False


def test_known_filename_matches_youtube_id():
    from whisper_fpmi.pipeline import _known_by_filename

    video = ChannelVideo("z8p4k4l-oU0", "Методы", "https://www.youtube.com/watch?v=z8p4k4l-oU0")
    found = _known_by_filename("z8p4k4l-oU0_Методы-оптимизации", {"z8p4k4l-oU0": video})
    assert found is video
    assert _known_by_filename("other", {"z8p4k4l-oU0": video}) is None


def test_already_done_accepts_final_transcript(tmp_path: Path, monkeypatch):
    timed = tmp_path / "download"
    timed.mkdir()
    _word_file(timed / "new-lecture.mp4.txt")
    monkeypatch.setattr("whisper_fpmi.pipeline.TIMED_DIR", timed)
    video = ChannelVideo("1_2", "Другое название", "https://vk.com/video1_2")
    state = {"videos": {"1_2": {"slug": "new-lecture"}}}
    assert _already_done(video, state, set()) is True

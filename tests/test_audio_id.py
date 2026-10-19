from pathlib import Path
import random
import subprocess
import threading
import time

import pytest

from whisper_fpmi.audio_id import (
    MATCH_SIMILARITY,
    AudioIndex,
    final_slug_by_fold,
    similarity,
    transcript_slug,
)
from whisper_fpmi.dataset import attach_lecture_copies
from whisper_fpmi.state import mark_done
from whisper_fpmi.vk import ChannelVideo


def _tone(seed: int, length: int = 120) -> tuple[int, ...]:
    rng = random.Random(seed)
    return tuple(rng.getrandbits(32) for _ in range(length))


def _word_file(path: Path) -> None:
    lines = ["# Название: Тест", "# Модель: whisper large-v3", ""]
    for index in range(12):
        lines.append(f"[00:{index:02d}.000 --> 00:{index:02d}.400]  слово")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_identical_fingerprint_scores_one_and_noise_does_not():
    tone = _tone(1)
    assert similarity(tone, tone) == 1
    assert similarity(tone, _tone(2)) < MATCH_SIMILARITY
    assert similarity(tone[:10], tone[:10]) == 0


def test_same_audio_links_to_the_existing_transcript(tmp_path: Path):
    root = tmp_path / "repo"
    index = AudioIndex(tmp_path / "audio-index.json", root=root)
    tone = _tone(3)
    original = ChannelVideo("1_1", "Лекция", "https://vk.com/video1_1", duration=3600)
    assert index.consider(tone, 3600, original).transcribe
    _word_file(root / "download" / "lekciya.mp4.txt")
    index.commit(original, "lekciya")

    copy = ChannelVideo("yt1", "Та же лекция на YouTube", "https://youtu.be/yt1", duration=3602)
    decision = index.consider(tone, 3602, copy)
    assert decision.transcribe is False
    assert decision.slug == "lekciya"
    assert decision.plain == "result/lekciya.txt"
    assert decision.same_audio_as == "1_1"
    assert index.linked_slug("yt1") == "lekciya"
    assert index.same_audio_as("yt1") == "1_1"

    loaded = AudioIndex(tmp_path / "audio-index.json", root=root)
    again = loaded.consider(
        tone,
        3600,
        ChannelVideo("yt2", "Ещё копия", "https://youtu.be/yt2", duration=3590),
    )
    assert again.transcribe is False
    assert again.slug == "lekciya"


def test_different_length_is_not_treated_as_the_same_video(tmp_path: Path):
    index = AudioIndex(tmp_path / "audio-index.json", root=tmp_path)
    tone = _tone(4)
    first = ChannelVideo("1_1", "Длинная", "https://vk.com/video1_1", duration=1000)
    assert index.consider(tone, 1000, first).transcribe
    second = ChannelVideo("1_2", "Короткая", "https://vk.com/video1_2", duration=2000)
    holder: dict = {}

    def run() -> None:
        holder["decision"] = index.consider(tone, 2000, second)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(2)
    assert not thread.is_alive()
    assert holder["decision"].transcribe is True


def test_second_copy_waits_until_the_first_transcript_is_committed(tmp_path: Path):
    root = tmp_path / "repo"
    index = AudioIndex(tmp_path / "audio-index.json", root=root)
    tone = _tone(5)
    original = ChannelVideo("1_1", "Лекция", "https://vk.com/video1_1", duration=4000)
    assert index.consider(tone, 4000, original).transcribe
    _word_file(root / "download" / "lekciya.mp4.txt")
    copy = ChannelVideo("yt1", "Копия", "https://youtu.be/yt1", duration=4000)
    holder: dict = {}

    def run() -> None:
        holder["decision"] = index.consider(tone, 4000, copy)

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.05)
    index.commit(original, "lekciya")
    thread.join(2)
    assert not thread.is_alive()
    assert holder["decision"].transcribe is False
    assert holder["decision"].slug == "lekciya"


def test_dataset_row_keeps_its_own_url_and_points_at_the_transcript():
    records = [
        {
            "slug": "lekciya",
            "video_id": "1_1",
            "title": "1. Лекция",
            "url": "https://vk.com/video1_1",
            "plain": "result/lekciya.txt",
            "timed": "download/lekciya.mp4.txt",
            "words": 1000,
            "same_lecture_as": None,
        }
    ]
    copies = index_copies()
    rows = attach_lecture_copies(records, copies)
    assert len(rows) == 2
    assert rows[0]["video_id"] == "1_1"
    copy = rows[1]
    assert copy["video_id"] == "yt1"
    assert copy["url"] == "https://youtu.be/yt1"
    assert copy["plain"] == "result/lekciya.txt"
    assert copy["timed"] == "download/lekciya.mp4.txt"
    assert copy["words"] == 1000
    assert copy["same_lecture_as"] == "1_1"
    assert copy["transcript_url"] == "https://vk.com/video1_1"
    assert copy["upload_title"] == "Та же лекция на YouTube"
    assert attach_lecture_copies(rows, copies) == rows


def index_copies():
    from whisper_fpmi.lecture_id import LectureCopy

    return [
        LectureCopy(
            video_id="yt1",
            title="Та же лекция на YouTube",
            url="https://youtu.be/yt1",
            slug="lekciya",
            same_lecture_as="1_1",
            plain="result/lekciya.txt",
            timed="download/lekciya.mp4.txt",
        )
    ]


def test_mark_done_stores_the_shared_transcript():
    state: dict = {"videos": {}}
    mark_done(
        state,
        "yt1",
        title="Копия",
        url="https://youtu.be/yt1",
        slug="lekciya",
        model="large-v3",
        same_lecture_as="1_1",
    )
    record = state["videos"]["yt1"]
    assert record["plain"] == "result/lekciya.txt"
    assert record["timed"] == "download/lekciya.mp4.txt"
    assert record["url"] == "https://youtu.be/yt1"
    assert record["same_lecture_as"] == "1_1"


def test_transcript_slug_finds_final_file_by_title(tmp_path: Path):
    result = tmp_path / "result"
    timed = tmp_path / "download"
    result.mkdir()
    timed.mkdir()
    (result / "lekciya.txt").write_text("текст", encoding="utf-8")
    (result / "staraya.txt").write_text("текст", encoding="utf-8")
    _word_file(timed / "lekciya.mp4.txt")
    (timed / "staraya.mp4.txt").write_text("без модели\n", encoding="utf-8")

    found = final_slug_by_fold(result, timed)
    assert found == {"lekciya": "lekciya"}
    video = ChannelVideo("9_9", "lekciya", "https://vk.com/video9_9")
    assert transcript_slug(video, {"videos": {}}, found, timed) == "lekciya"
    other = ChannelVideo("8_8", "Старая", "https://vk.com/video8_8")
    assert transcript_slug(other, {"videos": {"8_8": {"slug": "staraya"}}}, found, timed) is None


def test_matching_many_unrelated_tracks_stays_fast(tmp_path: Path):
    index = AudioIndex(tmp_path / "audio-index.json", root=tmp_path)
    from whisper_fpmi.audio_id import _Track

    for number in range(200):
        index._tracks[str(number)] = _Track(
            video_id=str(number),
            title=str(number),
            url="",
            slug=str(number),
            duration=5000,
            fingerprint=_tone(1000 + number, 400),
            plain=f"result/{number}.txt",
            timed=f"download/{number}.mp4.txt",
        )
    started = time.perf_counter()
    decision = index.consider(
        _tone(7, 400),
        5000,
        ChannelVideo("new", "Новая", "https://vk.com/video1_9", duration=5000),
    )
    assert time.perf_counter() - started < 5
    assert decision.transcribe is True


def test_reencoded_audio_matches_and_other_noise_does_not(tmp_path: Path):
    if not _chromaprint_available():
        pytest.skip("в ffmpeg нет chromaprint")
    wav = tmp_path / "a.wav"
    mp3 = tmp_path / "a.mp3"
    other = tmp_path / "b.wav"
    _ffmpeg(
        ["-f", "lavfi", "-i", "anoisesrc=d=20:c=pink:r=44100:a=0.8:seed=1", str(wav)]
    )
    _ffmpeg(["-i", str(wav), "-c:a", "libmp3lame", "-b:a", "32k", str(mp3)])
    _ffmpeg(
        ["-f", "lavfi", "-i", "anoisesrc=d=20:c=pink:r=44100:a=0.8:seed=2", str(other)]
    )
    root = tmp_path / "repo"
    index = AudioIndex(tmp_path / "audio-index.json", root=root)
    original = ChannelVideo("1_1", "Лекция", "https://vk.com/video1_1", duration=20)
    assert index.prepare(wav, original).transcribe
    _word_file(root / "download" / "Лекция.mp4.txt")
    index.commit(original, "Лекция")
    copy = index.prepare(
        mp3,
        ChannelVideo("yt", "YouTube", "https://youtu.be/yt", duration=20),
    )
    assert copy.transcribe is False
    assert copy.slug == "Лекция"
    different = index.prepare(
        other,
        ChannelVideo("1_2", "Другая", "https://vk.com/video1_2", duration=20),
    )
    assert different.transcribe is True


def _ffmpeg(args: list[str]) -> None:
    subprocess.check_call(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _chromaprint_available() -> bool:
    try:
        raw = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-formats"],
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return b"chromaprint" in raw

from datetime import datetime, timedelta
from pathlib import Path

from whisper_fpmi.shorts_publish import (
    CATEGORY_ID,
    MSK,
    ShortFile,
    _shift_past_slots,
    merge_plan,
    parse_short_name,
    plan_uploads,
    publish_at_utc,
    publish_slots,
    short_description,
    short_tags,
    short_title,
    view_score,
)


def _file(name: str, word: str, part: int, duration: float = 45.0) -> ShortFile:
    return ShortFile(name, Path(name), word, part, duration)


def test_card_is_an_education_short_about_the_word():
    text = short_description("интеграл")
    assert text.startswith("Нарезка лекций ФПМИ: каждый раз, когда лектор говорит «интеграл».")
    assert "#интеграл #лекция #мфти #shorts" in text
    assert "mihahamgenerator" not in text
    tags = short_tags("интеграл")
    assert tags[:2] == ["интеграл", "интегралы"]
    assert "мфти" in tags and "фпми" in tags
    assert CATEGORY_ID == "27"


def test_name_splits_word_and_part():
    assert parse_short_name("интеграл-1") == ("интеграл", 1)
    assert parse_short_name("заметка") is None


def test_recognizable_word_outranks_a_later_part_and_a_filler():
    assert view_score("интеграл", 1, 45) > view_score("вероятность", 1, 45)
    assert view_score("вероятность", 1, 45) > view_score("интеграл", 2, 45)
    assert view_score("интеграл", 5, 45) > view_score("более", 1, 45)
    assert view_score("интеграл", 1, 16) is None
    assert view_score("модуль", 1, 26) < view_score("модуль", 1, 45)


def test_queue_puts_the_hook_first_and_keeps_tuesday_and_friday():
    now = datetime(2026, 11, 12, 21, 0, tzinfo=MSK)
    rows = plan_uploads(
        [
            _file("более-1.mp4", "более", 1),
            _file("интеграл-2.mp4", "интеграл", 2, duration=16),
            _file("интеграл-3.mp4", "интеграл", 3),
            _file("интеграл-1.mp4", "интеграл", 1),
            _file("вероятность-1.mp4", "вероятность", 1),
            _file("теорема-1.mp4", "теорема", 1),
        ],
        now,
    )
    assert [row["file"] for row in rows] == [
        "интеграл-1.mp4",
        "вероятность-1.mp4",
        "теорема-1.mp4",
        "интеграл-3.mp4",
        "более-1.mp4",
    ]
    assert rows[3]["edition"] == 2
    assert rows[3]["title"].endswith("· 2")
    moments = [datetime.fromisoformat(row["publish_at"]) for row in rows]
    assert [(item.weekday(), item.hour, item.minute) for item in moments] == [
        (4, 10, 0),
        (1, 10, 0),
        (4, 10, 0),
        (1, 10, 0),
        (4, 10, 0),
    ]
    assert moments[0] == datetime(2026, 11, 13, 10, 0, tzinfo=MSK)
    assert moments[1] == datetime(2026, 11, 17, 10, 0, tzinfo=MSK)
    assert publish_at_utc(rows[0]["publish_at"]) == "2026-11-13T07:00:00.000Z"
    assert short_title("интеграл", 1) == "Каждый раз, когда лектор говорит «ИНТЕГРАЛ»"
    assert short_title("интеграл", 2).endswith("· 2")


def test_same_morning_is_already_too_late_for_today():
    friday = datetime(2026, 11, 13, 10, 30, tzinfo=MSK)
    first = publish_slots(1, friday)[0]
    assert first == datetime(2026, 11, 17, 10, 0, tzinfo=MSK)


def test_uploaded_slot_stays_and_a_missed_one_moves_forward():
    locked = {
        "rank": 1,
        "file": "интеграл-1.mp4",
        "word": "интеграл",
        "part": 1,
        "score": 96,
        "publish_at": "2026-11-13T10:00:00+03:00",
        "status": "scheduled",
        "youtube_id": "abc",
        "title": "t",
        "description": "d",
    }
    fresh = [
        dict(locked, youtube_id=None, status="pending"),
        {
            "rank": 2,
            "file": "вероятность-1.mp4",
            "word": "вероятность",
            "part": 1,
            "score": 93,
            "publish_at": "2026-11-17T10:00:00+03:00",
            "status": "pending",
            "youtube_id": None,
            "title": "t",
            "description": "d",
        },
    ]
    merged = merge_plan([locked], fresh)
    assert merged[0]["youtube_id"] == "abc"
    assert merged[0]["publish_at"] == "2026-11-13T10:00:00+03:00"
    assert merged[1]["file"] == "вероятность-1.mp4"
    assert merged[1]["publish_at"] == "2026-11-17T10:00:00+03:00"

    late = datetime(2026, 11, 18, 12, 0, tzinfo=MSK)
    rows = [
        {
            "file": "старый.mp4",
            "publish_at": "2026-11-13T10:00:00+03:00",
            "youtube_id": None,
        },
        {
            "file": "живой.mp4",
            "publish_at": "2026-11-20T10:00:00+03:00",
            "youtube_id": None,
        },
    ]
    _shift_past_slots(rows, late)
    assert rows[1]["publish_at"] == "2026-11-20T10:00:00+03:00"
    moved = datetime.fromisoformat(rows[0]["publish_at"])
    assert moved.weekday() in {1, 4}
    assert moved.hour == 10
    assert moved > late + timedelta(minutes=20)
    assert rows[0]["publish_at"] != rows[1]["publish_at"]

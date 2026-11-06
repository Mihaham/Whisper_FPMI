from whisper_fpmi.lecture_id import (
    DescriptionBook,
    LectureCatalog,
    LectureHit,
    lecturer_key,
    same_lecture_title,
)
from whisper_fpmi.vk import ChannelVideo

LECTURER = "Евстигнеев Георгий Дмитриевич"
DATE = "2026-04-22"


def _hit(title: str, video_id: str = "1_1", timecode_count: int = 3) -> LectureHit:
    return LectureHit(
        video_id=video_id,
        title=title,
        url=f"https://vk.com/video{video_id}",
        slug="algebra-11",
        lecturer=LECTURER,
        event_date=DATE,
        plain="result/algebra-11.txt",
        timed="download/algebra-11.mp4.txt",
        timecode_count=timecode_count,
    )


def _described(body: str) -> str:
    return (
        "00:00:00 Начало\n"
        "00:10:00 Тема\n"
        "00:20:00 Конец\n"
        "Дата лекции: 22.04.2026\n"
        "Лектор: Евстигнеев Георгий Дмитриевич\n"
        + body
    )


def test_same_lecturer_date_and_timecode_count_match_despite_language():
    catalog = LectureCatalog()
    catalog.add(_hit("Алгебра и геометрия 11. Метод Якоби", timecode_count=3))
    book = DescriptionBook({"yt1": _described("")})
    video = ChannelVideo(
        "yt1",
        "Algebra and Geometry 11. Jacobi method",
        "https://youtu.be/yt1",
    )
    hit = catalog.match_video(video, book)
    assert hit is not None
    assert hit.slug == "algebra-11"
    assert hit.plain == "result/algebra-11.txt"


def test_next_lecture_the_same_day_is_not_the_same():
    catalog = LectureCatalog()
    catalog.add(_hit("Алгебра и геометрия 11. Метод Якоби", timecode_count=3))
    book = DescriptionBook(
        {
            "2_2": (
                "00:00:00 Начало\n"
                "00:05:00 Другая тема\n"
                "Дата лекции: 22.04.2026\n"
                "Лектор: евстигнеев георгий дмитриевич\n"
            )
        }
    )
    video = ChannelVideo(
        "2_2",
        "Алгебра и геометрия 12. Ортогональные операторы",
        "https://vk.com/video2_2",
    )
    assert catalog.match_video(video, book) is None
    assert same_lecture_title(
        "Алгебра и геометрия 11. Метод Якоби",
        "Алгебра и геометрия 12. Ортогональные операторы",
    ) is False


def test_missing_date_does_not_match_by_title_alone():
    catalog = LectureCatalog()
    catalog.add(_hit("Алгебра и геометрия 11. Метод Якоби"))
    book = DescriptionBook({"3_3": "Лектор: Евстигнеев Георгий Дмитриевич\n"})
    video = ChannelVideo("3_3", "Алгебра и геометрия 11. Метод Якоби", "https://vk.com/video3_3")
    assert catalog.match_video(video, book) is None


def test_one_shared_word_does_not_make_two_lectures():
    assert same_lecture_title(
        "Теория вероятностей 5. Матожидание",
        "Теория вероятностей 5. Независимость случайных величин, матожидание",
    ) is False
    assert same_lecture_title(
        "Алгебра и геометрия 11. Метод Якоби",
        "Алгебра и геометрия 11. Метод Якоби. Критерий",
    ) is True


def test_extra_chapters_are_not_the_main_course():
    assert same_lecture_title(
        "Сложность вычислений: дополнительные главы 12. Завершение доказательства",
        "Сложность вычислений 12. Интерактивные доказательства",
    ) is False


def test_publication_within_two_weeks_is_the_same_upload():
    from whisper_fpmi.lecture_id import publish_gap, same_publication

    assert publish_gap("2026-09-01", "2026-09-03") == 2
    assert same_publication("2026-09-01", "2026-09-15") is True
    assert same_publication("2026-07-01", "2026-09-01") is False


def test_lecturer_key_ignores_order_and_case():
    assert lecturer_key("Евстигнеев Георгий Дмитриевич") == lecturer_key(
        "георгий дмитриевич евстигнеев"
    )

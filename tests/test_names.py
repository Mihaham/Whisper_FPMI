from whisper_fpmi.names import parse_stem, sanitize_title


def test_sanitize_keeps_double_dashes_from_parentheses():
    assert (
        sanitize_title("Алгоритмы и структуры данных (продвинутый поток) 3. Сортировки")
        == "Алгоритмы-и-структуры-данных--продвинутый-поток--3.-Сортировки"
    )


def test_parse_basic_course_number_title():
    parsed = parse_stem("C++-1-stdmap-stdlist")
    assert parsed.course == "C++"
    assert parsed.number == 1
    assert parsed.title == "stdmap-stdlist"
    assert parsed.display_title == "1. stdmap stdlist"
    assert parsed.display_course == "C++"


def test_parse_stream_before_number():
    parsed = parse_stem(
        "Алгоритмы-и-структуры-данных--продвинутый-поток--3-Сортировки"
    )
    assert parsed.course == "Алгоритмы-и-структуры-данных"
    assert parsed.stream == "продвинутый-поток"
    assert parsed.number == 3
    assert "продвинутый поток" in parsed.display_course


def test_parse_number_then_stream():
    parsed = parse_stem(
        "Алгоритмы-и-структуры-данных-16--базовый-поток--Выпуклая-оболочка"
    )
    assert parsed.number == 16
    assert parsed.stream == "базовый-поток"
    assert parsed.title == "Выпуклая-оболочка"


def test_parse_tftds_lecture():
    parsed = parse_stem("TFTDS--Лекция-17--HotStuff")
    assert parsed.course == "TFTDS"
    assert parsed.stream == "Лекция"
    assert parsed.number == 17
    assert parsed.title == "HotStuff"


def test_parse_tag_and_title_dashes_are_not_streams():
    parsed = parse_stem("C++-8-stdfunction--stdbind--stdany")
    assert parsed.course == "C++"
    assert parsed.number == 8
    assert parsed.stream is None
    assert parsed.title == "stdfunction--stdbind--stdany"


def test_parse_dop_seminar_tag():
    parsed = parse_stem("[Допсем]-Матлогика-14-Вычислимые-функции-Арифметическая-иерархия")
    assert parsed.tag == "[Допсем]"
    assert parsed.course == "Матлогика"
    assert parsed.number == 14
    assert parsed.display_course.startswith("[Допсем]")


def test_trailing_stream_without_parentheses():
    parsed = parse_stem("Алгоритмы-и-структуры-данных-базовый-поток-1-Асимптотика")
    assert parsed.course == "Алгоритмы-и-структуры-данных"
    assert parsed.stream == "базовый-поток"
    assert parsed.number == 1


def test_parse_dotted_lecture_number():
    parsed = parse_stem("Алгебра-и-Геометрия-11.-Метод-Грама-Шмидта.-Объём")
    assert parsed.course == "Алгебра-и-Геометрия"
    assert parsed.number == 11
    assert parsed.title.startswith("Метод-Грама")
    streamed = parse_stem(
        "Алгоритмы-и-структуры-данных--основной-поток--1.-Асимптотика,-бинарный-поиск"
    )
    assert streamed.course == "Алгоритмы-и-структуры-данных"
    assert streamed.stream == "основной-поток"
    assert streamed.number == 1


def test_fold_key_ignores_punctuation():
    from whisper_fpmi.names import fold_key

    assert fold_key("Теория групп 3. Гомоморфизмы") == fold_key(
        "Теория-групп-3-Гомоморфизмы"
    )

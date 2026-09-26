from pathlib import Path

from whisper_fpmi.quota import can_add_to_batch, dir_size, format_gib, media_files


def test_empty_batch_accepts_file_larger_than_limit():
    assert can_add_to_batch(used=0, extra=12 * 1024**3, max_bytes=10 * 1024**3, batch_count=0)


def test_full_batch_rejects_next_file():
    used = 9 * 1024**3
    extra = 2 * 1024**3
    assert not can_add_to_batch(used, extra, 10 * 1024**3, batch_count=3)


def test_fits_under_limit():
    used = 8 * 1024**3
    extra = 1 * 1024**3
    assert can_add_to_batch(used, extra, 10 * 1024**3, batch_count=2)


def test_unknown_size_keeps_filling_until_limit():
    assert can_add_to_batch(used=5 * 1024**3, extra=0, max_bytes=10 * 1024**3, batch_count=1)
    assert not can_add_to_batch(used=10 * 1024**3, extra=0, max_bytes=10 * 1024**3, batch_count=1)


def test_zero_limit_means_one_file_at_a_time():
    assert can_add_to_batch(0, 100, 0, 0)
    assert not can_add_to_batch(100, 100, 0, 1)


def test_dir_size_and_media_files(tmp_path: Path):
    (tmp_path / "a.mp4").write_bytes(b"x" * 100)
    (tmp_path / "b.txt").write_bytes(b"y" * 50)
    (tmp_path / "c.webm").write_bytes(b"z" * 20)
    assert dir_size(tmp_path) == 170
    names = {path.name for path in media_files(tmp_path)}
    assert names == {"a.mp4", "c.webm"}
    assert format_gib(1024**3) == "1.00 ГиБ"


def test_leftover_filename_roundtrip():
    from pathlib import Path

    from whisper_fpmi.pipeline import _video_from_media

    video = _video_from_media(
        Path("-206078025_456241952_Теория-групп-3-Гомоморфизмы.mp4")
    )
    assert video.video_id == "-206078025_456241952"
    assert video.title == "Теория-групп-3-Гомоморфизмы"

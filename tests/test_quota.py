from pathlib import Path

from whisper_fpmi.quota import (
    ASSUMED_FILE_BYTES,
    can_add_to_batch,
    dir_size,
    format_gib,
    media_files,
    next_quota_step,
)


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


def test_parallel_quota_fills_until_limit():
    step = next_quota_step(used=0, extra=1024**3, max_bytes=3 * 1024**3, batch_count=0)
    assert step.accept and not step.stop
    step = next_quota_step(used=step.reserved, extra=1024**3, max_bytes=3 * 1024**3, batch_count=1)
    assert step.accept and not step.stop
    step = next_quota_step(used=step.reserved, extra=1024**3, max_bytes=3 * 1024**3, batch_count=2)
    assert step.accept and step.stop
    assert step.reserved == 3 * 1024**3


def test_unknown_size_keeps_scheduling_until_limit():
    used = 0
    count = 0
    max_bytes = 3 * 1024**3
    while count < 100:
        step = next_quota_step(used=used, extra=0, max_bytes=max_bytes, batch_count=count)
        if not step.accept:
            break
        count += 1
        used = step.reserved
        if step.stop:
            break
    assert count == max_bytes // ASSUMED_FILE_BYTES
    assert used >= max_bytes


def test_over_quota_is_rejected_after_first_file():
    step = next_quota_step(used=2 * 1024**3, extra=2 * 1024**3, max_bytes=3 * 1024**3, batch_count=1)
    assert not step.accept and step.stop


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

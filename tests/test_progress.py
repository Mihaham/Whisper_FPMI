from whisper_fpmi.progress import (
    HourDayTqdm,
    ProgressState,
    audio_seconds,
    eta_seconds,
    format_audio_left,
    format_span,
)
from whisper_fpmi.vk import ChannelVideo


def test_format_span_pairs_hours_with_days():
    assert format_span(152) == "02:32"
    assert format_span(131.7 * 3600) == "131.7 ч (5.5 д)"
    assert format_audio_left(1312) == "1312 ч (54.7 д)"


def test_hour_day_bar_puts_days_into_long_eta():
    import io

    bar = HourDayTqdm(total=100, file=io.StringIO(), disable=False, mininterval=0)
    try:
        bar.start_t = bar._time() - 3600
        bar.n = 1
        text = bar.format_dict["left"]
        assert "ч" in text
        assert "д" in text
    finally:
        bar.close()


def test_overall_postfix_shows_remaining_audio_in_days():
    from whisper_fpmi.progress import RunProgress

    progress = RunProgress(total_files=2, total_audio=48 * 3600, disable=True)
    try:
        progress._sync()
        assert "48 ч (2.0 д)" in progress.overall.postfix
    finally:
        progress.close()


def test_eta_from_partial_progress():
    assert eta_seconds(elapsed=10, done=2, total=10) == 40
    assert eta_seconds(elapsed=0, done=1, total=10) is None
    assert eta_seconds(elapsed=5, done=0, total=10) is None


def test_audio_seconds_sums_durations():
    videos = [
        ChannelVideo("1", "a", "u", duration=3600),
        ChannelVideo("2", "b", "u", duration=1800),
        ChannelVideo("3", "c", "u", duration=None),
    ]
    assert audio_seconds(videos) == 5400


def test_state_tracks_video_batch_and_overall():
    state = ProgressState(total_files=3, total_audio=3 * 3600)
    state.start_batch(2, 2 * 3600)
    state.start_video("GPU", 3600)
    state.update_video("GPU", 1800, 3600)
    assert state.overall_n() == 0.5
    assert state.batch_n() == 0.5
    state.finish_video("GPU")
    assert state.done_files == 1
    assert state.overall_n() == 1.0
    assert state.batch_done_files == 1

    state.start_video("CPU-1", 3600)
    state.update_video("CPU-1", 900, 3600)
    assert abs(state.overall_n() - 1.25) < 1e-9
    assert abs(state.batch_n() - 1.0) < 1e-9


def test_cpu_video_survives_next_gpu_batch():
    state = ProgressState(total_files=4, total_audio=4 * 3600)
    state.start_batch(1, 3600)
    state.start_video("CPU-1", 3600)
    state.update_video("CPU-1", 1800, 3600)
    state.start_batch(2, 2 * 3600)
    assert state.active["CPU-1"][0] == 1800
    state.start_video("GPU", 3600)
    state.finish_video("GPU")
    assert state.batch_done_files == 1
    state.finish_video("CPU-1")
    assert state.done_files == 2
    assert state.batch_done_files == 1


def test_state_falls_back_to_files_without_audio():
    state = ProgressState(total_files=4, total_audio=0)
    state.start_batch(2, 0)
    state.start_video("GPU", 100)
    state.update_video("GPU", 50, 100)
    assert state.overall_n() == 0.5
    state.finish_video("GPU")
    assert state.overall_n() == 1.0
    assert state.overall_unit() == "файл"


def test_run_progress_updates_without_tty():
    from whisper_fpmi.progress import RunProgress

    progress = RunProgress(total_files=1, total_audio=3600, disable=True)
    try:
        video = ChannelVideo("1", "Лекция", "u", duration=3600)
        progress.start_batch(1, 1, 3600)
        progress.start_video("GPU", video)
        progress.update_video("GPU", 1800, 3600)
        assert abs(progress.state.overall_n() - 0.5) < 1e-9
        progress.finish_video("GPU")
        assert progress.state.done_files == 1
    finally:
        progress.close()

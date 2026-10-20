from pathlib import Path

from whisper_fpmi.download import (
    WORST_FORMAT,
    YOUTUBE_FORMAT,
    DownloadProgress,
    _progressive_url,
    _ydl_options,
    brief_ydl_error,
    download_parallel_http,
    format_for_url,
    parse_download_hook,
    split_ranges,
)
from whisper_fpmi.pipeline import _last_video_hooks
from whisper_fpmi.vk import ChannelVideo


def test_parse_bytes_and_stream_kind():
    tick = parse_download_hook(
        {
            "status": "downloading",
            "downloaded_bytes": 500,
            "total_bytes": 1000,
            "info_dict": {"format_id": "18", "vcodec": "avc1", "acodec": "none"},
        }
    )
    assert tick is not None
    assert tick.key == "18"
    assert tick.done == 500
    assert tick.total == 1000
    assert tick.unit == "B"
    assert tick.label == "видео"
    assert tick.finished is False


def test_parse_finished_snaps_to_total():
    tick = parse_download_hook(
        {
            "status": "finished",
            "downloaded_bytes": 900,
            "total_bytes": 1000,
            "info_dict": {"format_id": "audio", "vcodec": "none", "acodec": "mp4a"},
        }
    )
    assert tick is not None
    assert tick.finished is True
    assert tick.done == 1000
    assert tick.total == 1000
    assert tick.label == "аудио"


def test_parse_fragments_when_size_unknown():
    tick = parse_download_hook(
        {
            "status": "downloading",
            "fragment_index": 3,
            "fragment_count": 10,
            "info_dict": {"format_id": "hls"},
        }
    )
    assert tick is not None
    assert tick.unit == "frag"
    assert tick.done == 3
    assert tick.total == 10
    assert tick.label == "файл"


def test_split_ranges_covers_the_file():
    ranges = split_ranges(10, 3)
    assert ranges == [(0, 2), (3, 5), (6, 9)]
    assert split_ranges(0, 4) == []
    assert split_ranges(4, 8) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_progressive_url_skips_playlists_and_hls():
    assert _progressive_url({"url": "https://cdn/v.mp4", "protocol": "https"}) == "https://cdn/v.mp4"
    assert _progressive_url({"url": "https://cdn/v.m3u8", "protocol": "m3u8_native"}) is None
    assert _progressive_url({"url": "https://cdn/v.mp4", "requested_formats": [{}, {}]}) is None


def test_parallel_http_joins_ranges(tmp_path: Path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    payload = bytes(range(256)) * 40

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            raw = self.headers.get("Range", "")
            start_s, end_s = raw.removeprefix("bytes=").split("-")
            start, end = int(start_s), int(end_s)
            chunk = payload[start : end + 1]
            body = chunk
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        target = tmp_path / "lecture.mp4"
        download_parallel_http(
            f"http://{host}:{port}/v.mp4",
            target,
            connections=4,
        )
        assert target.read_bytes() == payload
    finally:
        server.shutdown()
        server.server_close()


def test_youtube_uses_https_audio_format():
    youtube = ChannelVideo("abc", "Лекция", "https://www.youtube.com/watch?v=abc")
    vk = ChannelVideo("1_2", "Лекция", "https://vk.com/video1_2")
    assert format_for_url(youtube.url) == YOUTUBE_FORMAT
    assert "protocol^=https" in YOUTUBE_FORMAT
    assert format_for_url(vk.url) == WORST_FORMAT
    options = _ydl_options(youtube, Path("videos"))
    assert options["format"] == YOUTUBE_FORMAT
    assert options["extractor_args"]["youtube"]["player_client"] == ["android"]
    assert options["socket_timeout"] == 60
    assert options["logger"] is not None


def test_cookie_essay_becomes_one_line():
    essay = (
        "ERROR: [youtube] abc: Sign in to confirm you're not a bot. "
        "Use --cookies-from-browser or --cookies for the authentication. "
        "See https://github.com/yt-dlp/yt-dlp/wiki/FAQ"
    )
    assert brief_ydl_error(RuntimeError(essay)) == "YouTube закрыл веб-страницу без куков"
    formats = (
        "ERROR: [vk] 1_2: No video formats found!; please report this issue on "
        "https://github.com/yt-dlp/yt-dlp/issues?q= , filling out the appropriate issue template."
    )
    assert brief_ydl_error(RuntimeError(formats)) == "ERROR: [vk] 1_2: No video formats found!"


def test_next_batch_starts_when_last_video_remains():
    class Bar:
        def start_video(self, label, video):
            return

    started = []
    on_start, wrapped = _last_video_hooks(
        2,
        lambda *args: None,
        Bar(),
        lambda: started.append("next"),
    )
    on_start("GPU", "first")
    wrapped("first", None, True, None, "GPU")
    assert started == []
    on_start("GPU", "last")
    assert started == ["next"]
    wrapped("last", None, True, None, "GPU")
    assert started == ["next"]


def test_single_video_batch_starts_next_immediately():
    class Bar:
        def start_video(self, label, video):
            return

    started = []
    on_start, _wrapped = _last_video_hooks(
        1,
        lambda *args: None,
        Bar(),
        lambda: started.append("next"),
    )
    on_start("GPU", "only")
    assert started == ["next"]


def test_fragment_threads_are_passed_to_yt_dlp():
    video = ChannelVideo("1", "Лекция", "https://vk.com/video1")
    options = _ydl_options(video, Path("videos"), fragment_threads=8)
    assert options["concurrent_fragment_downloads"] == 8
    clamped = _ydl_options(video, Path("videos"), fragment_threads=0)
    assert clamped["concurrent_fragment_downloads"] == 1


def test_parallel_timeout_falls_back_to_one_connection(tmp_path, monkeypatch):
    import sys
    import types

    import whisper_fpmi.download as download

    target = tmp_path / "abc_Лекция.mp4"

    class FakeYDL:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=False):
            return {
                "url": "https://rr.googlevideo.com/videoplayback",
                "protocol": "https",
                "ext": "mp4",
                "http_headers": {},
            }

        def prepare_filename(self, info):
            return str(target)

        def add_progress_hook(self, hook):
            return None

        def process_info(self, info):
            target.write_bytes(b"lecture")

    fake_yt_dlp = types.ModuleType("yt_dlp")
    fake_yt_dlp.YoutubeDL = lambda options: FakeYDL()
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_yt_dlp)

    def fail_parallel(*args, **kwargs):
        raise TimeoutError("Read timed out. (read timeout=15)")

    monkeypatch.setattr(download, "download_parallel_http", fail_parallel)
    path = download.download_worst_quality(
        ChannelVideo("abc", "Лекция", "https://www.youtube.com/watch?v=abc"),
        tmp_path,
        fragment_threads=4,
    )
    assert path.read_bytes() == b"lecture"


def test_parse_ignores_other_statuses():
    assert parse_download_hook({"status": "error"}) is None
    assert parse_download_hook({}) is None


def test_progress_bar_tracks_bytes_without_tty():
    bar = DownloadProgress("Лекция про map", disable=True)
    try:
        bar.hook(
            {
                "status": "downloading",
                "downloaded_bytes": 40,
                "total_bytes": 100,
                "info_dict": {"format_id": "18", "vcodec": "avc1", "acodec": "mp4a"},
            }
        )
        assert bar._bar is not None
        assert bar._bar.n == 40
        assert bar._bar.total == 100
        bar.hook(
            {
                "status": "finished",
                "downloaded_bytes": 100,
                "total_bytes": 100,
                "info_dict": {"format_id": "18", "vcodec": "avc1", "acodec": "mp4a"},
            }
        )
        assert bar._bar.n == 100
    finally:
        bar.close()

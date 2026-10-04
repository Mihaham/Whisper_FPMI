from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
from typing import Callable

from loguru import logger as lg

from whisper_fpmi.names import format_timestamp, sanitize_title
from whisper_fpmi.paths import DEFAULT_MODEL, LANGUAGE, RESULT_DIR, TIMED_DIR


def load_whisper_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    *,
    compute_type: str | None = None,
    cpu_threads: int = 0,
    num_workers: int = 1,
):
    from faster_whisper import WhisperModel

    resolved_device, resolved_compute = _device_settings(device, compute_type)
    lg.info("Загружаю Whisper {} / {} / {}", model_name, resolved_device, resolved_compute)
    kwargs: dict[str, str | int] = {
        "device": resolved_device,
        "compute_type": resolved_compute,
        "num_workers": max(1, num_workers),
    }
    if cpu_threads > 0:
        kwargs["cpu_threads"] = cpu_threads
    return WhisperModel(model_name, **kwargs)


def transcribe_file(
    media_path: Path,
    *,
    title: str,
    source_url: str,
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    language: str = LANGUAGE,
    result_dir: Path = RESULT_DIR,
    timed_dir: Path = TIMED_DIR,
    model=None,
    on_progress: Callable[[float, float], None] | None = None,
) -> tuple[Path, Path]:
    whisper = model or load_whisper_model(model_name, device)

    result_dir.mkdir(parents=True, exist_ok=True)
    timed_dir.mkdir(parents=True, exist_ok=True)
    slug = sanitize_title(title)
    plain_path = result_dir / f"{slug}.txt"
    timed_path = timed_dir / f"{slug}.mp4.txt"

    resolved_device, compute_type = _device_settings(device)
    lg.debug(
        "Whisper {} / {} / {} ← {}",
        model_name,
        resolved_device,
        compute_type,
        media_path.name,
    )
    segments_iter, info = whisper.transcribe(
        str(media_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )
    duration = float(info.duration or 0.0)
    lg.debug("Длительность аудио: {:.1f} с, язык: {}", duration, info.language)
    _emit_progress(on_progress, 0.0, duration)

    timed_lines: list[str] = []
    plain_parts: list[str] = []
    last_emit = 0.0
    last_pos = 0.0
    for segment in segments_iter:
        text = segment.text.strip()
        if not text:
            continue
        words = getattr(segment, "words", None) or []
        wrote_word = False
        for word in words:
            token = word.word.strip()
            if not token:
                continue
            w_start = format_timestamp(word.start)
            w_end = format_timestamp(word.end)
            timed_lines.append(f"[{w_start} --> {w_end}]  {token}")
            wrote_word = True
        if not wrote_word:
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            timed_lines.append(f"[{start} --> {end}]  {text}")
        plain_parts.append(text)
        last_pos = float(segment.end)
        now = time.monotonic()
        if now - last_emit >= 0.3:
            _emit_progress(on_progress, last_pos, duration)
            last_emit = now
    _emit_progress(on_progress, duration or last_pos, duration)

    header = _header(title, source_url, model_name, language)
    plain_body = _wrap_plain(plain_parts)
    timed_path.write_text(header + "\n".join(timed_lines) + "\n", encoding="utf-8")
    plain_path.write_text(header + plain_body + "\n", encoding="utf-8")
    lg.debug("Сохранено: {} и {}", plain_path.name, timed_path.name)
    return plain_path, timed_path


def _header(title: str, source_url: str, model_name: str, language: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"# Название: {title}\n"
        f"# Источник: {source_url}\n"
        f"# Модель: whisper {model_name}\n"
        f"# Язык: {language}\n"
        f"# Дата: {stamp}\n"
        "\n"
    )


def _wrap_plain(parts: list[str]) -> str:
    paragraphs: list[str] = []
    bucket: list[str] = []
    for part in parts:
        bucket.append(part)
        if part.endswith((".", "!", "?", "…")) and sum(len(item) for item in bucket) > 180:
            paragraphs.append(" ".join(bucket))
            bucket = []
    if bucket:
        paragraphs.append(" ".join(bucket))
    return "\n".join(paragraphs)


def _emit_progress(
    on_progress: Callable[[float, float], None] | None,
    pos: float,
    duration: float,
) -> None:
    if on_progress is None:
        return
    on_progress(pos, duration)


def has_cuda() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def _device_settings(device: str, compute_type: str | None = None) -> tuple[str, str]:
    if device == "cpu":
        return "cpu", compute_type or "int8"
    if device == "cuda":
        return "cuda", compute_type or "float16"
    if has_cuda():
        return "cuda", compute_type or "float16"
    return "cpu", compute_type or "int8"


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg не найден в PATH. Установите ffmpeg — он нужен и yt-dlp, и Whisper."
        )

from __future__ import annotations

from pathlib import Path
import re
import threading

from loguru import logger as lg

from whisper_fpmi.catalog import write_catalog
from whisper_fpmi.download import download_worst_quality, probe_filesize
from whisper_fpmi.names import fold_key, sanitize_title
from whisper_fpmi.paths import (
    CHANNEL_URL,
    DEFAULT_CPU_CORES,
    DEFAULT_MAX_GB,
    DEFAULT_MODEL,
    RESULT_DIR,
    VIDEO_DIR,
)
from whisper_fpmi.quota import (
    bytes_from_gb,
    can_add_to_batch,
    dir_size,
    format_gib,
    media_files,
)
from whisper_fpmi.progress import (
    RunProgress,
    audio_seconds,
    jobs_audio,
    probe_media_duration,
)
from whisper_fpmi.state import load_state, mark_done, save_state
from whisper_fpmi.transcribe import has_cuda, require_ffmpeg, transcribe_file
from whisper_fpmi.vk import ChannelVideo, list_channel_videos
from whisper_fpmi.workers import WorkerPlan, plan_workers, transcribe_jobs

LEFTOVER_NAME = re.compile(r"^(-?\d+_\d+)_(.+)$")


def run_pipeline(
    *,
    channel_url: str = CHANNEL_URL,
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    limit: int | None = None,
    skip_existing: bool = True,
    keep_video: bool = False,
    cookies: str | None = None,
    refresh_catalog: bool = True,
    max_gb: float = DEFAULT_MAX_GB,
    cpu_cores: int = DEFAULT_CPU_CORES,
    cpu_workers: int | None = None,
    cpu_threads: int | None = None,
    gpu_workers: int | None = None,
) -> int:
    require_ffmpeg()
    max_bytes = bytes_from_gb(max_gb)
    videos = list_channel_videos(channel_url)
    lg.info("На канале {} видео", len(videos))
    if limit is not None:
        videos = videos[:limit]

    state = load_state()
    existing = _existing_keys()
    pending = [
        video
        for video in videos
        if not (skip_existing and _already_done(video, state, existing))
    ]
    skipped = len(videos) - len(pending)
    use_gpu = device != "cpu" and (device == "cuda" or has_cuda())
    plan = plan_workers(
        use_gpu=use_gpu,
        cpu_cores=cpu_cores,
        cpu_workers=cpu_workers,
        cpu_threads=cpu_threads,
        gpu_workers=gpu_workers,
    )
    lg.info(
        "К обработке: {}, пропуск сразу: {}, лимит пачки: {}, воркеры: {}",
        len(pending),
        skipped,
        format_gib(max_bytes) if max_bytes else "без накопления",
        plan.describe(),
    )

    progress = RunProgress(
        total_files=len(pending),
        total_audio=audio_seconds(pending),
    )
    processed = 0
    failed = 0
    cycle = 0
    known = {video.video_id: video for video in videos}

    try:
        while True:
            leftover = _collect_leftovers(known)
            if leftover:
                cycle += 1
                lg.info(
                    "Цикл {}: на диске уже {} ({}), сначала расшифровываю их",
                    cycle,
                    len(leftover),
                    format_gib(dir_size(VIDEO_DIR)),
                )
                done, errors = _transcribe_batch(
                    leftover,
                    model_name=model_name,
                    plan=plan,
                    state=state,
                    existing=existing,
                    keep_video=keep_video,
                    progress=progress,
                    cycle=cycle,
                )
                processed += done
                failed += errors
                pending = [
                    video
                    for video in pending
                    if fold_key(sanitize_title(video.title)) not in existing
                ]
                if keep_video:
                    lg.warning(
                        "Исходники оставлены (--keep-video), "
                        "новый круг скачивания не стартую"
                    )
                    break
                continue

            pending = [
                video
                for video in pending
                if fold_key(sanitize_title(video.title)) not in existing
            ]
            if not pending:
                break

            cycle += 1
            batch, pending = _download_batch(
                pending,
                max_bytes=max_bytes,
                cookies=cookies,
                cycle=cycle,
            )
            if not batch:
                lg.warning("Цикл {}: ничего не скачалось, останавливаюсь", cycle)
                break

            done, errors = _transcribe_batch(
                batch,
                model_name=model_name,
                plan=plan,
                state=state,
                existing=existing,
                keep_video=keep_video,
                progress=progress,
                cycle=cycle,
            )
            processed += done
            failed += errors
            if keep_video and dir_size(VIDEO_DIR) >= max_bytes > 0:
                lg.warning(
                    "Папка videos/ заполнена, а --keep-video не даёт её очистить"
                )
                break

        if refresh_catalog:
            catalog = write_catalog()
            lg.info("Каталог обновлён: {}", catalog)

        lg.info(
            "Готово: новых {}, пропущено {}, ошибок {}",
            processed,
            skipped,
            failed,
        )
        return failed
    finally:
        progress.close()


def transcribe_local(
    media_path: Path,
    *,
    title: str | None = None,
    source_url: str = "",
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
) -> None:
    require_ffmpeg()
    name = title or media_path.stem
    duration = probe_media_duration(media_path)
    video = ChannelVideo(
        video_id=media_path.stem,
        title=name,
        url=source_url or media_path.as_posix(),
        duration=duration,
    )
    progress = RunProgress(total_files=1, total_audio=duration or 0.0)
    try:
        progress.start_batch(1, 1, duration or 0.0)
        progress.start_video("файл", video)
        transcribe_file(
            media_path,
            title=name,
            source_url=video.url,
            model_name=model_name,
            device=device,
            on_progress=lambda pos, dur: progress.update_video("файл", pos, dur),
        )
        progress.finish_video("файл")
    finally:
        progress.close()
    write_catalog()


def _download_batch(
    pending: list[ChannelVideo],
    *,
    max_bytes: int,
    cookies: str | None,
    cycle: int,
) -> tuple[list[tuple[ChannelVideo, Path]], list[ChannelVideo]]:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_partials(VIDEO_DIR)
    batch: list[tuple[ChannelVideo, Path]] = []
    rest = list(pending)
    lg.info("Цикл {}: скачиваю пачку до {}", cycle, format_gib(max_bytes) if max_bytes else "1 файл")

    while rest:
        used = dir_size(VIDEO_DIR)
        video = rest[0]
        extra = probe_filesize(video, cookies=cookies)
        if not can_add_to_batch(used, extra, max_bytes, len(batch)):
            lg.info(
                "Пачка набрана: {} / {}, файлов {}",
                format_gib(used),
                format_gib(max_bytes),
                len(batch),
            )
            break
        try:
            path = download_worst_quality(
                video,
                VIDEO_DIR,
                cookies=cookies,
                quiet=True,
            )
        except Exception as exc:  # noqa: BLE001
            lg.exception("Не скачалось {}: {}", video.video_id, exc)
            rest.pop(0)
            continue
        rest.pop(0)
        batch.append((video, path))
        used = dir_size(VIDEO_DIR)
        lg.info("В videos/: {} после {}", format_gib(used), path.name)
        if max_bytes <= 0 or (batch and used >= max_bytes):
            break

    return batch, rest


def _transcribe_batch(
    batch: list[tuple[ChannelVideo, Path]],
    *,
    model_name: str,
    plan: WorkerPlan,
    state: dict,
    existing: set[str],
    keep_video: bool,
    progress: RunProgress,
    cycle: int,
) -> tuple[int, int]:
    state_lock = threading.Lock()
    progress.start_batch(cycle, len(batch), jobs_audio(batch))
    lg.info("Расшифровываю {} файл(ов)", len(batch))

    def on_result(video, path, ok, _error, label) -> None:
        progress.finish_video(label)
        with state_lock:
            if ok:
                mark_done(
                    state,
                    video.video_id,
                    title=video.title,
                    url=video.url,
                    slug=sanitize_title(video.title),
                    model=model_name,
                )
                save_state(state)
                existing.add(fold_key(video.title))
            if not keep_video:
                path.unlink(missing_ok=True)
                lg.debug("Удалил исходник {} ({})", path.name, label)

    processed, failed = transcribe_jobs(
        batch,
        plan=plan,
        model_name=model_name,
        on_result=on_result,
        on_start=progress.start_video,
        on_progress=progress.update_video,
    )
    lg.info("После расшифровки на диске {}", format_gib(dir_size(VIDEO_DIR)))
    return processed, failed


def _cleanup_partials(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.suffix.lower() in {".part", ".ytdl", ".tmp"}:
            lg.warning("Удаляю незавершённое скачивание {}", item.name)
            item.unlink(missing_ok=True)


def _collect_leftovers(
    known: dict[str, ChannelVideo] | None = None,
) -> list[tuple[ChannelVideo, Path]]:
    items: list[tuple[ChannelVideo, Path]] = []
    for path in media_files(VIDEO_DIR):
        items.append((_video_from_media(path, known), path))
    return items


def _video_from_media(
    path: Path,
    known: dict[str, ChannelVideo] | None = None,
) -> ChannelVideo:
    match = LEFTOVER_NAME.match(path.stem)
    if match:
        video_id, slug = match.group(1), match.group(2)
        video = ChannelVideo(
            video_id=video_id,
            title=slug,
            url=f"https://vk.com/video{video_id}",
        )
    else:
        video = ChannelVideo(video_id=path.stem, title=path.stem, url="")
    meta = (known or {}).get(video.video_id)
    duration = (meta.duration if meta else None) or probe_media_duration(path)
    title = meta.title if meta else video.title
    url = meta.url if meta else video.url
    return ChannelVideo(
        video_id=video.video_id,
        title=title,
        url=url,
        duration=duration,
    )


def _existing_keys() -> set[str]:
    return {fold_key(path.stem) for path in RESULT_DIR.glob("*.txt")}


def _already_done(video: ChannelVideo, state: dict, existing: set[str]) -> bool:
    slug = sanitize_title(video.title)
    if fold_key(slug) in existing:
        return True
    record = state.get("videos", {}).get(video.video_id)
    if not record:
        return False
    stored = RESULT_DIR / f"{record.get('slug', slug)}.txt"
    return stored.exists() and stored.stat().st_size > 0

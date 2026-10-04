from __future__ import annotations

from pathlib import Path
import queue
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from loguru import logger as lg

from whisper_fpmi.catalog import write_catalog
from whisper_fpmi.download import download_worst_quality
from whisper_fpmi.names import fold_key, sanitize_title
from whisper_fpmi.paths import (
    CHANNEL_URL,
    DEFAULT_CPU_CORES,
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_FRAGMENT_THREADS,
    DEFAULT_MAX_GB,
    DEFAULT_MODEL,
    RESULT_DIR,
    RETRANSCRIBE_PATH,
    TIMED_DIR,
    VIDEO_DIR,
    YOUTUBE_URL,
)
from whisper_fpmi.quota import (
    bytes_from_gb,
    dir_size,
    format_gib,
    media_files,
    next_quota_step,
)
from whisper_fpmi.progress import (
    MEDIA_BAR_POSITION,
    RunProgress,
    audio_seconds,
    jobs_audio,
    probe_media_duration,
)
from whisper_fpmi.state import load_state, mark_done, save_state
from whisper_fpmi.transcript import assess_timed, classify_corpus, write_retranscribe_list
from whisper_fpmi.transcribe import has_cuda, require_ffmpeg, transcribe_file
from whisper_fpmi.vk import ChannelVideo, list_channel_videos
from whisper_fpmi.youtube import list_youtube_videos
from whisper_fpmi.workers import (
    CpuLane,
    TranscribeHooks,
    WorkerPlan,
    _path_key,
    plan_workers,
    transcribe_jobs,
)

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
    cpu_queue: int | None = None,
    download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
    fragment_threads: int = DEFAULT_FRAGMENT_THREADS,
    include_youtube: bool = True,
    youtube_url: str = YOUTUBE_URL,
) -> int:
    require_ffmpeg()
    max_bytes = bytes_from_gb(max_gb)
    videos = list_channel_videos(channel_url)
    lg.info("На канале VK {} видео", len(videos))
    if limit is not None:
        videos = videos[:limit]
        include_youtube = False

    state = load_state()
    existing, redo = classify_corpus(RESULT_DIR, TIMED_DIR)
    write_retranscribe_list(redo, RETRANSCRIBE_PATH)
    lg.info(
        "Готовы large-v3 по словам: {}, к перерасшифровке: {}",
        len(existing),
        len(redo),
    )
    vk_pending = [
        video
        for video in videos
        if not (skip_existing and _already_done(video, state, existing))
    ]
    youtube_all: list[ChannelVideo] = []
    youtube_pending: list[ChannelVideo] = []
    if include_youtube:
        youtube_all = list_youtube_videos(youtube_url)
        youtube_pending = [
            video
            for video in youtube_all
            if not (skip_existing and _already_done(video, state, existing))
        ]
    pending = [*vk_pending, *youtube_pending]
    skipped = (len(videos) - len(vk_pending)) + (
        len(youtube_all) - len(youtube_pending)
    )
    lg.info(
        "Сначала VK: {} видео, потом YouTube: {}",
        len(vk_pending),
        len(youtube_pending),
    )
    use_gpu = device != "cpu" and (device == "cuda" or has_cuda())
    plan = plan_workers(
        use_gpu=use_gpu,
        cpu_cores=cpu_cores,
        cpu_workers=cpu_workers,
        cpu_threads=cpu_threads,
        gpu_workers=gpu_workers,
        cpu_queue=cpu_queue,
    )
    gpu_plan = WorkerPlan(
        gpu_workers=plan.gpu_workers,
        cpu_workers=0,
        cpu_threads=0,
        cpu_compute=plan.cpu_compute,
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
    known = {video.video_id: video for video in [*videos, *youtube_all]}
    state_lock = threading.Lock()
    on_result = _result_handler(
        progress,
        state,
        existing,
        keep_video,
        model_name,
        state_lock,
    )
    cpu_lane: CpuLane | None = None
    if plan.independent_cpu:
        cpu_lane = CpuLane(
            plan,
            model_name,
            hooks=_cpu_hooks(on_result, progress),
        )
        cpu_lane.start()
        lg.info(
            "CPU отдельно: {} процесс, {} потоков, по {} видео. GPU его не ждёт",
            plan.cpu_workers,
            plan.cpu_threads,
            plan.cpu_queue,
        )

    try:
        while True:
            owned = cpu_lane.owned_paths() if cpu_lane is not None else set()
            leftover = [
                item
                for item in _collect_leftovers(known)
                if _path_key(item[1]) not in owned
            ]
            if leftover:
                cycle += 1
                lg.info(
                    "Цикл {}: на диске уже {} ({}), сначала расшифровываю их",
                    cycle,
                    len(leftover),
                    format_gib(dir_size(VIDEO_DIR)),
                )
                done, errors = _transcribe_ready(
                    leftover,
                    cpu_lane=cpu_lane,
                    plan=gpu_plan if cpu_lane is not None else plan,
                    model_name=model_name,
                    on_result=on_result,
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
                download_workers=download_workers,
                fragment_threads=fragment_threads,
            )
            if not batch:
                lg.warning("Цикл {}: ничего не скачалось, останавливаюсь", cycle)
                break

            done, errors = _transcribe_ready(
                batch,
                cpu_lane=cpu_lane,
                plan=gpu_plan if cpu_lane is not None else plan,
                model_name=model_name,
                on_result=on_result,
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

        if cpu_lane is not None:
            lg.info("GPU закончил свои видео, жду очередь CPU")
            cpu_lane.close_and_join()
            processed += cpu_lane.ok
            failed += cpu_lane.err
            cpu_lane = None

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
        if cpu_lane is not None:
            cpu_lane.close_and_join()
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
    download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
    fragment_threads: int = DEFAULT_FRAGMENT_THREADS,
) -> tuple[list[tuple[ChannelVideo, Path]], list[ChannelVideo]]:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_partials(VIDEO_DIR)
    workers = max(1, download_workers)
    threads = max(1, fragment_threads)
    batch: list[tuple[ChannelVideo, Path]] = []
    batch_lock = threading.Lock()
    rest = list(pending)
    limit = format_gib(max_bytes) if max_bytes else "1 файл"
    lg.info(
        "Цикл {}: скачиваю пачку до {}, параллельно {}, потоков на файл {}",
        cycle,
        limit,
        workers,
        threads,
    )

    slots: queue.Queue[int] = queue.Queue()
    for offset in range(workers):
        slots.put(MEDIA_BAR_POSITION + offset)

    def one(video: ChannelVideo) -> None:
        position = slots.get()
        try:
            path = download_worst_quality(
                video,
                VIDEO_DIR,
                cookies=cookies,
                bar_position=position,
                fragment_threads=threads,
            )
        except Exception as exc:  # noqa: BLE001
            lg.exception("Не скачалось {}: {}", video.video_id, exc)
            return
        finally:
            slots.put(position)
        with batch_lock:
            batch.append((video, path))
        lg.info("В videos/: {} после {}", format_gib(dir_size(VIDEO_DIR)), path.name)

    reserved = dir_size(VIDEO_DIR)
    inflight: list[Future[None]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while rest:
            scheduled = 0
            while rest:
                step = next_quota_step(
                    used=reserved,
                    extra=0,
                    max_bytes=max_bytes,
                    batch_count=len(batch) + len(inflight),
                )
                if not step.accept:
                    break
                inflight.append(pool.submit(one, rest.pop(0)))
                reserved = step.reserved
                scheduled += 1
                if step.stop:
                    break
            if scheduled == 0:
                break
            lg.info(
                "Качаю {} файл(ов), одновременно {}",
                scheduled,
                min(workers, scheduled),
            )
            for future in inflight:
                future.result()
            inflight.clear()
            reserved = dir_size(VIDEO_DIR)
            if not rest or max_bytes <= 0 or reserved >= max_bytes:
                lg.info(
                    "Пачка набрана: {} / {}, файлов {}",
                    format_gib(reserved),
                    limit,
                    len(batch),
                )
                break

    return batch, rest


def _cpu_hooks(on_result, progress: RunProgress) -> TranscribeHooks:
    return TranscribeHooks(
        on_result=on_result,
        on_start=progress.start_video,
        on_progress=progress.update_video,
    )


def _result_handler(
    progress: RunProgress,
    state: dict,
    existing: set[str],
    keep_video: bool,
    model_name: str,
    state_lock: threading.Lock,
):
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

    return on_result


def _handoff_cpu(
    batch: list[tuple[ChannelVideo, Path]],
    cpu_lane: CpuLane | None,
) -> tuple[list[tuple[ChannelVideo, Path]], list[tuple[ChannelVideo, Path]]]:
    if cpu_lane is None or not batch:
        return [], list(batch)
    taken: list[tuple[ChannelVideo, Path]] = []
    rest: list[tuple[ChannelVideo, Path]] = []
    for job in batch:
        if cpu_lane.submit(job):
            taken.append(job)
        else:
            rest.append(job)
    cpu_lane.seal()
    return taken, rest


def _transcribe_ready(
    batch: list[tuple[ChannelVideo, Path]],
    *,
    cpu_lane: CpuLane | None,
    plan: WorkerPlan,
    model_name: str,
    on_result,
    progress: RunProgress,
    cycle: int,
) -> tuple[int, int]:
    cpu_jobs, gpu_jobs = _handoff_cpu(batch, cpu_lane)
    if cpu_lane is not None:
        cpu_lane.arm_gpu(gpu_jobs)
        lg.info(
            "CPU забирает {} видео, по {} на каждый. GPU — остальные {}. "
            "Следующие {} CPU возьмёт, когда допишет свою пачку",
            len(cpu_jobs),
            cpu_lane.plan.cpu_queue,
            len(gpu_jobs),
            cpu_lane.plan.cpu_queue,
        )
        if not gpu_jobs:
            return 0, 0
        progress.start_batch(cycle, len(gpu_jobs), jobs_audio(gpu_jobs))
        lg.info("Расшифровываю на GPU {} файл(ов)", len(gpu_jobs))
        processed, failed = cpu_lane.drain_gpu(
            model_name,
            TranscribeHooks(
                on_result=on_result,
                on_start=progress.start_video,
                on_progress=progress.update_video,
            ),
        )
        lg.info("GPU освободился, на диске {}", format_gib(dir_size(VIDEO_DIR)))
        return processed, failed
    if not gpu_jobs:
        return 0, 0
    return _transcribe_batch(
        gpu_jobs,
        model_name=model_name,
        plan=plan,
        on_result=on_result,
        progress=progress,
        cycle=cycle,
    )


def _transcribe_batch(
    batch: list[tuple[ChannelVideo, Path]],
    *,
    model_name: str,
    plan: WorkerPlan,
    on_result,
    progress: RunProgress,
    cycle: int,
) -> tuple[int, int]:
    progress.start_batch(cycle, len(batch), jobs_audio(batch))
    where = "GPU" if plan.gpu_workers else "CPU"
    lg.info("Расшифровываю на {} {} файл(ов)", where, len(batch))
    processed, failed = transcribe_jobs(
        batch,
        plan=plan,
        model_name=model_name,
        on_result=on_result,
        on_start=progress.start_video,
        on_progress=progress.update_video,
    )
    lg.info("{} освободился, на диске {}", where, format_gib(dir_size(VIDEO_DIR)))
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
        found = _known_by_filename(path.stem, known or {})
        if found is not None:
            video = found
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


def _known_by_filename(stem: str, known: dict[str, ChannelVideo]) -> ChannelVideo | None:
    for video_id, video in sorted(known.items(), key=lambda item: len(item[0]), reverse=True):
        if stem == video_id or stem.startswith(f"{video_id}_"):
            return video
    return None


def _already_done(video: ChannelVideo, state: dict, existing: set[str]) -> bool:
    slug = sanitize_title(video.title)
    if fold_key(slug) in existing:
        return True
    record = state.get("videos", {}).get(video.video_id)
    if not record:
        return False
    stored_slug = str(record.get("slug") or slug)
    return assess_timed(TIMED_DIR / f"{stored_slug}.mp4.txt") == "final"

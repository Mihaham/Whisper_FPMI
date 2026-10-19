from __future__ import annotations

from pathlib import Path
import queue
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from loguru import logger as lg

from whisper_fpmi.catalog import write_catalog
from whisper_fpmi.dataset import read_jsonl
from whisper_fpmi.download import brief_ydl_error, download_worst_quality
from whisper_fpmi.names import fold_key, sanitize_title
from whisper_fpmi.paths import (
    CHANNEL_URL,
    DEFAULT_CPU_CORES,
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_FRAGMENT_THREADS,
    DEFAULT_MAX_GB,
    DEFAULT_MODEL,
    LECTURES_PATH,
    RESULT_DIR,
    RETRANSCRIBE_PATH,
    TIMED_DIR,
    VIDEO_DIR,
    YOUTUBE_URL,
)
from whisper_fpmi.quota import (
    ASSUMED_FILE_BYTES,
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
from whisper_fpmi.lecture_id import (
    build_catalog,
    load_description_book,
    prefetch_descriptions,
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
    transcribe_as_available,
    transcribe_jobs,
)

LEFTOVER_NAME = re.compile(r"^(-?\d+_\d+)_(.+)$")


class _ReadyInbox:
    """Файлы, которые уже лежат на диске, пока остальная пачка ещё качается."""

    def __init__(self) -> None:
        self._items: list[tuple[ChannelVideo, Path]] = []
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def add(self, video: ChannelVideo, path: Path) -> None:
        with self._lock:
            self._items.append((video, path))
            self._ready.set()

    def pop(self) -> list[tuple[ChannelVideo, Path]]:
        with self._lock:
            if not self._items:
                return []
            items = self._items
            self._items = []
            self._ready.clear()
            return items

    def wait(self, timeout: float) -> None:
        self._ready.wait(timeout)


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
    existing, redo, slug_by_fold = classify_corpus(RESULT_DIR, TIMED_DIR)
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
    book = load_description_book()
    catalog = build_catalog(slug_by_fold, state, book, read_jsonl(LECTURES_PATH))
    state_lock = threading.Lock()
    queued = len(pending)
    pending = _release_same_lectures(
        pending,
        catalog=catalog,
        book=book,
        state=state,
        existing=existing,
        state_lock=state_lock,
        model_name=model_name,
    )
    skipped += queued - len(pending)
    lg.info(
        "По дате и лектору уже есть расшифровка: {}. Звук не сравниваю",
        queued - len(pending),
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
    on_result = _result_handler(
        progress,
        state,
        existing,
        keep_video,
        model_name,
        state_lock,
        book,
        catalog,
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

    description_stop = threading.Event()
    description_thread = threading.Thread(
        target=prefetch_descriptions,
        args=(pending, book, description_stop),
        name="lecture-descriptions",
        daemon=True,
    )
    prefetch_thread: threading.Thread | None = None
    prefetch_result: dict[str, list[ChannelVideo]] = {}
    pending_box = {"items": pending}
    inbox = _ReadyInbox()
    bar_cycle = 0
    download_failures: dict[str, int] = {}

    def start_next_download() -> None:
        nonlocal prefetch_thread, cycle
        if prefetch_thread is not None and prefetch_thread.is_alive():
            return
        if keep_video and dir_size(VIDEO_DIR) >= max_bytes > 0:
            lg.warning(
                "Папка videos/ заполнена, а --keep-video не даёт её очистить"
            )
            pending_box["items"] = []
            return
        pending_box["items"] = _still_pending(
            pending_box["items"],
            catalog=catalog,
            book=book,
            state=state,
            existing=existing,
            state_lock=state_lock,
            model_name=model_name,
            progress=progress,
        )
        if not pending_box["items"]:
            return
        snapshot = pending_box["items"]
        pending_box["items"] = []
        cycle += 1
        next_cycle = cycle

        def run_with_rest() -> None:
            lg.info("На GPU остался последний ролик, качаю следующую пачку")
            try:
                _ignored, rest = _download_batch(
                    snapshot,
                    max_bytes=max_bytes,
                    cookies=cookies,
                    cycle=next_cycle,
                    download_workers=download_workers,
                    fragment_threads=fragment_threads,
                    on_ready=inbox.add,
                    fail_counts=download_failures,
                )
                prefetch_result["rest"] = rest
                del _ignored
            except Exception as exc:  # noqa: BLE001
                lg.warning("Следующая пачка не скачалась: {}", brief_ydl_error(exc))
                prefetch_result["rest"] = snapshot

        prefetch_thread = threading.Thread(
            target=run_with_rest, name="next-batch", daemon=True
        )
        prefetch_thread.start()

    def admit(ready: list[tuple[ChannelVideo, Path]]) -> None:
        nonlocal bar_cycle
        if cycle != bar_cycle:
            bar_cycle = cycle
            progress.start_batch(bar_cycle, len(ready), jobs_audio(ready))
        else:
            progress.grow_batch(len(ready), jobs_audio(ready))
        lg.info(
            "Из новой пачки готово {} ролик(ов), GPU берёт их, не дожидаясь остальных",
            len(ready),
        )

    def poll_more() -> list[tuple[ChannelVideo, Path]] | None:
        nonlocal prefetch_thread
        ready = inbox.pop()
        if ready:
            admit(ready)
            return ready
        thread = prefetch_thread
        if thread is not None and thread.is_alive():
            return []
        if thread is not None:
            thread.join()
            pending_box["items"] = prefetch_result.get("rest") or []
            prefetch_thread = None
            ready = inbox.pop()
            if ready:
                admit(ready)
                return ready
        return None

    def transcribe_follow(
        batch: list[tuple[ChannelVideo, Path]],
        follow_cycle: int,
    ) -> tuple[int, int]:
        nonlocal bar_cycle
        bar_cycle = follow_cycle
        progress.start_batch(follow_cycle, len(batch), jobs_audio(batch))
        where = "GPU" if plan.gpu_workers else "CPU"
        lg.info("Расшифровываю на {} {} файл(ов)", where, len(batch))
        return transcribe_as_available(
            batch,
            plan=plan,
            model_name=model_name,
            on_result=on_result,
            on_start=progress.start_video,
            on_progress=progress.update_video,
            pull_more=poll_more,
            on_last=start_next_download,
        )

    try:
        description_thread.start()
        while True:
            owned = cpu_lane.owned_paths() if cpu_lane is not None else set()
            if prefetch_thread is not None:
                leftover = []
            else:
                leftover = [
                    item
                    for item in _collect_leftovers(known)
                    if _path_key(item[1]) not in owned
                ]
            if leftover:
                fresh_ids = {
                    video.video_id
                    for video in _release_same_lectures(
                        [video for video, _path in leftover],
                        catalog=catalog,
                        book=book,
                        state=state,
                        existing=existing,
                        state_lock=state_lock,
                        model_name=model_name,
                        progress=progress,
                    )
                }
                kept_files = []
                for video, path in leftover:
                    if video.video_id in fresh_ids:
                        kept_files.append((video, path))
                    elif not keep_video:
                        path.unlink(missing_ok=True)
                leftover = kept_files
            if leftover:
                cycle += 1
                lg.info(
                    "Цикл {}: на диске уже {} ({}), сначала расшифровываю их",
                    cycle,
                    len(leftover),
                    format_gib(dir_size(VIDEO_DIR)),
                )
                if cpu_lane is None:
                    done, errors = transcribe_follow(leftover, cycle)
                    processed += done
                    failed += errors
                    break
                done, errors = _transcribe_ready(
                    leftover,
                    cpu_lane=cpu_lane,
                    plan=gpu_plan,
                    model_name=model_name,
                    on_result=on_result,
                    progress=progress,
                    cycle=cycle,
                    on_last=start_next_download,
                )
                processed += done
                failed += errors
                if done == 0:
                    lg.warning(
                        "Файлы на диске не расшифровались, останавливаюсь"
                    )
                    break
                pending_box["items"] = _still_pending(
                    pending_box["items"],
                    catalog=catalog,
                    book=book,
                    state=state,
                    existing=existing,
                    state_lock=state_lock,
                    model_name=model_name,
                    progress=progress,
                )
                if keep_video:
                    lg.warning(
                        "Исходники оставлены (--keep-video), "
                        "новый круг скачивания не стартую"
                    )
                    break
                continue

            if prefetch_thread is not None:
                while True:
                    extra = poll_more()
                    if extra:
                        cycle += 1
                        done, errors = _transcribe_ready(
                            extra,
                            cpu_lane=cpu_lane,
                            plan=gpu_plan if cpu_lane is not None else plan,
                            model_name=model_name,
                            on_result=on_result,
                            progress=progress,
                            cycle=cycle,
                            on_last=start_next_download,
                        )
                        processed += done
                        failed += errors
                        continue
                    if extra is None:
                        break
                    inbox.wait(0.5)
                continue
            else:
                pending_box["items"] = _still_pending(
                    pending_box["items"],
                    catalog=catalog,
                    book=book,
                    state=state,
                    existing=existing,
                    state_lock=state_lock,
                    model_name=model_name,
                    progress=progress,
                )
                if not pending_box["items"]:
                    break
                cycle += 1
                batch, pending_box["items"] = _download_batch(
                    pending_box["items"],
                    max_bytes=max_bytes,
                    cookies=cookies,
                    cycle=cycle,
                    download_workers=download_workers,
                    fragment_threads=fragment_threads,
                    fail_counts=download_failures,
                )
            if not batch:
                lg.warning("Цикл {}: ничего не скачалось, останавливаюсь", cycle)
                break

            if cpu_lane is None:
                done, errors = transcribe_follow(batch, cycle)
                processed += done
                failed += errors
                break
            done, errors = _transcribe_ready(
                batch,
                cpu_lane=cpu_lane,
                plan=gpu_plan,
                model_name=model_name,
                on_result=on_result,
                progress=progress,
                cycle=cycle,
                on_last=start_next_download,
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
            catalog_path = write_catalog()
            lg.info("Каталог обновлён: {}", catalog_path)

        lg.info(
            "Готово: новых {}, пропущено {}, ошибок {}",
            processed,
            skipped,
            failed,
        )
        return failed
    finally:
        description_stop.set()
        description_thread.join(timeout=3)
        if prefetch_thread is not None:
            prefetch_thread.join()
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
    on_ready=None,
    fail_counts: dict[str, int] | None = None,
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
    counts = fail_counts if fail_counts is not None else {}
    retry: list[ChannelVideo] = []

    def one(video: ChannelVideo) -> None:
        nonlocal committed
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
            message = brief_ydl_error(exc)
            lg.warning("Не скачалось {}: {}", video.video_id, message)
            if "no video formats" in message.casefold():
                return
            with batch_lock:
                attempt = counts.get(video.video_id, 0) + 1
                counts[video.video_id] = attempt
                if attempt < 3:
                    retry.append(video)
                    again = True
                else:
                    again = False
            if again:
                lg.info("Верну {} в очередь, попытка {}", video.video_id, attempt)
            else:
                lg.warning("Пропускаю {} после {} неудач", video.video_id, attempt)
            return
        finally:
            slots.put(position)
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:
            size = 0
        if size <= 0:
            size = ASSUMED_FILE_BYTES
        with batch_lock:
            batch.append((video, path))
            committed += size
        if on_ready is not None:
            on_ready(video, path)
        lg.info("В videos/: {} после {}", format_gib(dir_size(VIDEO_DIR)), path.name)

    # Место, которое освобождает GPU, удаляя уже расшифрованное, в эту пачку не добавляется.
    baseline = dir_size(VIDEO_DIR)
    committed = 0
    inflight: list[Future[None]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while rest:
            reserved = baseline + committed
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
            if not rest or max_bytes <= 0 or baseline + committed >= max_bytes:
                lg.info(
                    "Пачка набрана: {} / {}, файлов {}, дальше следующая пачка",
                    format_gib(baseline + committed),
                    limit,
                    len(batch),
                )
                break

    return batch, retry + rest


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
    book,
    catalog,
):
    def on_result(video, path, ok, _error, label) -> None:
        progress.finish_video(label)
        with state_lock:
            if ok:
                slug = sanitize_title(video.title)
                mark_done(
                    state,
                    video.video_id,
                    title=video.title,
                    url=video.url,
                    slug=slug,
                    model=model_name,
                )
                save_state(state)
                existing.add(fold_key(video.title))
                existing.add(fold_key(sanitize_title(video.title)))
                existing.add(fold_key(slug))
                catalog.remember(video, slug, book.get(video.video_id) or "")
            if not keep_video:
                path.unlink(missing_ok=True)
                lg.debug("Удалил исходник {} ({})", path.name, label)

    return on_result


def _still_pending(
    videos: list[ChannelVideo],
    *,
    catalog,
    book,
    state: dict,
    existing: set[str],
    state_lock: threading.Lock,
    model_name: str,
    progress: RunProgress,
) -> list[ChannelVideo]:
    titled = [
        video
        for video in videos
        if not _already_done(video, state, existing)
    ]
    return _release_same_lectures(
        titled,
        catalog=catalog,
        book=book,
        state=state,
        existing=existing,
        state_lock=state_lock,
        model_name=model_name,
        progress=progress,
    )


def _release_same_lectures(
    videos: list[ChannelVideo],
    *,
    catalog,
    book,
    state: dict,
    existing: set[str],
    state_lock: threading.Lock,
    model_name: str,
    progress: RunProgress | None = None,
) -> list[ChannelVideo]:
    kept: list[ChannelVideo] = []
    linked: list[tuple[ChannelVideo, object]] = []
    with state_lock:
        for video in videos:
            record = (state.get("videos") or {}).get(video.video_id) or {}
            if record.get("same_lecture_as"):
                continue
            hit = catalog.match_video(video, book)
            if hit is None:
                kept.append(video)
                continue
            mark_done(
                state,
                video.video_id,
                title=video.title,
                url=video.url,
                slug=hit.slug,
                model=model_name,
                same_lecture_as=hit.video_id or hit.slug,
            )
            existing.add(fold_key(video.title))
            existing.add(fold_key(sanitize_title(video.title)))
            existing.add(fold_key(hit.slug))
            linked.append((video, hit))
        if linked:
            save_state(state)
    for video, hit in linked:
        lg.info(
            "Та же лекция ({}, {}). Whisper пропускаю, расшифровка: {}",
            hit.event_date,
            hit.lecturer,
            hit.plain,
        )
        if progress is not None:
            progress.note_skipped(video)
    return kept


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


def _last_video_hooks(total: int, on_result, progress: RunProgress, on_last):
    """Когда на расшифровке остаётся один ролик, можно качать следующую пачку."""
    started = 0
    done = 0
    fired = False
    lock = threading.Lock()

    def claim() -> bool:
        nonlocal fired
        inflight = started - done
        if fired or on_last is None or total <= 0:
            return False
        if inflight == 1 and started == total:
            fired = True
            return True
        return False

    def on_start(label: str, video: ChannelVideo) -> None:
        nonlocal started
        progress.start_video(label, video)
        with lock:
            started += 1
            launch = claim()
        if launch:
            on_last()

    def wrapped(video, path, ok, error, label) -> None:
        nonlocal done
        on_result(video, path, ok, error, label)
        with lock:
            done += 1
            launch = claim()
        if launch:
            on_last()

    return on_start, wrapped


def _transcribe_ready(
    batch: list[tuple[ChannelVideo, Path]],
    *,
    cpu_lane: CpuLane | None,
    plan: WorkerPlan,
    model_name: str,
    on_result,
    progress: RunProgress,
    cycle: int,
    on_last=None,
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
        on_start, wrapped = _last_video_hooks(len(gpu_jobs), on_result, progress, on_last)
        processed, failed = cpu_lane.drain_gpu(
            model_name,
            TranscribeHooks(
                on_result=wrapped,
                on_start=on_start,
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
        on_last=on_last,
    )


def _transcribe_batch(
    batch: list[tuple[ChannelVideo, Path]],
    *,
    model_name: str,
    plan: WorkerPlan,
    on_result,
    progress: RunProgress,
    cycle: int,
    on_last=None,
) -> tuple[int, int]:
    progress.start_batch(cycle, len(batch), jobs_audio(batch))
    where = "GPU" if plan.gpu_workers else "CPU"
    lg.info("Расшифровываю на {} {} файл(ов)", where, len(batch))
    on_start, wrapped = _last_video_hooks(len(batch), on_result, progress, on_last)
    processed, failed = transcribe_jobs(
        batch,
        plan=plan,
        model_name=model_name,
        on_result=wrapped,
        on_start=on_start,
        on_progress=progress.update_video,
    )
    lg.info("{} освободился, на диске {}", where, format_gib(dir_size(VIDEO_DIR)))
    return processed, failed


def _cleanup_partials(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        name = item.name.casefold()
        if ".part" in name or name.endswith(".ytdl") or name.endswith(".tmp"):
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


def _disk_video_ids(known: dict[str, ChannelVideo]) -> set[str]:
    found: set[str] = set()
    for path in media_files(VIDEO_DIR):
        match = LEFTOVER_NAME.match(path.stem)
        if match:
            found.add(match.group(1))
            continue
        video = _known_by_filename(path.stem, known)
        if video is not None:
            found.add(video.video_id)
    return found


def _known_by_filename(stem: str, known: dict[str, ChannelVideo]) -> ChannelVideo | None:
    for video_id, video in sorted(known.items(), key=lambda item: len(item[0]), reverse=True):
        if stem == video_id or stem.startswith(f"{video_id}_"):
            return video
    return None


def _already_done(video: ChannelVideo, state: dict, existing: set[str]) -> bool:
    from whisper_fpmi.lecture_id import linked_slug

    linked = linked_slug(video.video_id)
    if linked and assess_timed(TIMED_DIR / f"{linked}.mp4.txt") == "final":
        return True
    slug = sanitize_title(video.title)
    if fold_key(slug) in existing:
        return True
    record = state.get("videos", {}).get(video.video_id)
    if not record:
        return False
    stored_slug = str(record.get("slug") or slug)
    return assess_timed(TIMED_DIR / f"{stored_slug}.mp4.txt") == "final"

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Callable

from loguru import logger as lg

from whisper_fpmi.paths import (
    DEFAULT_CPU_CORES,
    DEFAULT_MAX_CPU_WORKERS,
    DEFAULT_MODEL,
)
from whisper_fpmi.transcribe import load_whisper_model, transcribe_file
from whisper_fpmi.vk import ChannelVideo

MediaJob = tuple[ChannelVideo, Path]
OnResult = Callable[[ChannelVideo, Path, bool, str | None, str], None]
OnStart = Callable[[str, ChannelVideo], None]
OnProgress = Callable[[str, float, float], None]


@dataclass(frozen=True)
class TranscribeHooks:
    on_result: OnResult | None = None
    on_start: OnStart | None = None
    on_progress: OnProgress | None = None


@dataclass(frozen=True)
class WorkerPlan:
    gpu_workers: int
    cpu_workers: int
    cpu_threads: int
    cpu_compute: str = "int8"

    @property
    def total_cpu_threads(self) -> int:
        return self.cpu_workers * self.cpu_threads

    def describe(self) -> str:
        parts: list[str] = []
        if self.gpu_workers:
            parts.append(f"{self.gpu_workers}×GPU FP16")
        if self.cpu_workers:
            parts.append(
                f"{self.cpu_workers}×CPU {self.cpu_compute} "
                f"({self.cpu_threads} потоков на воркер, "
                f"всего {self.total_cpu_threads})"
            )
        return " + ".join(parts) if parts else "нет воркеров"


def plan_workers(
    *,
    use_gpu: bool,
    cpu_cores: int = DEFAULT_CPU_CORES,
    cpu_workers: int | None = None,
    cpu_threads: int | None = None,
    gpu_workers: int | None = None,
    cpu_compute: str = "int8",
) -> WorkerPlan:
    if cpu_cores < 1:
        raise ValueError("cpu_cores должен быть >= 1")

    gpu = 0
    if use_gpu:
        gpu = 1 if gpu_workers is None else max(0, gpu_workers)

    workers, threads = _cpu_split(cpu_cores, cpu_workers, cpu_threads)
    if gpu == 0 and workers == 0:
        workers, threads = _cpu_split(cpu_cores, None, None)

    return WorkerPlan(
        gpu_workers=gpu,
        cpu_workers=workers,
        cpu_threads=threads,
        cpu_compute=cpu_compute,
    )


def _cpu_split(
    cpu_cores: int,
    cpu_workers: int | None,
    cpu_threads: int | None,
) -> tuple[int, int]:
    if cpu_workers == 0:
        return 0, 0
    if cpu_workers is None:
        workers = max(1, min(DEFAULT_MAX_CPU_WORKERS, cpu_cores // 5))
    else:
        workers = max(0, cpu_workers)
    if workers == 0:
        return 0, 0
    if cpu_threads is None:
        return workers, max(1, cpu_cores // workers)
    return workers, max(1, cpu_threads)


class JobQueue:
    """Общая очередь: CPU берёт лишние файлы, последние оставляем GPU."""

    def __init__(self, jobs: list[MediaJob], gpu_workers: int) -> None:
        self._jobs = list(jobs)
        self._idx = 0
        self._gpu_workers = max(0, gpu_workers)
        self._gpu_alive = self._gpu_workers > 0
        self._cond = threading.Condition()

    def close_gpu(self) -> None:
        with self._cond:
            self._gpu_alive = False
            self._cond.notify_all()

    def take(self, is_cpu: bool, *, wait: bool = True) -> MediaJob | None:
        with self._cond:
            while True:
                left = len(self._jobs) - self._idx
                if left <= 0:
                    return None
                reserved = is_cpu and self._gpu_alive
                if reserved and left <= self._gpu_workers:
                    if not wait:
                        return None
                    self._cond.wait(timeout=0.5)
                    continue
                job = self._jobs[self._idx]
                self._idx += 1
                return job


class _CpuModelCache:
    def __init__(self, plan: WorkerPlan, model_name: str) -> None:
        self._plan = plan
        self._model_name = model_name
        self._lock = threading.Lock()
        self._model = None
        self._error: BaseException | None = None

    def get(self):
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._model is not None:
                return self._model
            try:
                self._model = load_whisper_model(
                    self._model_name,
                    "cpu",
                    compute_type=self._plan.cpu_compute,
                    cpu_threads=self._plan.cpu_threads,
                    num_workers=self._plan.cpu_workers,
                )
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
                raise
            return self._model


def transcribe_jobs(
    batch: list[MediaJob],
    *,
    plan: WorkerPlan,
    model_name: str = DEFAULT_MODEL,
    on_result: OnResult | None = None,
    on_start: OnStart | None = None,
    on_progress: OnProgress | None = None,
) -> tuple[int, int]:
    if not batch:
        return 0, 0
    if plan.gpu_workers == 0 and plan.cpu_workers == 0:
        raise ValueError("Нужен хотя бы один GPU- или CPU-воркер")

    hooks = TranscribeHooks(
        on_result=on_result,
        on_start=on_start,
        on_progress=on_progress,
    )
    queue = JobQueue(batch, plan.gpu_workers)
    tally = {"ok": 0, "err": 0}
    tally_lock = threading.Lock()
    cpu_models = _CpuModelCache(plan, model_name)
    lg.info("Параллельная расшифровка: {} файл(ов), {}", len(batch), plan.describe())
    threads = _spawn_workers(
        plan,
        queue,
        cpu_models,
        model_name,
        hooks,
        tally,
        tally_lock,
    )
    for thread in threads:
        thread.join()
    return tally["ok"], tally["err"]


def _spawn_workers(
    plan: WorkerPlan,
    queue: JobQueue,
    cpu_models: _CpuModelCache,
    model_name: str,
    hooks: TranscribeHooks,
    tally: dict[str, int],
    tally_lock: threading.Lock,
) -> list[threading.Thread]:
    def record(video, path, ok, error, label) -> None:
        _record(tally, tally_lock, hooks.on_result, video, path, ok, error, label)

    threads: list[threading.Thread] = []
    if plan.gpu_workers:
        threads.append(
            threading.Thread(
                target=_gpu_loop,
                args=(queue, model_name, record, hooks),
                name="whisper-gpu",
                daemon=True,
            )
        )
    for index in range(plan.cpu_workers):
        threads.append(
            threading.Thread(
                target=_cpu_loop,
                args=(index + 1, queue, cpu_models, model_name, record, hooks),
                name=f"whisper-cpu-{index + 1}",
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()
    return threads


def _record(
    tally: dict[str, int],
    lock: threading.Lock,
    on_result: OnResult | None,
    video: ChannelVideo,
    path: Path,
    ok: bool,
    error: str | None,
    label: str,
) -> None:
    with lock:
        tally["ok" if ok else "err"] += 1
    if on_result is not None:
        on_result(video, path, ok, error, label)


def _run_one(
    job: MediaJob,
    model,
    model_name: str,
    label: str,
    record: OnResult,
    hooks: TranscribeHooks,
) -> None:
    video, path = job
    lg.debug("{} ← {}", label, path.name)
    if hooks.on_start is not None:
        hooks.on_start(label, video)
    device = "cuda" if label.startswith("GPU") else "cpu"

    def on_progress(pos: float, duration: float) -> None:
        if hooks.on_progress is not None:
            hooks.on_progress(label, pos, duration)

    try:
        transcribe_file(
            path,
            title=video.title,
            source_url=video.url,
            model_name=model_name,
            device=device,
            model=model,
            on_progress=on_progress,
        )
    except Exception as exc:  # noqa: BLE001
        lg.exception("Ошибка расшифровки {} на {}: {}", video.video_id, label, exc)
        record(video, path, False, str(exc), label)
        return
    record(video, path, True, None, label)


def _gpu_loop(
    queue: JobQueue,
    model_name: str,
    record: OnResult,
    hooks: TranscribeHooks,
) -> None:
    try:
        try:
            model = load_whisper_model(model_name, "cuda")
        except Exception as exc:  # noqa: BLE001
            lg.exception("GPU Whisper не загрузился, очередь заберёт CPU: {}", exc)
            return
        while True:
            job = queue.take(is_cpu=False)
            if job is None:
                return
            _run_one(job, model, model_name, "GPU", record, hooks)
    finally:
        queue.close_gpu()


def _cpu_loop(
    index: int,
    queue: JobQueue,
    cpu_models: _CpuModelCache,
    model_name: str,
    record: OnResult,
    hooks: TranscribeHooks,
) -> None:
    model = cpu_models.get()
    label = f"CPU-{index}"
    while True:
        job = queue.take(is_cpu=True)
        if job is None:
            return
        _run_one(job, model, model_name, label, record, hooks)

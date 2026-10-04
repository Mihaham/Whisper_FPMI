from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import queue
import threading
from typing import Callable

from loguru import logger as lg

from whisper_fpmi.paths import (
    DEFAULT_CPU_CORES,
    DEFAULT_CPU_LANE_QUEUE,
    DEFAULT_CPU_LANE_THREADS,
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
    cpu_queue: int = DEFAULT_CPU_LANE_QUEUE
    independent_cpu: bool = False

    @property
    def total_cpu_threads(self) -> int:
        return self.cpu_workers * self.cpu_threads

    def describe(self) -> str:
        parts: list[str] = []
        if self.gpu_workers:
            parts.append(f"{self.gpu_workers}×GPU FP16")
        if self.cpu_workers:
            if self.independent_cpu:
                parts.append(
                    f"{self.cpu_workers}×CPU {self.cpu_compute} "
                    f"({self.cpu_threads} потоков, по {self.cpu_queue} видео, GPU не ждёт)"
                )
            else:
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
    cpu_queue: int | None = None,
    cpu_compute: str = "int8",
) -> WorkerPlan:
    if cpu_cores < 1:
        raise ValueError("cpu_cores должен быть >= 1")

    gpu = 0
    if use_gpu:
        gpu = 1 if gpu_workers is None else max(0, gpu_workers)

    if cpu_workers is None and gpu > 0:
        workers = 1
        threads = cpu_threads if cpu_threads else DEFAULT_CPU_LANE_THREADS
    else:
        workers, threads = _cpu_split(cpu_cores, cpu_workers, cpu_threads)
    if gpu == 0 and workers == 0:
        workers, threads = _cpu_split(cpu_cores, None, None)

    queue_size = DEFAULT_CPU_LANE_QUEUE if cpu_queue is None else max(1, cpu_queue)
    return WorkerPlan(
        gpu_workers=gpu,
        cpu_workers=workers,
        cpu_threads=threads,
        cpu_compute=cpu_compute,
        cpu_queue=queue_size,
        independent_cpu=gpu > 0 and workers > 0,
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


def _path_key(path: Path) -> str:
    return str(path.resolve()).lower()


_BATCH_END = object()


@dataclass
class _CpuSlot:
    """Одна пачка видео у одного CPU. Следующая пачка — только после этой."""

    queue: queue.Queue = field(default_factory=queue.Queue)
    held: int = 0
    accepting: bool = True
    paths: set[str] = field(default_factory=set)
    dead: bool = False


class CpuLane:
    """Каждый CPU держит свою пачку. Остальное лежит у GPU, пока CPU не допишет."""

    def __init__(
        self,
        plan: WorkerPlan,
        model_name: str,
        hooks: TranscribeHooks,
    ) -> None:
        self.plan = plan
        self._model_name = model_name
        self._hooks = hooks
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._steal: deque[MediaJob] = deque()
        self._closed = False
        self.ok = 0
        self.err = 0
        workers = max(1, plan.cpu_workers)
        self._slots = [_CpuSlot() for _ in range(workers)]
        self._threads = [
            threading.Thread(
                target=self._loop,
                args=(index,),
                name=f"whisper-cpu-{index + 1}",
                daemon=True,
            )
            for index in range(workers)
        ]

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def free_slots(self) -> int:
        with self._lock:
            return sum(
                max(0, self.plan.cpu_queue - slot.held)
                for slot in self._slots
                if slot.accepting and not slot.dead
            )

    def owned_paths(self) -> set[str]:
        with self._lock:
            owned: set[str] = set()
            for slot in self._slots:
                owned.update(slot.paths)
            return owned

    def submit(self, job: MediaJob) -> bool:
        _video, path = job
        key = _path_key(path)
        with self._lock:
            slot = next(
                (
                    item
                    for item in self._slots
                    if item.accepting and not item.dead and item.held < self.plan.cpu_queue
                ),
                None,
            )
            if slot is None:
                return False
            slot.held += 1
            slot.paths.add(key)
            slot.queue.put(job)
            if slot.held >= self.plan.cpu_queue:
                slot.accepting = False
                slot.queue.put(_BATCH_END)
            return True

    def seal(self) -> None:
        """Закрыть неполную пачку, чтобы после неё CPU взял следующие видео у GPU."""
        with self._lock:
            for slot in self._slots:
                if slot.accepting and slot.held > 0 and not slot.dead:
                    slot.accepting = False
                    slot.queue.put(_BATCH_END)

    def arm_gpu(self, jobs: list[MediaJob]) -> None:
        with self._lock:
            self._steal.extend(jobs)

    def take_gpu(self) -> MediaJob | None:
        with self._lock:
            if not self._steal:
                return None
            return self._steal.popleft()

    def gpu_pending(self) -> int:
        with self._lock:
            return len(self._steal)

    def drain_gpu(self, model_name: str, hooks: TranscribeHooks) -> tuple[int, int]:
        with self._lock:
            if not self._steal:
                return 0, 0
        tally = {"ok": 0, "err": 0}
        tally_lock = threading.Lock()

        def record(video, path, ok, error, label) -> None:
            _record(tally, tally_lock, hooks.on_result, video, path, ok, error, label)

        try:
            model = load_whisper_model(model_name, "cuda")
        except Exception as exc:  # noqa: BLE001
            lg.exception("GPU Whisper не загрузился, видео останутся на диске: {}", exc)
            with self._lock:
                self._steal.clear()
            return 0, 0
        while True:
            job = self.take_gpu()
            if job is None:
                break
            _run_one(job, model, model_name, "GPU", record, hooks)
        return tally["ok"], tally["err"]

    def close_and_join(self) -> None:
        with self._lock:
            self._closed = True
        for slot in self._slots:
            slot.queue.put(None)
        for thread in self._threads:
            if thread.is_alive():
                thread.join()

    def _loop(self, index: int) -> None:
        slot = self._slots[index]
        label = f"CPU-{index + 1}"
        try:
            with self._load_lock:
                model = load_whisper_model(
                    self._model_name,
                    "cpu",
                    compute_type=self.plan.cpu_compute,
                    cpu_threads=self.plan.cpu_threads,
                    num_workers=1,
                )
        except Exception as exc:  # noqa: BLE001
            lg.exception("CPU-{} Whisper не загрузился: {}", index + 1, exc)
            self._drop_slot(slot)
            return
        while True:
            job = slot.queue.get()
            if job is None:
                return
            if job is _BATCH_END:
                self._refill(slot, index + 1)
                continue
            self._run(job, model, label)

    def _refill(self, slot: _CpuSlot, number: int) -> None:
        with self._lock:
            slot.held = 0
            if self._closed or slot.dead:
                slot.accepting = False
                return
            stolen: list[MediaJob] = []
            while self._steal and len(stolen) < self.plan.cpu_queue:
                job = self._steal.popleft()
                stolen.append(job)
                slot.paths.add(_path_key(job[1]))
            if not stolen:
                slot.accepting = True
                return
            slot.held = len(stolen)
            slot.accepting = False
            for job in stolen:
                slot.queue.put(job)
            slot.queue.put(_BATCH_END)
        lg.info(
            "CPU-{} дописал пачку, забирает ещё {} видео у GPU",
            number,
            len(stolen),
        )

    def _drop_slot(self, slot: _CpuSlot) -> None:
        with self._lock:
            slot.dead = True
            slot.accepting = False
            slot.held = 0
            returned: list[MediaJob] = []
            while True:
                try:
                    item = slot.queue.get_nowait()
                except queue.Empty:
                    break
                if item is None or item is _BATCH_END:
                    continue
                returned.append(item)
                slot.paths.discard(_path_key(item[1]))
            for job in reversed(returned):
                self._steal.appendleft(job)

    def _run(self, job: MediaJob, model, label: str) -> None:
        try:
            _run_one(job, model, self._model_name, label, self._record, self._hooks)
        finally:
            _, path = job
            key = _path_key(path)
            with self._lock:
                for slot in self._slots:
                    slot.paths.discard(key)

    def _record(
        self,
        video: ChannelVideo,
        path: Path,
        ok: bool,
        error: str | None,
        label: str,
    ) -> None:
        with self._lock:
            if ok:
                self.ok += 1
            else:
                self.err += 1
        if self._hooks.on_result is not None:
            self._hooks.on_result(video, path, ok, error, label)


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

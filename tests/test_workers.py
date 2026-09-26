from pathlib import Path

from whisper_fpmi.vk import ChannelVideo
from whisper_fpmi.workers import JobQueue, plan_workers


def test_plan_20_cores_with_gpu():
    plan = plan_workers(use_gpu=True, cpu_cores=20)
    assert plan.gpu_workers == 1
    assert plan.cpu_workers == 4
    assert plan.cpu_threads == 5
    assert plan.total_cpu_threads == 20
    assert plan.cpu_compute == "int8"
    assert "GPU" in plan.describe()
    assert "CPU" in plan.describe()


def test_plan_cpu_only_still_splits_cores():
    plan = plan_workers(use_gpu=False, cpu_cores=20)
    assert plan.gpu_workers == 0
    assert plan.cpu_workers == 4
    assert plan.cpu_threads == 5


def test_plan_disable_cpu_keeps_gpu():
    plan = plan_workers(use_gpu=True, cpu_workers=0)
    assert plan.gpu_workers == 1
    assert plan.cpu_workers == 0
    assert plan.cpu_threads == 0


def test_plan_explicit_split():
    plan = plan_workers(use_gpu=True, cpu_workers=2, cpu_threads=10)
    assert plan.cpu_workers == 2
    assert plan.cpu_threads == 10
    assert plan.total_cpu_threads == 20


def test_plan_no_workers_falls_back_to_cpu():
    plan = plan_workers(use_gpu=False, cpu_workers=0, gpu_workers=0)
    assert plan.gpu_workers == 0
    assert plan.cpu_workers >= 1


def test_job_queue_cpu_leaves_last_for_gpu():
    jobs = [
        (ChannelVideo("1", "a", "u"), Path("a.mp4")),
        (ChannelVideo("2", "b", "u"), Path("b.mp4")),
        (ChannelVideo("3", "c", "u"), Path("c.mp4")),
    ]
    queue = JobQueue(jobs, gpu_workers=1)
    first = queue.take(is_cpu=True, wait=False)
    second = queue.take(is_cpu=True, wait=False)
    assert first is not None and second is not None
    assert queue.take(is_cpu=True, wait=False) is None
    last = queue.take(is_cpu=False, wait=False)
    assert last is not None
    assert last[0].video_id == "3"


def test_job_queue_cpu_drains_when_gpu_closed():
    jobs = [(ChannelVideo("1", "a", "u"), Path("a.mp4"))]
    queue = JobQueue(jobs, gpu_workers=1)
    assert queue.take(is_cpu=True, wait=False) is None
    queue.close_gpu()
    leftover = queue.take(is_cpu=True, wait=False)
    assert leftover is not None
    assert leftover[0].video_id == "1"

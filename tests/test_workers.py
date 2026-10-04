from pathlib import Path

from whisper_fpmi.vk import ChannelVideo
from whisper_fpmi.workers import JobQueue, plan_workers


def test_plan_gpu_gets_independent_cpu_lane():
    plan = plan_workers(use_gpu=True, cpu_cores=20)
    assert plan.gpu_workers == 1
    assert plan.cpu_workers == 1
    assert plan.cpu_threads == 10
    assert plan.cpu_queue == 10
    assert plan.independent_cpu is True
    assert "GPU не ждёт" in plan.describe()


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


def test_cpu_lane_takes_ten_and_leaves_the_rest_for_gpu(tmp_path: Path):
    from whisper_fpmi.pipeline import _handoff_cpu
    from whisper_fpmi.workers import CpuLane, TranscribeHooks

    plan = plan_workers(use_gpu=True)
    lane = CpuLane(plan, "large-v3", TranscribeHooks())
    jobs = [
        (ChannelVideo(str(index), f"лекция {index}", "u"), tmp_path / f"{index}.mp4")
        for index in range(12)
    ]
    for _video, path in jobs:
        path.write_bytes(b"x")
    taken, rest = _handoff_cpu(jobs, lane)
    assert len(taken) == 10
    assert [job[0].video_id for job in rest] == ["10", "11"]
    assert len(lane.owned_paths()) == 10
    assert _handoff_cpu(rest, lane)[0] == []


def test_two_cpus_take_ten_each_then_next_batch_from_gpu(tmp_path: Path):
    from whisper_fpmi.pipeline import _handoff_cpu
    from whisper_fpmi.workers import CpuLane, TranscribeHooks, _BATCH_END

    plan = plan_workers(use_gpu=True, cpu_workers=2, cpu_threads=8)
    lane = CpuLane(plan, "large-v3", TranscribeHooks())
    jobs = [
        (ChannelVideo(str(index), f"лекция {index}", "u"), tmp_path / f"{index}.mp4")
        for index in range(25)
    ]
    for _video, path in jobs:
        path.write_bytes(b"x")
    taken, rest = _handoff_cpu(jobs, lane)
    assert [job[0].video_id for job in taken] == [str(index) for index in range(20)]
    assert [job[0].video_id for job in rest] == [str(index) for index in range(20, 25)]
    assert _handoff_cpu(rest, lane)[0] == []
    lane.arm_gpu(rest)
    slot = lane._slots[0]
    while True:
        item = slot.queue.get_nowait()
        if item is _BATCH_END:
            break
    lane._refill(slot, 1)
    assert slot.held == 5
    assert lane.gpu_pending() == 0
    assert lane._slots[1].held == 10


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

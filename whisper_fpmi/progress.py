from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
import threading
from typing import Iterable

from whisper_fpmi.vk import ChannelVideo

HOUR = 3600.0
MINUTE = 60.0
# У каждого CPU своя строка. Скачивание и GPU занимают строки ниже.
CPU_BAR_POSITION = 2
CPU_BAR_ROWS = 4
MEDIA_BAR_POSITION = CPU_BAR_POSITION + CPU_BAR_ROWS


def audio_seconds(videos: Iterable[ChannelVideo]) -> float:
    return float(sum(video.duration or 0.0 for video in videos))


def jobs_audio(batch: Iterable[tuple[ChannelVideo, Path]]) -> float:
    return audio_seconds(video for video, _ in batch)


def probe_media_duration(path: Path) -> float | None:
    if not path.exists():
        return None
    commands = [
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
    ]
    for command in commands:
        try:
            raw = subprocess.check_output(
                command,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
        try:
            value = float(raw.strip())
        except ValueError:
            continue
        if value > 0:
            return value
    return None


def eta_seconds(elapsed: float, done: float, total: float) -> float | None:
    if elapsed <= 0 or done <= 0 or total <= 0:
        return None
    remaining = max(0.0, total - done)
    return remaining * elapsed / done


def format_span(seconds: float) -> str:
    """Короткий остаток как мм:сс, долгий — часы и дни вместе."""
    seconds = max(0.0, float(seconds))
    if seconds < HOUR:
        total = int(round(seconds))
        minutes, secs = divmod(total, 60)
        return f"{minutes:02d}:{secs:02d}"
    hours = seconds / HOUR
    days = hours / 24.0
    return f"{hours:.1f} ч ({days:.1f} д)"


def format_audio_left(hours: float) -> str:
    left = max(0.0, float(hours))
    return f"{left:.0f} ч ({left / 24.0:.1f} д)"


@dataclass
class ProgressState:
    total_files: int
    total_audio: float
    done_files: int = 0
    done_audio: float = 0.0
    batch_files: int = 0
    batch_audio: float = 0.0
    batch_done_files: int = 0
    batch_done_audio: float = 0.0
    active: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def use_audio(self) -> bool:
        return self.total_audio > 0

    def start_batch(self, files: int, audio: float) -> None:
        self.batch_files = files
        self.batch_audio = audio
        self.batch_done_files = 0
        self.batch_done_audio = 0.0
        self.active = {
            label: value
            for label, value in self.active.items()
            if _is_cpu_label(label)
        }

    def start_video(self, label: str, duration: float | None) -> None:
        self.active[label] = (0.0, float(duration or 0.0))

    def update_video(self, label: str, pos: float, duration: float | None) -> None:
        previous = self.active.get(label, (0.0, 0.0))
        dur = float(duration or previous[1] or 0.0)
        self.active[label] = (max(0.0, pos), dur)

    def finish_video(self, label: str) -> tuple[float, float]:
        pos, dur = self.active.pop(label, (0.0, 0.0))
        added = dur if dur > 0 else pos
        self.done_files += 1
        self.done_audio += added
        if not _is_cpu_label(label):
            self.batch_done_files += 1
            self.batch_done_audio += added
        return pos, dur

    def audio_now(self) -> float:
        return self.done_audio + sum(pos for pos, _ in self.active.values())

    def batch_audio_now(self) -> float:
        return self.batch_done_audio + sum(
            pos for pos, _ in self._active_values(include_cpu=False)
        )

    def fraction_active(self, *, include_cpu: bool = True) -> float:
        total = 0.0
        for pos, dur in self._active_values(include_cpu=include_cpu):
            if dur > 0:
                total += min(pos / dur, 1.0)
        return total

    def _active_values(self, *, include_cpu: bool):
        for label, value in self.active.items():
            if include_cpu or not _is_cpu_label(label):
                yield value

    def overall_n(self) -> float:
        if self.use_audio:
            return self.audio_now() / HOUR
        return float(self.done_files) + self.fraction_active()

    def overall_total(self) -> float:
        if self.use_audio:
            return max(self.total_audio / HOUR, 0.001)
        return float(max(self.total_files, 1))

    def batch_n(self) -> float:
        if self.batch_audio > 0:
            return self.batch_audio_now() / HOUR
        return float(self.batch_done_files) + self.fraction_active(include_cpu=False)

    def batch_total(self) -> float:
        if self.batch_audio > 0:
            return max(self.batch_audio / HOUR, 0.001)
        return float(max(self.batch_files, 1))

    def overall_unit(self) -> str:
        return "ч" if self.use_audio else "файл"

    def batch_unit(self) -> str:
        return "ч" if self.batch_audio > 0 else "файл"


def _hour_day_bar_class():
    from tqdm import tqdm

    class HourDayBar(tqdm):
        @property
        def format_dict(self):
            data = super().format_dict
            elapsed = float(data.get("elapsed") or 0)
            done = float(data.get("n") or 0)
            total = float(data.get("total") or 0)
            left = eta_seconds(elapsed, done, total)
            data["left"] = format_span(left) if left is not None else "?"
            return data

    return HourDayBar


_HOUR_DAY_BAR = None


def HourDayTqdm(*args, **kwargs):
    """Полоска tqdm, где долгий остаток виден и в часах, и в днях."""
    global _HOUR_DAY_BAR
    if _HOUR_DAY_BAR is None:
        _HOUR_DAY_BAR = _hour_day_bar_class()
    return _HOUR_DAY_BAR(*args, **kwargs)


class RunProgress:
    BAR_FORMAT = (
        "{desc}: {percentage:3.0f}%|{bar}| "
        "{n_fmt}/{total_fmt} "
        "прошло {elapsed} осталось {left}"
    )

    def __init__(
        self,
        *,
        total_files: int,
        total_audio: float,
        disable: bool | None = None,
    ) -> None:
        from tqdm import tqdm

        tqdm.set_lock(threading.RLock())
        self.state = ProgressState(total_files=total_files, total_audio=total_audio)
        self._disable = sys.stderr.isatty() is False if disable is None else disable
        self._tqdm = tqdm
        self._lock = threading.RLock()
        self._positions: dict[str, int] = {}
        self._next_video_pos = MEDIA_BAR_POSITION
        self.overall = self._bar(
            desc=f"Всего ({self.state.overall_unit()})",
            total=self.state.overall_total(),
            position=0,
            leave=True,
        )
        self.batch = self._bar(
            desc="Пачка",
            total=1.0,
            position=1,
            leave=True,
        )
        self.videos: dict[str, object] = {}
        self._sync()

    def start_batch(self, cycle: int, files: int, audio: float) -> None:
        with self._lock:
            self.state.start_batch(files, audio)
            self.batch.reset(total=self.state.batch_total())
            self.batch.set_description(
                f"Пачка {cycle} ({self.state.batch_unit()})",
                refresh=False,
            )
            self._sync_locked()

    def start_video(self, label: str, video: ChannelVideo) -> None:
        with self._lock:
            duration = float(video.duration or 0.0)
            self.state.start_video(label, duration)
            previous = self.videos.pop(label, None)
            if previous is not None:
                previous.close()
            self.videos[label] = self._video_bar(label, video.title, duration)
            self._sync_locked()

    def update_video(self, label: str, pos: float, duration: float) -> None:
        with self._lock:
            if label not in self.state.active:
                self.state.start_video(label, duration)
            self.state.update_video(label, pos, duration)
            bar = self.videos.get(label)
            if bar is None:
                bar = self._video_bar(label, label, duration)
                self.videos[label] = bar
            total = max(duration or self.state.active.get(label, (0.0, 0.0))[1], 0.001)
            bar.total = total / MINUTE
            bar.n = min(pos, duration or pos) / MINUTE
            bar.refresh()
            self._sync_locked()

    def finish_video(self, label: str) -> None:
        with self._lock:
            self.state.finish_video(label)
            bar = self.videos.pop(label, None)
            if bar is not None:
                bar.close()
            self._sync_locked()

    def close(self) -> None:
        with self._lock:
            for bar in self.videos.values():
                bar.close()
            self.videos.clear()
            self.batch.close()
            self.overall.close()

    def _bar(self, *, desc: str, total: float, position: int, leave: bool):
        return HourDayTqdm(
            total=total,
            desc=desc,
            position=position,
            leave=leave,
            dynamic_ncols=True,
            mininterval=0.3,
            bar_format=self.BAR_FORMAT,
            disable=self._disable,
            file=sys.stderr,
        )

    def _video_bar(self, label: str, title: str, duration: float):
        if _is_cpu_label(label):
            self._positions[label] = _cpu_bar_position(label)
        elif label not in self._positions:
            self._positions[label] = self._next_video_pos
            self._next_video_pos += 1
        short = _short_title(title)
        total = max(duration, 0.001) / MINUTE
        return self._bar(
            desc=f"{label} {short} (мин)",
            total=total,
            position=self._positions[label],
            leave=False,
        )

    def _sync(self) -> None:
        with self._lock:
            self._sync_locked()

    def _sync_locked(self) -> None:
        if self.state.done_files > self.state.total_files:
            self.state.total_files = self.state.done_files
        if self.state.done_audio > self.state.total_audio:
            self.state.total_audio = self.state.done_audio
        overall_n = self.state.overall_n()
        self.overall.total = max(self.state.overall_total(), overall_n, 0.001)
        self.overall.n = overall_n
        self.overall.set_postfix_str(
            _files_postfix(
                self.state.done_files,
                self.state.total_files,
                overall_n,
                self.overall.total,
                self.state.overall_unit(),
            ),
            refresh=False,
        )
        self.overall.refresh()
        batch_n = self.state.batch_n()
        self.batch.total = max(self.state.batch_total(), batch_n, 0.001)
        self.batch.n = batch_n
        self.batch.set_postfix_str(
            _files_postfix(
                self.state.batch_done_files,
                self.state.batch_files,
                batch_n,
                self.batch.total,
                self.state.batch_unit(),
            ),
            refresh=False,
        )
        self.batch.refresh()


def _files_postfix(
    done_files: int,
    total_files: int,
    done_amount: float,
    total_amount: float,
    unit: str,
) -> str:
    text = f"файлов {done_files}/{total_files}"
    if unit == "ч":
        left = max(0.0, float(total_amount) - float(done_amount))
        text = f"{text}, ещё {format_audio_left(left)}"
    return text


def _is_cpu_label(label: str) -> bool:
    return label.startswith("CPU")


def _cpu_bar_position(label: str) -> int:
    tail = label.removeprefix("CPU").lstrip("-")
    index = int(tail) - 1 if tail.isdigit() else 0
    index = min(max(index, 0), CPU_BAR_ROWS - 1)
    return CPU_BAR_POSITION + index


def _short_title(title: str, limit: int = 28) -> str:
    text = " ".join(title.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"

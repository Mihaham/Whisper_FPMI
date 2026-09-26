from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger as lg

from whisper_fpmi.paths import (
    CHANNEL_URL,
    DEFAULT_CPU_CORES,
    DEFAULT_MAX_GB,
    DEFAULT_MODEL,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisper_fpmi",
        description=(
            "Скачивать лекции ФПМИ с VK пачками до 10 ГиБ, "
            "расшифровывать Whisper large-v3 и удалять исходники."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", help="Пересобрать CATALOG.md из result/")
    catalog.set_defaults(func=_cmd_catalog)

    listing = sub.add_parser("list", help="Показать видео канала VK")
    listing.add_argument("--channel", default=CHANNEL_URL)
    listing.add_argument("--limit", type=int, default=None)
    listing.set_defaults(func=_cmd_list)

    download = sub.add_parser("download", help="Скачать одно видео в худшем качестве")
    download.add_argument("url")
    download.add_argument("--title", default=None)
    download.add_argument("--cookies", default=None)
    download.set_defaults(func=_cmd_download)

    transcribe = sub.add_parser("transcribe", help="Расшифровать локальный файл")
    transcribe.add_argument("path", type=Path)
    transcribe.add_argument("--title", default=None)
    transcribe.add_argument("--url", default="")
    transcribe.add_argument("--model", default=DEFAULT_MODEL)
    transcribe.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    transcribe.set_defaults(func=_cmd_transcribe)

    run = sub.add_parser(
        "run",
        help="Скачивать пачками до N ГиБ, расшифровывать, удалять исходники и повторять",
    )
    run.add_argument("--channel", default=CHANNEL_URL)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--cookies", default=None)
    run.add_argument(
        "--max-gb",
        type=float,
        default=DEFAULT_MAX_GB,
        help="Максимум гигабайт в videos/ на один круг (по умолчанию 10)",
    )
    run.add_argument("--keep-video", action="store_true")
    run.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Перешифровать даже если текст уже лежит в result/",
    )
    run.add_argument(
        "--cpu-cores",
        type=int,
        default=DEFAULT_CPU_CORES,
        help="Сколько CPU-потоков отдать Whisper (по умолчанию 20 из 28, запас ffmpeg/системе)",
    )
    run.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="Сколько лекций параллельно на CPU (по умолчанию 4). 0 — только GPU",
    )
    run.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Потоков CTranslate2 на один CPU-воркер (по умолчанию cpu-cores / cpu-workers)",
    )
    run.add_argument(
        "--gpu-workers",
        type=int,
        default=None,
        help="Параллельных расшифровок на GPU (по умолчанию 1). 0 — только CPU",
    )
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = build_parser().parse_args(argv)
    return args.func(args)


def _setup_logging() -> None:
    from tqdm import tqdm

    lg.remove()
    lg.add(
        lambda message: tqdm.write(message.rstrip("\n")),
        format="<level>{time:HH:mm:ss} | {level} | {message}</level>",
        level="INFO",
        colorize=True,
    )
    lg.add(
        "debug.txt",
        format="{time} | {level} | {file}:{line} | {message}",
        level="DEBUG",
        rotation="50 MB",
        encoding="utf-8",
    )


def _cmd_catalog(_args: argparse.Namespace) -> int:
    from whisper_fpmi.catalog import write_catalog

    path = write_catalog()
    lg.info("Записан {}", path)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from whisper_fpmi.vk import list_channel_videos

    videos = list_channel_videos(args.channel)
    total = len(videos)
    if args.limit:
        videos = videos[: args.limit]
    for index, video in enumerate(videos, start=1):
        print(f"{index}. {video.video_id}\t{video.title}\t{video.url}")
    lg.info("Показано {} из {}", len(videos), total)
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from whisper_fpmi.download import download_worst_quality
    from whisper_fpmi.vk import ChannelVideo

    video_id = args.url.rstrip("/").split("video")[-1]
    title = args.title or video_id
    video = ChannelVideo(video_id=video_id, title=title, url=args.url)
    path = download_worst_quality(video, cookies=args.cookies)
    print(path)
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    from whisper_fpmi.pipeline import transcribe_local

    transcribe_local(
        args.path,
        title=args.title,
        source_url=args.url,
        model_name=args.model,
        device=args.device,
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from whisper_fpmi.pipeline import run_pipeline

    return run_pipeline(
        channel_url=args.channel,
        model_name=args.model,
        device=args.device,
        limit=args.limit,
        skip_existing=not args.no_skip_existing,
        keep_video=args.keep_video,
        cookies=args.cookies,
        max_gb=args.max_gb,
        cpu_cores=args.cpu_cores,
        cpu_workers=args.cpu_workers,
        cpu_threads=args.cpu_threads,
        gpu_workers=args.gpu_workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())

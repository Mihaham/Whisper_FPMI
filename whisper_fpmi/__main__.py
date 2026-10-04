from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger as lg

from whisper_fpmi.paths import (
    CHANNEL_URL,
    DEFAULT_CPU_CORES,
    DEFAULT_CPU_LANE_QUEUE,
    DEFAULT_CPU_LANE_THREADS,
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_FRAGMENT_THREADS,
    DEFAULT_MAX_GB,
    DEFAULT_MODEL,
    YOUTUBE_URL,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisper_fpmi",
        description=(
            "Сначала лекции ФПМИ с VK, потом с YouTube: "
            "пачки до 3 ГиБ, Whisper large-v3, удаление исходников."
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
        help="Сначала VK, потом YouTube: скачать, расшифровать, удалить исходники",
    )
    run.add_argument("--channel", default=CHANNEL_URL)
    run.add_argument("--youtube", default=YOUTUBE_URL)
    run.add_argument(
        "--no-youtube",
        action="store_true",
        help="Не брать YouTube после VK",
    )
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--cookies", default=None)
    run.add_argument(
        "--max-gb",
        type=float,
        default=DEFAULT_MAX_GB,
        help="Максимум гигабайт в videos/ на один круг (по умолчанию 3)",
    )
    run.add_argument(
        "--download-workers",
        type=int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        help="Сколько лекций качать одновременно (по умолчанию 4)",
    )
    run.add_argument(
        "--fragment-threads",
        type=int,
        default=DEFAULT_FRAGMENT_THREADS,
        help="HTTP-соединений на один ролик (по умолчанию 8)",
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
        help="Сколько CPU-потоков отдать Whisper, если GPU нет (по умолчанию 20)",
    )
    run.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="CPU-процессов рядом с GPU. По умолчанию 1. 0 — только GPU",
    )
    run.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help=(
            "Потоков на один CPU-процесс. "
            f"Рядом с GPU по умолчанию {DEFAULT_CPU_LANE_THREADS}"
        ),
    )
    run.add_argument(
        "--cpu-queue",
        type=int,
        default=DEFAULT_CPU_LANE_QUEUE,
        help="Сколько видео держать в очереди CPU, пока GPU считает остальное (по умолчанию 10)",
    )
    run.add_argument(
        "--gpu-workers",
        type=int,
        default=None,
        help="Параллельных расшифровок на GPU (по умолчанию 1). 0 — только CPU",
    )
    run.set_defaults(func=_cmd_run)

    build = sub.add_parser(
        "build",
        help="Датасет, облако, граф терминов, поисковый индекс и метрики",
    )
    build.add_argument("--offline", action="store_true", help="Не ходить в VK, брать кэш описаний")
    build.add_argument("--refresh-descriptions", action="store_true")
    build.add_argument("--workers", type=int, default=4)
    build.set_defaults(func=_cmd_build)

    cloud = sub.add_parser("wordcloud", help="Пересобрать облако слов")
    cloud.add_argument("--check", action="store_true", help="Код 1, если облако устарело")
    cloud.add_argument("--if-stale", action="store_true", help="Пересобрать только если тексты изменились")
    cloud.set_defaults(func=_cmd_wordcloud)

    finding = sub.add_parser("search", help="Поиск по лекциям без нейросети")
    finding.add_argument("query")
    finding.add_argument("--course", default=None)
    finding.add_argument("--limit", type=int, default=8)
    finding.set_defaults(func=_cmd_search)

    asking = sub.add_parser("rag", help="Ответ с цитатами по найденным фрагментам")
    asking.add_argument("query")
    asking.add_argument("--course", default=None)
    asking.add_argument("--limit", type=int, default=5)
    asking.set_defaults(func=_cmd_rag)

    site = sub.add_parser("serve", help="Страница поиска и графа на localhost")
    site.add_argument("--port", type=int, default=8765)
    site.set_defaults(func=_cmd_serve)
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


def _cmd_build(args: argparse.Namespace) -> int:
    from whisper_fpmi.analytics import build_all

    metrics = build_all(
        offline=args.offline,
        refresh=args.refresh_descriptions,
        workers=args.workers,
    )
    retrieval = metrics.get("retrieval") or {}
    chapters = retrieval.get("chapters") or {}
    titles = retrieval.get("titles") or {}
    rag = retrieval.get("rag") or {}
    lg.info(
        "Лекций {}, с таймкодами {}, таймкодов с полем часа {}",
        metrics.get("lectures"),
        metrics.get("with_timecodes"),
        metrics.get("timecodes_with_hour_field"),
    )
    lg.info(
        "Главы: hit@5 {}, MRR@10 {}, граница ±90с {}",
        chapters.get("hit_rate_at_5"),
        chapters.get("mrr_at_10"),
        chapters.get("boundary_hit_at_90s_at_5"),
    )
    lg.info("Названия: hit@1 {}, MRR@10 {}", titles.get("hit_rate_at_1"), titles.get("mrr_at_10"))
    lg.info(
        "RAG: попадание цитаты {}, опора на фрагменты {}",
        rag.get("support_hit_rate"),
        rag.get("grounded"),
    )
    return 0


def _cmd_wordcloud(args: argparse.Namespace) -> int:
    from whisper_fpmi.analytics import cloud_check, rebuild_cloud

    if args.check or args.if_stale:
        fresh, message = cloud_check()
        if fresh or args.check:
            print(message)
            return 0 if fresh else 1
    rebuild_cloud()
    print("облако обновлено")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from whisper_fpmi.analytics import format_hits, run_search

    hits = run_search(args.query, limit=args.limit, course=args.course)
    print(format_hits(hits, args.query))
    return 0


def _cmd_rag(args: argparse.Namespace) -> int:
    from whisper_fpmi.analytics import run_rag

    answer = run_rag(args.query, limit=args.limit, course=args.course)
    print(answer.answer)
    print()
    for citation in answer.citations:
        print(f"[{citation['rank']}] {citation['clock']} {citation['course']} — {citation['title']}")
        if citation.get("url"):
            print(f"    {citation['url']}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from whisper_fpmi.serve import serve

    serve(args.port)
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
        cpu_queue=args.cpu_queue,
        gpu_workers=args.gpu_workers,
        download_workers=args.download_workers,
        fragment_threads=args.fragment_threads,
        include_youtube=not args.no_youtube,
        youtube_url=args.youtube,
    )


if __name__ == "__main__":
    raise SystemExit(main())

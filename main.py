'''Main script for the WHISPER of FPMI'''
import sys

from loguru import logger as lg

from cache import load_cache, save_cache
from download import download, check_folders, get_videos
from whisper import whisper

CACHE_DOWNLOAD = "whisper_cache"
CACHE_FILE = "whisper_file_cache"
CHANEL = "UCdxesVp6Fs7wLpnp1XKkvZg"


@lg.catch()
def main() -> None:
    '''Whisperring all videos from channel'''

    lg.remove()
    lg.level("INFO", color="<cyan>")
    lg.add("debug.txt", format="{time} | {level} | {file} | {line} | {message}", level="DEBUG",
           rotation="100 MB",
           colorize=True, compression="zip")
    lg.add(sys.stdout,
           format="<level><b>{time} | {level} | {file} | {line} | {message}</b></level>",
           level="DEBUG",
           colorize=True)
    cache_download = load_cache(CACHE_DOWNLOAD)
    cache_file = load_cache(CACHE_FILE)

    lg.info(f"All folders found: {check_folders()}")

    videos = get_videos(CHANEL)
    for video in videos:
        link = "https://www.youtube.com/watch?v=" + video['videoId']
        download(link, cache_download)
        save_cache(cache_download, CACHE_DOWNLOAD)
        whisper(cache_file)
        save_cache(cache_file, CACHE_FILE)


if __name__ == '__main__':
    main()

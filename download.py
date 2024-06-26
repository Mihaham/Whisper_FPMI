'''Module for downloading videos from Youtube'''
import os
import scrapetube

from pytube import YouTube
from pytube.cli import on_progress

from loguru import logger as lg


# Add folders
def check_folders() -> bool:
    '''Adding folders if thet don't already exist'''

    flag = True

    check_content_folder = os.path.exists("content")
    check_down_load_folder = os.path.exists("download")
    check_high_folder = os.path.exists("high")
    check_part_folder = os.path.exists("parts")

    if not check_part_folder:
        os.mkdir("parts")
        lg.warning("Content folder was created")
        flag = False
    if not check_content_folder:
        os.mkdir("content")
        lg.warning("Content folder was created")
        flag = False
    if not check_down_load_folder:
        os.mkdir("download")
        lg.warning("Download folder was created")
        flag = False
    if not check_high_folder:
        os.mkdir("high")
        lg.warning("High folder was created")
        flag = False

    return flag


@lg.catch()
def download(video_url: str, files: list) -> int:
    '''Download file from YouTube url and save it to file'''
    yt = YouTube(video_url, on_progress_callback=on_progress)
    lg.info(f"Downloading video {yt.title}")
    if yt.title in files:
        lg.warning(f"File is already downloaded: {video_url}: {yt.title}")
        return 0
    lg.debug("Downloading video")
    stream = yt.streams.get_highest_resolution()
    output_path = "high/"
    stream.download(output_path)
    lg.debug("Adding file to cache")
    files.append(yt.title)
    lg.info(f"Download complete! {video_url}: {yt.title}")
    return 1


def get_videos(chanel) -> list:
    '''Getting videos from YouTube chanel'''
    videos = scrapetube.get_channel(chanel)
    lg.info(f"Got all videos from channel {chanel}")
    return videos

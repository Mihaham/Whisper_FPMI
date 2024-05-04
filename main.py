import os
import json
# Add folders
checkContentFolder = os.path.exists("content")
checkDownLoadFolder = os.path.exists("download")
checkHighFolder = os.path.exists("high")
if not checkContentFolder:
  os.mkdir("content")
if not checkDownLoadFolder:
  os.mkdir("download")
if not checkHighFolder:
  os.mkdir("high")
from inputimeout import inputimeout, TimeoutOccurred

from pytube import YouTube
from pytube.cli import on_progress

CACHE_FILE = "whisper_cache"
VERBOSE = True

def load_cache():
    files = []
    with open(CACHE_FILE, "r") as f:
        for line in f.readlines():
            files.append(json.loads(line))
    print(f"Loaded cache: {files}")
    return files

def save_cache(files):
    with open(CACHE_FILE, "w") as f:
        for file in files:
            f.write(json.dumps(file))
            f.write("\n")


def download(video_url, files):
  yt = YouTube(video_url, on_progress_callback=on_progress)
  if yt.title.find("ОКТЧ") != -1 and yt.title.find("ОКТЧ 13. Эйлеровость графа.") == -1 or 1:
    print("______________________________________________________________")
    print(yt.title)
    if yt.title not in files:
        files.append(yt.title)
        save_cache(files)
    else:
        print(f"File is already downloaded: {video_url}: {yt.title}")
        return 0
    try:
        stream = yt.streams.get_highest_resolution()

        output_path = "high/"

        stream.download(output_path)
        print(f"Download complete! {video_url}: {yt.title}")
    except Exception as e:
        print(f"SMTH went wrong with video: {video_url}: {yt.title}")
        print(e)
    print("______________________________________________________________")
    try:
        c = inputimeout(prompt="Если хотите остановить программу, введите q\n", timeout=5)
    except TimeoutOccurred:
        c = "timeout"
    if (c + "123")[0] == "q":
        exit()
    print("______________________________________________________________")
    return 1
  return 0

import scrapetube
videos = scrapetube.get_channel("UCdxesVp6Fs7wLpnp1XKkvZg")
amount = 0
files = load_cache()
for video in videos:
    link = "https://www.youtube.com/watch?v="+video['videoId']
    print(f"Downloading video {link}")
    amount += download(link, files)



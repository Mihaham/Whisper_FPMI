'''Script for extracting subclips'''
import os

from os import listdir
from os.path import isfile, join
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def main():
    '''Main script'''
    mypath = "high/"

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file_name in onlyfiles:
        os.rename(join(mypath, file_name), join(mypath, file_name.replace(" ", "-")))

    mypath = "download/"

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    rows = []

    for file_name in onlyfiles:
        with open(join(mypath, file_name), "r", encoding="utf-8") as file:
            data = file.readlines()
            for row in data:
                if row.find("друзья") != -1 or row.find("товарищ") != -1:
                    if file_name.find("Доп") == -1:
                        rows.append([file_name, row])

    for row in rows:
        print(row)
    print(len(rows))

    movie_path = "/mnt/c/Users/misha/PycharmProjects/WHISPER/high"


    for index, part in enumerate(rows):
        file_name = join(movie_path, part[0][:-4])
        row = part[1]
        data = row.split("]")
        timecode = data[0]
        start, end = timecode.split(" --> ")
        start = start[1:]

        def get_time(timing):
            timing = timing.split(".")[0].split(":")
            timing.reverse()
            tm = 1
            result = 0
            for time in timing:
                result += int(time) * tm
                tm *= 60
            return result

        start = get_time(start)
        end = get_time(end)

        print(file_name, start, end)
        ffmpeg_extract_subclip(file_name, start, end, targetname=f"parts/{index}.mp4")


if __name__ == "__main__":
    main()

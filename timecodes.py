import os
from os import listdir
from os.path import isfile, join

mypath = "high/"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for fileName in onlyfiles:
    new_name = join(mypath, fileName.replace(" ", "-"))
    os.rename(join(mypath, fileName), join(mypath, fileName.replace(" ", "-")))

mypath = "download/"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
rows = []

for fileName in onlyfiles:
    with open(join(mypath, fileName), "r") as file:
        data = file.readlines()
        for row in data:
            if row.find("друзья") != -1 or row.find("товарищ") != -1:
                if fileName.find("Доп") == -1:
                    rows.append([fileName, row])

for row in rows:
    print(row)
print(len(rows))

path = "/download"
movie_path = "/mnt/c/Users/misha/PycharmProjects/WHISPER/high"
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

checkPartFolder = os.path.exists("parts")
if not checkPartFolder:
    os.mkdir("parts")

for index, part in enumerate(rows):
    fileName = join(movie_path, part[0][:-4])
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

    print(fileName, start, end)
    try:
        ffmpeg_extract_subclip(fileName, start, end, targetname=f"parts/{index}.mp4")
    except:
        pass

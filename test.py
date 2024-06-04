from os import listdir
from os.path import isfile, join
import os
import json
from inputimeout import inputimeout, TimeoutOccurred
import time

mypath = "high/"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

CACHE_FILE = "whisper_file_cache"
VERBOSE = True
i = 0

def load_cache():
    files = []
    with open(CACHE_FILE, "r") as f:
        for line in f.readlines():
            files.append(json.loads(line))

    return files

def save_cache(files):
    with open(CACHE_FILE, "w") as f:
        for file in files:
            f.write(json.dumps(file))
            f.write("\n")

files = load_cache()
files.sort()

print(f"Loaded cache: ")
for i,file in enumerate(files):
    print(f"{i+1}. {file}")




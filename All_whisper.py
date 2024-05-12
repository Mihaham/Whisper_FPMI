import json

CACHE_FILE = "whisper_file_cache"
VERBOSE = True

def load_cache():
    files = []
    with open(CACHE_FILE, "r") as f:
        for line in f.readlines():
            files.append(json.loads(line))
    print(f"Loaded cache: ")
    for i,file in enumerate(files):
        print(f"{i}. {file}")
    return files

load_cache()
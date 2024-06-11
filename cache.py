import json

from loguru import logger as lg


def load_cache(CACHE_FILE: str) -> list:
    '''Loading cache from file'''
    lg.debug(f"Loading cache: {CACHE_FILE}")
    files = []
    with open(CACHE_FILE, "r") as f:
        for line in f.readlines():
            files.append(json.loads(line))
    lg.info(f"Loaded cache {CACHE_FILE}")
    files.sort()
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    return files


def save_cache(files: list, CACHE_FILE: str) -> None:
    '''Saving cache to file'''
    lg.debug(f"Saving cache: {CACHE_FILE}")
    with open(CACHE_FILE, "w") as f:
        for file in files:
            f.write(json.dumps(file))
            f.write("\n")


def check_cache(files: list, name):
    '''Сhecks if the file exists in cache'''
    lg.debug(f"Checking if file exists in cache: {name}")
    return name in files

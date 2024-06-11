'''This module helps to cache the results'''
import json

from loguru import logger as lg


def load_cache(cache_file: str) -> list:
    '''Loading cache from file'''
    lg.debug(f"Loading cache: {cache_file}")
    files = []
    with open(cache_file, "r", encoding="UTF-8") as f:
        for line in f.readlines():
            files.append(json.loads(line))
    lg.info(f"Loaded cache {cache_file}")
    files.sort()
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    return files


def save_cache(files: list, cache_file: str) -> None:
    '''Saving cache to file'''
    lg.debug(f"Saving cache: {cache_file}")
    with open(cache_file, "w", encoding="UTF-8") as f:
        for file in files:
            f.write(json.dumps(file))
            f.write("\n")


def check_cache(files: list, name):
    '''Сhecks if the file exists in cache'''
    lg.debug(f"Checking if file exists in cache: {name}")
    return name in files

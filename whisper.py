from datetime import datetime
from os import listdir
from os.path import isfile, join
import os
import json
from inputimeout import inputimeout, TimeoutOccurred
import time

from loguru import logger as lg

mypath = "high/"

def refactor_name(fileName : str) -> str:
    '''Refactoring filename to save mode for console'''
    new_name = join(mypath, fileName.replace(" ", "-"))
    new_name = new_name.replace("(", "-")
    new_name = new_name.replace(")", "-")
    new_name = new_name.replace("&", "-")
    new_name = new_name.replace("|", "-")
    return new_name

def whisper(files)-> None:
    '''Whispers user input'''
    lg.info("Getting all files in folder")
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for fileName in onlyfiles:
      new_name = refactor_name(fileName)
      lg.info(f"Refactored name into : {new_name}")
      os.rename(join(mypath,fileName),new_name)
      lg.debug(f"Renamed file : {new_name}")
      if new_name not in files:
        lg.info(f"Whispering {new_name}, start time = {datetime.now()}")
        start_time = time.time()
        os.system(f"whisper --language ru --model medium -o ./result/ --output_format txt --verbose True -- {new_name} > download/{new_name[5:]}.txt")
        end_time = time.time() - start_time
        lg.info(f"Whispered: {new_name} за {end_time//3600} часов, {(end_time//60)%60} минут, {round(end_time%60,2)} секунд")
        files.append(new_name)
        os.remove(new_name)
        lg.info(f"Deleted file : {new_name}")
      else:
        lg.warning(f"Раньше расшифровано {new_name}")
        os.remove(new_name)
        lg.info(f"Deleted file : {new_name}")
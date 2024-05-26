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
    print(f"Loaded cache: ")
    for i,file in enumerate(files):
        print(f"{i}. {file}")
    return files

def save_cache(files):
    with open(CACHE_FILE, "w") as f:
        for file in files:
            f.write(json.dumps(file))
            f.write("\n")

files = load_cache()

def rename(fileName):
    new_name = join(mypath, fileName.replace(" ", "-"))
    new_name = new_name.replace("(", "-")
    new_name = new_name.replace(")", "-")
    os.rename(join(mypath, fileName), new_name)
    return new_name

def solution(new_name_1, new_name_2, new_name_3):
    if new_name_1 not in files or new_name_2 not in files or new_name_3 not in files:
        print(f"Расшифровываю {new_name_1} и {new_name_2} и {new_name_3}")
        start_time = time.time()
        os.system(
            f"whisper --language ru --model medium -o ./result/ --output_format txt --verbose True -- {new_name_1} > download/{new_name_1[5:]}.txt & whisper --language ru --model medium -o ./result/ --output_format txt --verbose True -- {new_name_2} > download/{new_name_2[5:]}.txt & whisper --language ru --model medium -o ./result/ --output_format txt --verbose True -- {new_name_3} > download/{new_name_3[5:]}.txt ")
        end_time = time.time() - start_time
        print(
            f"Расшифровано: {new_name_1} и {new_name_2} и {new_name_3} за {end_time // 3600} часов, {(end_time // 60) % 60} минут, {end_time % 60} секунд")
        files.append(new_name_1)
        files.append(new_name_2)
        files.append(new_name_3)
        save_cache(files)
        os.remove(new_name_1)
        os.remove(new_name_2)
        os.remove(new_name_3)
    else:
        print(f"Раньше расшифровано {new_name_1} и {new_name_2} и {new_name_3}")
        os.remove(new_name_1)
        os.remove(new_name_2)
        os.remove(new_name_3)
    print("______________________________________________________________")

for i in range(0,len(onlyfiles),3):
  fileName_1 = onlyfiles[i]
  fileName_2 = onlyfiles[i+1]
  fileName_3 = onlyfiles[i+2]
  new_name_1 = rename(fileName_1)
  new_name_2 = rename(fileName_2)
  new_name_3 = rename(fileName_3)
  solution(new_name_1, new_name_2, new_name_3)
  try:
      c = inputimeout(prompt="Если хотите остановить программу, введите q\n", timeout=5)
  except TimeoutOccurred:
      c = "timeout"
  if (c + "123")[0] == "q":
      exit()
  print(f"_____________________{i}______________________________________________________________________________")
  i+=1
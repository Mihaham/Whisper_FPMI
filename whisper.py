from os import listdir
from os.path import isfile, join
import os
import json
from inputimeout import inputimeout, TimeoutOccurred

mypath = "high/"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

CACHE_FILE = "whisper_file_cache"
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

i = 0
files = load_cache()
for fileName in onlyfiles:
  new_name = join(mypath,fileName.replace(" ", "-"))
  new_name = new_name.replace("(","-")
  new_name = new_name.replace(")","-")
  os.rename(join(mypath,fileName),new_name)
  if new_name not in files:
    print(f"Расшифровываю {new_name}")
    os.system(f"whisper --language ru --model medium -o ./result/ --output_format txt --verbose True -- {new_name} > download/{new_name[5:]}.txt")
    print(f"Расшифровано: {new_name}")
    files.append(new_name)
    save_cache(files)
    os.remove(new_name)
  else:
    print(f"Раньше расшифровано {new_name}")
  print("______________________________________________________________")
  try:
      c = inputimeout(prompt="Если хотите остановить программу, введите q\n", timeout=5)
  except TimeoutOccurred:
      c = "timeout"
  if (c + "123")[0] == "q":
      exit()
  print("______________________________________________________________")
  i+=1
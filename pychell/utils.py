# Python default modules
from functools import reduce
import operator
import numpy as np
from pdb import set_trace as stop
import time
import pychell
import os
from google_drive_downloader import GoogleDriveDownloader as gdd

# Helpful timer
class StopWatch:
    
    def __init__(self):
        self.seed = time.time()
        self.laps = {'seed': self.seed}
        
    def time_since(self, name=None):
        if name is None:
            name = 'seed'
        return time.time() - self.laps[name]
    
    def reset(self):
        self.seed = time.time()
        self.laps = {'seed': self.seed}
    
    def lap(self, name):
        self.laps[name] = time.time()
   
# finds all items within a dictionary recursively.
def find_all_items(obj, key, keys=None):
    ret = []
    if not keys:
        keys = []
    if key in obj:
        out_keys = keys + [key]
        ret.append((out_keys, obj[key]))
    for k, v in obj.items():
        if isinstance(v, dict):
            found_items = find_all_items(v, key, keys=(keys+[k]))
            ret += found_items
    return ret

# Helper
def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

# Helper
def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value
    

# Check if the templates path exists
def templates_path_exists(path=None):
    if path is None:
        pychell_path = os.path.dirname(os.path.abspath(pychell.__file__)) + os.sep
        path = pychell_path + 'templates' + os.sep
    if os.path.exists(path):
        return True
    else:
        return False

# Downlaod templates from google drive
def download_templates(overwrite=False):
    pychell_path = os.path.dirname(os.path.abspath(pychell.__file__)) + os.sep
    dest = pychell_path + os.sep + 'templates.zip'
    print('Downloading Templates to')
    print('  ' + dest)
    try:
        gdd.download_file_from_google_drive(file_id='1B_dgE4qfGt1fYHIVMTYiV5r3-kNfUHX4', dest_path=dest, unzip=True, showsize=True, overwrite=overwrite)
        os.remove(dest)
    except:
        print("ERROR: Unable to download templates!")
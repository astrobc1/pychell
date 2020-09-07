# Python default modules
from functools import reduce
import operator
import numpy as np
from pdb import set_trace as stop
import sys
import time
import traceback
import pathlib
import logging
import pychell
from datetime import datetime
import glob
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

# Downlaod templates from google drive
def download_templates(dest, file_id=None):
    
    if file_id is None:
        file_id = '1ubwYyH6DidtDfxRdg707-8M9iORS-jql'
    
    if not os.path.exists(dest):
        os.makedirs(dest)
        
    if os.path.exists(dest + 'templates'):
        raise ValueError("Move existing templates sub folder in templates_path")
        
    print('Downloading Templates to')
    print('  ' + dest)
    
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    dest_zip = dest + 'templates_' + dt_string + '.zip'
        
    try:
        gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_zip, unzip=True, showsize=True, overwrite=False)
        os.remove(dest_zip)
        template_files = glob.glob(dest + 'templates' + os.sep + '*')
        for tfile in template_files:
            p = pathlib.Path(tfile).absolute()
            parent_dir = p.parents[1]
            p.rename(parent_dir / p.name)
        os.rmdir(dest + 'templates')
        print('Success! Remember to set force_download_templates to False for future runs!')
    except Exception as e:
        print("ERROR: Unable to download templates!")
        logging.error(traceback.format_exc())

def get_size(obj, seen=None):
    
    """Recursively finds size of objects"""
    
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
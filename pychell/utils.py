# Python default modules
from functools import reduce
import operator
import numpy as np
from pdb import set_trace as stop
import time
import pychell.rvs.model_components as pcmodels
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
def download_templates(desination):
    gdd.download_file_from_google_drive(file_id='1l9AzgAPI_9v4k1mktIyqJGPhqhIhSrp0',
                                    dest_path=destination,
                                    unzip=True)
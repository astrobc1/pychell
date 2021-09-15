# Python default modules
from functools import reduce
import operator
import numpy as np
import sys
import time
import pickle
import traceback
import importlib
import pathlib
import logging
import pychell
from itertools import chain, combinations
from barycorrpy.utils import get_stellar_data
from datetime import datetime
import glob
import os
from google_drive_downloader import GoogleDriveDownloader as gdd
import webcolors

PLOTLY_COLORS = ['darkmagenta', 'mediumslateblue', 'orangered', 'sienna', 'darkblue', 'teal', 'springgreen', 'seagreen', 'darkseagreen', 'plum', 'indianred', 'lawngreen', 'mediumorchid', 'rosybrown', 'turquoise', 'lightgreen', 'cadetblue', 'mediumblue', 'darkorchid', 'olivedrab', 'darkgreen', 'royalblue', 'mediumspringgreen', 'darkviolet', 'yellowgreen', 'mediumturquoise', 'lightpink', 'mediumaquamarine', 'forestgreen', 'slateblue', 'blue', 'mediumpurple', 'burlywood', 'deepskyblue', 'palegreen', 'magenta', 'darkgoldenrod', 'goldenrod', 'cornflowerblue', 'salmon', 'wheat', 'lime', 'coral', 'chartreuse', 'darkturquoise', 'paleturquoise', 'fuchsia', 'purple', 'maroon', 'mediumvioletred', 'palevioletred', 'violet', 'darkkhaki', 'aquamarine', 'darkred', 'darksalmon', 'black', 'darkcyan', 'dodgerblue', 'gold', 'lightblue', 'moccasin', 'lightseagreen', 'orchid', 'palegoldenrod', 'aqua', 'chocolate', 'tan', 'powderblue', 'orange', 'navy', 'lightskyblue', 'darkorange', 'saddlebrown', 'greenyellow', 'green', 'hotpink', 'blueviolet', 'brown', 'limegreen', 'skyblue', 'firebrick', 'darkolivegreen', 'lightsteelblue', 'khaki', 'cyan', 'indigo', 'olive', 'linen', 'sandybrown', 'lightcyan', 'mediumseagreen', 'peru', 'steelblue', 'pink', 'red', 'midnightblue', 'deeppink', 'crimson', 'tomato']

COLORS_HEX_GADFLY = ['#00BEFF', '#D4CA3A', '#FF6DAE', '#67E1B5', '#EBACFA', '#9E9E9E', '#F1988E', '#5DB15A', '#E28544', '#52B8AA']

def hex_to_rgba(h, a=1.0):
    h = h.strip("#")
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    s = 'rgba(' + str(r) + ',' + str(g) + ',' + str(b) + ',' + str(a) + ')'
    return s

def csscolor_to_rgba(color, a=1.0):
    r, g, b = webcolors.name_to_rgb(color)
    s = 'rgba(' + str(r) + ',' + str(g) + ',' + str(b) + ',' + str(a) + ')'
    return s

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

def gendatestr(time=False):
    now = datetime.now()
    if time:
        dt_string = now.strftime("%Y%m%d_%H%M%S")
    else:
        dt_string = now.strftime("%Y%m%d")
    return dt_string
     
def dict_diff(d1, d2):
    out = {}
    common_keys = set(d1) - set(d2)
    for key in common_keys:
        if key in d1:
            out[key] = d1[key]
    return out

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def module_from_file(fname):
    spec = importlib.util.spec_from_file_location("user_mod", fname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def nightly_iteration(n_obs_nights):
    """A generator for iterating over observations within a given night.

    Args:
        n_obs_nights (np.ndarray): The number of observations on each night.

    Yields:
        int: The night index.
        int: The index of the first observation for this night.
        int: The index of the last observation for this night + 1. The additional + 1 is so one can index the array via array[f:l].
    """
    n_nights = len(n_obs_nights)
    f, l = 0, n_obs_nights[0]
    for i in range(n_nights):
        yield i, f, l
        if i < n_nights - 1:
            f += n_obs_nights[i]
            l += n_obs_nights[i+1]

def list_diff(l1, l2):
    return [i for i in l1 + l2 if i not in l1 or i not in l2]


def get_stellar_rv(star_name):
    result = get_stellar_data(star_name)
    rv = result[0]["rv"]
    return rv


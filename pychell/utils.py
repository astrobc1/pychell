# Python default modules
from functools import reduce
import operator
from string import ascii_lowercase
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

def csscolor_to_rgba2(color, a=1.0):
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


class SpectralRegion:
    
    __slots__ = ['pixmin', 'pixmax', 'wavemin', 'wavemax', 'label', 'data_inds']
    
    def __init__(self, pixmin, pixmax, wavemin, wavemax, label=None):
        self.pixmin = pixmin
        self.pixmax = pixmax
        self.wavemin = wavemin
        self.wavemax = wavemax
        self.label = label
        self.data_inds = np.arange(self.pixmin, self.pixmax + 1).astype(int)
        
    def __len__(self):
        return self.wavemax - self.wavemin
    
    def wave_len(self):
        return self.wavemax - self.wavemin
    
    def pix_len(self):
        return self.pixmax - self.pixmin + 1
        
    def pix_within(self, pixels, pad=0):
        good = np.where((pixels >= self.pixmin - pad) & (pixels <= self.pixmax + pad))[0]
        return good
        
    def wave_within(self, waves, pad=0):
        good = np.where((waves >= self.wavemin - pad) & (waves <= self.wavemax + pad))[0]
        return good
    
    def midwave(self):
        return self.wavemin + self.wave_len() / 2
    
    def midpix(self):
        return self.pixmin + self.pix_len() / 2
        
    def pix_per_wave(self):
        return (self.pixmax - self.pixmin) / (self.wavemax - self.wavemin)
    
    def __repr__(self):
        s = "Pix: (" + str(self.pixmin) + ", " + str(self.pixmax) + ") Wave: (" + str(self.wavemin) + ", " + str(self.wavemax) + ")"
        return s
    
def gendatestr(time=False):
    now = datetime.now()
    if time:
        dt_string = now.strftime("%Y%m%d_%H%M%S")
    else:
        dt_string = now.strftime("%Y%m%d")
    return dt_string

class SessionState:
    """Session State for Streamlit.
    """
    
    def __init__(self, fname=None, use_prev=True):
        
        # Load existing state
        if fname is not None and os.path.exists(fname) and use_prev:
            self = self.load(fname)
        # Start from new state
        else:
            self.data = {}
            if fname is None:
                self.fname = "state_" + gendatestr(time=True) + ".pkl"
            else:
                self.fname = fname
        
    def save(self):
        with open(self.fname, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def make_default_fname():
        fname = 'results_' + pcutils.gendatestr(True) + '.pkl'
        return fname
            
    def __setitem__(self, key, value):
        self.data[key] = value
        
    def __setattr__(self, key, value):
        if key == "fname":
            self.fname = value
        else:
            self.data[key] = value
    
    def __getitem__(self, key):
        return self.data[key]
        
    @staticmethod
    def load(fname):
        with open(fname, 'rb') as f:
            d = pickle.load(f)
        return d
     
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


def list_diff(l1, l2):
    return [i for i in l1 + l2 if i not in l1 or i not in l2]
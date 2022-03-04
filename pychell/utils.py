import numpy as np
import sys
import time
import importlib
from itertools import chain, combinations
from barycorrpy.utils import get_stellar_data
from datetime import datetime
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

def list_diff(l1, l2):
    return [i for i in l1 + l2 if i not in l1 or i not in l2]


def get_stellar_rv(star_name):
    """Fetches the absolute stellar rv from a simbad recognizable name.

    Args:
        star_name (str): The name of the star, must be simbad recognizable. Any underscorse are replaced with a space.

    Returns:
        float: The stellar rv in m/s
    """
    result = get_stellar_data(star_name.replace("_", " "))
    rv = result[0]["rv"]
    return rv

def get_spec_module(spectrograph, spec_mod_func=None):
    spec_mod = getattr(importlib.import_module("pychell.data"), spectrograph.lower())
    if spec_mod_func is not None:
        spec_mod_func(spec_mod)
    return spec_mod


def flatten_jagged_list(x):
    x_out = np.array([], dtype=float)
    inds = []
    for i in range(len(x)):
        j_start = len(x_out)
        x_out = np.concatenate((x_out, x[i]))
        inds += [(i, j) for j in range(len(x[i]))]
    return x_out, inds

def get_utc_offset(site=None, lon=None):
    if site is not None:
        return int(site.lon.value / 15)
    else:
        return int(lon.value / 15)
    


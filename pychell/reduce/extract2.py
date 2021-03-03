# Python default modules
import os
import glob
import sys
import copy
import warnings

# Science / Math
import numpy as np
import scipy.interpolate
try:
    import torch
except:
    warnings.warn("Could not import pytorch!")
import scipy.signal
from astropy.io import fits

# LLVM
from numba import njit, jit, prange

# Graphics
import matplotlib.pyplot as plt

# Pychell modules
import pychell.utils as pcutils
import pychell.maths as pcmath
import pychell.reduce.calib as pccalib
import pychell.reduce.order_map as pcomap
import pychell.data as pcdata

# Optimize
import optimize.knowledge as optknow
import optimize.models as optmodels
import optimize.frameworks as optframeworks
import optimize.optimizers as optimizers
import optimize.scores as optscores
    
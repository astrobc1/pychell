# Python built in modules
import copy
from collections import OrderedDict
import glob # File searching
import os # Making directories
import importlib.util # importing other modules from files
import warnings # ignore warnings
import time # Time the code
import sys # sys utils
from sys import platform # plotting backend
import pdb # debugging
stop = pdb.set_trace

# Multiprocessing
from joblib import Parallel, delayed

# Graphics
import matplotlib # to set the backend
import matplotlib.pyplot as plt # Plotting

# Science/math
import scipy
from scipy import constants as cs # cs.c = speed of light in m/s
import numpy as np # Math, Arrays
import scipy.interpolate # Cubic interpolation, Akima interpolation

# llvm
from numba import njit, jit, prange

# User defined/pip modules
import pychell.rvs.model_components as pcmodelcomponents # the data objects
import pychell.maths as pcmath

def simple_rms(gp, forward_model, templates_dict):
    """Target function which returns the RMS and constraint. The RMS is weighted by bad pixels only (i.e., a binary mask). The constraint is used to force the LSF to be positive everywhere.

    Args:
        gp (Parameters): The Parameters object.
        forward_model (ForwardModel): The forwad model object
    Returns:
        (float): The rms.
        (float): The constraint.
    """

    # Generate the forward model
    wave_lr, model_lr = forward_model.build_full(gp, templates_dict)

    # Weights are just bad pixels
    weights = np.copy(forward_model.data.mask)
    
    # Compute rms ignoring bad pixels
    rms = pcmath.rmsloss(forward_model.data.flux, model_lr, weights=weights, flag_worst=forward_model.flag_n_worst_pixels)
    cons = np.nanmin(forward_model.models_dict['lsf'].build(gp))

    # Return rms and constraint
    return rms, cons


def weighted_data_flux(gp, forward_model, templates_dict):
    """Target function which returns the RMS and constraint. The RMS is weighted by bad pixels and the provided flux uncertainties. The constraint is used to force the LSF to be positive everywhere.

    Args:
        gp (Parameters): The Parameters object.
        forward_model (ForwardModel): The forwad model object
    Returns:
        (float): The rms.
        (float): The constraint.
    """
    # Generate the forward model
    wave_lr, model_lr = forward_model.build_full(gp, templates_dict)
    
    # Build weights from flux uncertainty
    weights = 1 / forward_model.data.flux_unc**2 * forward_model.data.mask

    # RMS and cons
    rms = pcmath.rmsloss(forward_model.data.flux, model_lr, weights=weights, flag_worst=forward_model.flag_n_worst_pixels)
    cons = np.nanmin(forward_model.models_dict['lsf'].build(gp)) # Ensure LSF is greater than zero

    # Return rms and constraint
    return wrms, cons
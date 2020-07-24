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

def simple_rms(gp, forward_model, iter_index):
    """Target function which returns the RMS and constraint. The RMS is weighted by bad pixels only (i.e., a binary mask). The constraint is used to force the LSF to be positive everywhere.

    Args:
        gp (Parameters): The Parameters object.
        forward_model (ForwardModel): The forwad model object
        iter_index (int): The iteration to generate RVs from.
    Returns:
        (float): The rms.
        (float): The constraint.
    """

    # Generate the forward model
    wave_lr, model_lr = forward_model.build_full(gp, iter_index)

    # Weights are just bad pixels
    weights = np.copy(forward_model.data.badpix)

    # Differences
    #diffs2 = (np.log(forward_model.data.flux) - np.log(model_lr))**2
    diffs2 = (forward_model.data.flux - model_lr)**2
    good = np.where(np.isfinite(diffs2) & (weights > 0))[0]
    if good.size < 100:
        return 1, -1
    residuals2 = diffs2[good]
    weights = weights[good]

    # Taper the ends
    left_taper = np.array([0.2, 0.4, 0.6, 0.8])
    right_taper = np.array([0.8, 0.6, 0.4, 0.2])

    residuals2[:4] *= left_taper
    residuals2[-4:] *= right_taper

    # Ignore worst 20 pixels
    ss = np.argsort(residuals2)
    weights[ss[-1*forward_model.flag_n_worst_pixels:]] = 0
    residuals2[ss[-1*forward_model.flag_n_worst_pixels:]] = np.nan
    
    # Compute rms ignoring bad pixels
    rms = (np.nansum(residuals2 * weights) / np.nansum(weights))**0.5
    cons = np.nanmin(forward_model.models_dict['lsf'].build(gp)) >= 0 # Ensure LSF is >= zero

    # Return rms and constraint
    return rms, cons


def weighted_data_flux(gp, forward_model, iter_index):
    """Target function which returns the RMS and constraint. The RMS is weighted by bad pixels and the provided flux uncertainties. The constraint is used to force the LSF to be positive everywhere.

    Args:
        gp (Parameters): The Parameters object.
        forward_model (ForwardModel): The forwad model object
        iter_index (int): The iteration to generate RVs from.
    Returns:
        (float): The rms.
        (float): The constraint.
    """
    # Generate the forward model
    wave_lr, model_lr = forward_model.build_full(gp, iter_index)
    
    # Build weights from flux uncertainty
    weights = 1 / forward_model.data.flux_unc**2 * forward_model.data.badpix

    # weighted RMS
    wdiffs2 = (forward_model.data.flux - model_lr)**2 * weights
    good = np.where(np.isfinite(wdiffs2) & (weights > 0))[0]
    if good.size < forward_model.data.flux.size * 0.1:
        return 1, -1
    wresiduals2 = wdiffs2[good]
    weights = weights[good]

    # Taper the ends
    left_taper = np.array([0.2, 0.4, 0.6, 0.8])
    right_taper = np.array([0.8, 0.6, 0.4, 0.2])

    wresiduals2[:4] *= left_taper
    wresiduals2[-4:] *= right_taper

    # Ignore worst 20 pixels
    ss = np.argsort(wresiduals2)
    weights[ss[-1*forward_model.flag_n_worst_pixels:]] = 0
    wresiduals2[ss[-1*forward_model.flag_n_worst_pixels:]] = np.nan
    
    # Compute weighted rms
    wrms = (np.nansum(wresiduals2) / np.nansum(weights))**0.5
    cons = np.nanmin(forward_model.models_dict['lsf'].build(gp)) # Ensure LSF is greater than zero

    # Return rms and constraint
    return wrms, cons


def binary_tellmask(gp, forward_model, iter_index):
    
    """Target function which returns the RMS and constraint. The RMS is weighted by bad pixels and a binary telluric mask which flags regions of telluric absorption greater than 95 percent. The constraint is used to force the LSF to be positive everywhere.

    Args:
        gp (Parameters): The Parameters object.
        forward_model (ForwardModel): The forwad model object
        iter_index (int): The iteration to generate RVs from.
    Returns:
        (float): The rms.
        (float): The constraint.
    """
    # Generate the forward model
    wave_lr, model_lr = forward_model.build_full(gp, iter_index)
    
    # Build weights from flux uncertainty
    tell_flux = forward_model.models_dict['tellurics'].build(gp, forward_model.templates_dict['tellurics'], wave_lr)
    tell_cut = 0.3
    bad = np.where(tell_flux < tell_cut)[0]
    weights = np.copy(forward_model.data.badpix)
    if bad.size > 0:
        weights[bad] = 0

    # weighted RMS
    wdiffs2 = (forward_model.data.flux - model_lr)**2 * weights
    good = np.where(np.isfinite(wdiffs2) & (weights > 0))[0]
    wresiduals2 = wdiffs2[good]
    weights = weights[good]

    # Taper the ends
    left_taper = np.array([0.2, 0.4, 0.6, 0.8])
    right_taper = np.array([0.8, 0.6, 0.4, 0.2])

    wresiduals2[:4] *= left_taper
    wresiduals2[-4:] *= right_taper

    # Ignore worst 20 pixels
    ss = np.argsort(wresiduals2)
    weights[ss[-1*forward_model.flag_n_worst_pixels:]] = 0
    wresiduals2[ss[-1*forward_model.flag_n_worst_pixels:]] = np.nan
    
    # Compute weighted rms
    wrms = (np.nansum(wresiduals2) / np.nansum(weights))**0.5
    cons = np.nanmin(forward_model.models_dict['lsf'].build(gp)) # Ensure LSF is greater than zero

    # Return rms and constraint
    return wrms, cons


def binary_tellmask_ignore(gp, forward_model, iter_index):
    
    """Target function which returns the RMS and constraint. The RMS is weighted by bad pixels and a binary telluric mask which flags regions of telluric absorption greater than 95 percent. The constraint is used to force the LSF to be positive everywhere.

    Args:
        gp (Parameters): The Parameters object.
        forward_model (ForwardModel): The forwad model object
        iter_index (int): The iteration to generate RVs from.
    Returns:
        (float): The rms.
        (float): The constraint.
    """
    # Generate the forward model
    wave_lr, model_lr = forward_model.build_full(gp, iter_index)
    
    # Build weights from flux uncertainty
    tell_flux = forward_model.models_dict['tellurics'].build(gp, forward_model.templates_dict['tellurics'], wave_lr)
    tell_cut = 0.98
    bad = np.where(tell_flux < tell_cut)[0]
    weights = np.copy(forward_model.data.badpix)
    if bad.size > 0:
        weights[bad] = 0

    # weighted RMS
    wdiffs2 = (forward_model.data.flux - model_lr)**2 * weights
    good = np.where(np.isfinite(wdiffs2) & (weights > 0))[0]
    wresiduals2 = wdiffs2[good]
    weights = weights[good]

    # Taper the ends
    left_taper = np.array([0.2, 0.4, 0.6, 0.8])
    right_taper = np.array([0.8, 0.6, 0.4, 0.2])

    wresiduals2[:4] *= left_taper
    wresiduals2[-4:] *= right_taper

    # Ignore worst 20 pixels
    ss = np.argsort(wresiduals2)
    weights[ss[-1*forward_model.flag_n_worst_pixels:]] = 0
    wresiduals2[ss[-1*forward_model.flag_n_worst_pixels:]] = np.nan
    
    # Compute weighted rms
    wrms = (np.nansum(wresiduals2) / np.nansum(weights))**0.5
    cons = np.nanmin(forward_model.models_dict['lsf'].build(gp)) # Ensure LSF is greater than zero

    # Return rms and constraint
    return wrms, cons


# Default Python modules
import os
import glob
import sys
from pdb import set_trace as stop


# Graphics
import matplotlib.pyplot as plt

# Science/Math
import numpy as np
from astropy.coordinates import Angle, SkyCoord
import astropy.units as units
from astropy.io import fits

# Clustering algorithms (DBSCAN)
import sklearn.cluster

# LLVM
from numba import jit, njit, prange

# Pychell modules
import pychell.utils as pcutils
import pychell.maths as pcmath
import pychell.reduce.data2d as pcdata

# Generate a master flat from a flat cube
def generate_master_flat(flats_cube, master_dark=None, master_bias=None):
    """Computes a median master flat field image from a subset of flat field images.

        Args:
            flats_cube (np.ndarray): The flats data cube, with shape=(n_flats, ny, nx)
            master_dark (np.ndarray): The master dark image (no dark subtraction if None)
            master_bias (np.ndarray): The master bias image (no bias subtraction if None).
        Returns:
            master_flat (np.ndarray): The median combined and corrected master flat image
        """
    for i in range(flats_cube.shape[0]):
        if master_bias is not None:
            flats_cube[i, :, :] -= master_bias
        if master_dark is not None:
            flats_cube[i, :, :] -= master_dark
        flats_cube[i, :, :] /= pcmath.weighted_median(flats_cube[i, :, :], med_val=0.99)
        bad = np.where((flats_cube[i, :, :] < 0) | (flats_cube[i, :, :] > 100))
        if bad[0].size > 0:
            flats_cube[i, :, :][bad] = np.nan
    master_flat = np.nanmedian(flats_cube, axis=0)
    bad = np.where((master_flat < 0) | (master_flat > 100))
    if bad[0].size > 0:
        master_flat[bad] = np.nan
    return master_flat

# Generate a master dark image
def generate_master_dark(darks_cube):
    """Generates a median master dark image given a darks image cube.

    Args:
        darks_cube (np.ndarray): The data cube of dark images, with shape=(n_bias, ny, nx)
    Returns:
        master_dark (np.ndarray): The master dark image.
    """
    master_dark = np.nanmedian(darks_cube, axis=0)
    return master_dark

# Generate a master bias image
def generate_master_bias(bias_cube):
    """Generates a median master bias image given a bias image cube.

    Args:
        bias_cube (np.ndarray): The data cube of bias images, with shape=(n_bias, ny, nx)
    Returns:
        master_bias (np.ndarray): The master bias image.
    """
    master_bias = np.nanmedian(bias_cube, axis=0)
    return master_bias

# Bias, Dark, Flat calibration
def standard_calibration(data_image, master_bias_image=None, master_flat_image=None, master_dark_image=None):
    """Performs standard bias, flat, and dark corrections.

    Args:
        data_image (np.ndarray): The data to calibrate.
        master_bias_image (np.ndarray): The master bias image (no bias subtraction if None).
        master_dark_image (np.ndarray): The master dark image (no dark subtraction if None).
        master_flat_image (np.ndarray): The master flat image (no flat correction if None).
    Returns:
        data_image (np.ndarray): The corrected data image.
    """
    
    # Bias correction
    if master_bias_image is not None:
        data_image -= master_bias_image
        
    # Dark correction
    if master_dark_image is not None:
        data_image -= master_dark_image
        
    # Flat division
    if master_flat_image is not None:
        data_image /= master_flat_image

    return data_image
# Default Python modules
import os

# Graphics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

# Science/Math
import numpy as np
import astropy.units as units
import scipy.interpolate
from astropy.io import fits

# Clustering algorithms (DBSCAN)
import sklearn.cluster
import scipy.signal

# LLVM
from numba import jit, njit, prange

# Pychell modules
import pychell.maths as pcmath
import pychell.spectralmodeling.data as pcspecdata

def gen_master_calib_images(data, do_bias, do_dark, do_flat, flat_percentile=None):
    
    # Bias image (only 1)
    if do_bias:
        gen_master_bias(data['master_bias'][0])
        
    # Master Dark images
    if do_dark:
        print('Creating Master Dark(s) ...', flush=True)
        for mdark in data['master_darks']:
            mdark_image = gen_master_dark(mdark, do_bias)
            mdark.save(mdark_image)
        
    # Flat field images
    if do_flat:
        print('Creating Master Flat(s) ...', flush=True)
        for mflat in data['master_flats']:
            mflat_image = gen_master_flat(mflat, do_bias, do_dark, flat_percentile)
            mflat.save(mflat_image)

    # Misc.
    if "master_fiber_flats" in data:
        for mfiberflat in data['master_fiber_flats']:
            mfiberflat_image = gen_master_fiber_flat(mfiberflat, do_bias, do_dark, do_flat)
            mfiberflat.save(mfiberflat_image)

def gen_master_flat(master_flat, do_bias, do_dark, flat_percentile=0.5):
    
    # Generate a data cube
    flats_cube = pcspecdata.Echellogram.generate_cube(master_flat.group)

    # Median crunch
    master_flat_image = np.nanmedian(flats_cube, axis=0)

    # Dark and Bias subtraction
    if do_bias:
        master_bias_image = master_flat.master_bias.parse_image()
        master_flat_image -= master_bias_image
    if do_dark:
        master_dark_image = master_flat.master_dark.parse_image()
        master_flat_image -= master_dark_image

    # Normalize
    master_flat_image /= pcmath.weighted_median(master_flat_image, percentile=flat_percentile)
    
    # Flag obvious bad pixels
    bad = np.where((master_flat_image <= flat_percentile*0.01) | (master_flat_image > flat_percentile * 100))
    if bad[0].size > 0:
        master_flat_image[bad] = np.nan
        
    # Return
    return master_flat_image

def gen_master_dark(master_dark, do_bias):
    """Computes a median master flat field image from a subset of flat field images. Dark and bias subtraction is also performed if set.

        Args:
            group (list): The list of DarkImages.
            bias_subtraction (bool): Whether or not to perform bias subtraction. If so, each dark must have a master_bias attribute.
        Returns:
            master_dark (np.ndarray): The median combined master dark image
    """
    # Generate a data cube
    n_darks = len(master_dark.group)
    darks_cube = pcspecdata.Echellogram.generate_cube(master_dark.group)

    # Median crunch
    master_dark_image = np.nanmedian(darks_cube, axis=0)

    # Bias subtraction
    if do_bias:
        master_bias_image = master_dark.master_bias.parse_image()
        master_dark_image -= master_bias_image
    
    # Flag obvious bad pixels
    bad = np.where(master_dark_image < 0)
    if bad[0].size > 0:
        master_dark_image[bad] = np.nan
        
    # Return
    return master_dark_image

def gen_master_bias(master_bias):
    """Generates a median master bias image.

    Args:
        group (list): The list of BiasImages.
    Returns:
        master_bias (np.ndarray): The master bias image.
    """
    bias_cube = group[0].generate_cube(master_bias.group)
    master_bias_image = np.nanmedian(bias_cube, axis=0)
    return master_bias_image
    
def gen_master_fiber_flat(master_fiber_flat, do_bias, do_dark, do_flat):

    # Generate a data cube
    n_fiber_flats = len(master_fiber_flat.group)
    fiber_flats_cube = pcspecdata.Echellogram.generate_cube(master_fiber_flat.group)

    # Median crunch
    master_fiber_flat_image = np.nanmedian(fiber_flats_cube, axis=0)

    # Dark and Bias subtraction
    if do_bias:
        master_bias_image = master_fiber_flat.master_bias.parse_image()
        master_fiber_flat_image -= master_bias_image
    if do_dark:
        master_dark_image = master_fiber_flat.master_dark.parse_image()
        master_fiber_flat_image -= master_dark_image
    if do_flat:
        master_flat_image = master_fiber_flat.master_flat.parse_image()
        master_fiber_flat_image /= master_flat_image
    
    # Flag obvious bad pixels
    bad = np.where(master_fiber_flat_image <= 0)
    if bad[0].size > 0:
        master_fiber_flat_image[bad] = np.nan
        
    # Return
    return master_fiber_flat_image


def pre_calibrate(data, data_image, do_bias, do_dark, do_flat):
    
    # Bias correction
    if do_bias:
        master_bias_image = data.master_bias.parse_image()
        data_image -= master_bias_image
        
    # Dark correction
    if do_dark:
        master_dark_image = data.master_dark.parse_image()
        data_image -= master_dark_image
        
    # Flat division
    if do_flat:
        master_flat_image = data.master_flat.parse_image()
        data_image /= master_flat_image